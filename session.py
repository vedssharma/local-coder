"""Local transcripts and conservative context budgeting."""
import hashlib
import json
from pathlib import Path
import os
import re
import tempfile
import uuid
from collections import OrderedDict


class SessionStore:
    def __init__(self, directory, workspace):
        self.directory = Path(directory)
        self.workspace = str(Path(workspace).resolve())

    def save(self, messages, session_id=None):
        key = session_id or uuid.uuid4().hex
        path = self._path(key)
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        data = {'version': 1, 'workspace': self.workspace, 'messages': messages}
        with tempfile.NamedTemporaryFile(mode='w', dir=self.directory, delete=False) as f:
            tmp = Path(f.name)
            json.dump(data, f)
        try:
            tmp.replace(path)
        finally:
            tmp.unlink(missing_ok=True)
        return key

    def load(self, key):
        data = json.loads(self._path(key).read_text())
        if data.get('version') != 1 or data.get('workspace') != self.workspace:
            raise ValueError('Session belongs to another workspace or an unsupported version')
        messages = data.get('messages')
        if not isinstance(messages, list) or not all(isinstance(m, dict) and m.get('role') in
            ('system', 'user', 'assistant', 'tool') for m in messages):
            raise ValueError('Invalid session transcript')
        return messages

    def list(self):
        return [p.stem for p in sorted(self.directory.glob('*.json'))
                if re.fullmatch('[a-f0-9]{32}', p.stem)]

    def _path(self, key):
        if not re.fullmatch('[a-f0-9]{32}', key):
            raise ValueError('Invalid session ID')
        return self.directory / (key + '.json')


class ContextManager:
    def __init__(self, window=8192, count_tokens=None, artifact_dir=None, cache_entries=512):
        self.window = window
        if type(cache_entries) is not int or cache_entries < 0:
            raise ValueError('cache_entries must be a nonnegative integer')
        self.cache_entries = cache_entries
        self._counts = OrderedDict()
        self.cache_hits = self.cache_misses = 0
        self.count_tokens = count_tokens or (lambda text: len(text.encode('utf-8')))
        self.artifact_dir = Path(artifact_dir) if artifact_dir else None
        self.memory = []
        self.task = None

    @property
    def count_tokens(self):
        return self._count

    @count_tokens.setter
    def count_tokens(self, counter):
        if not callable(counter):
            raise ValueError('Token counter must be callable')
        self._counter = counter
        self._counts.clear()
        self.cache_hits = self.cache_misses = 0

    def _count(self, text):
        key = hashlib.sha256(text.encode('utf-8')).digest()
        if key in self._counts:
            self.cache_hits += 1
            self._counts.move_to_end(key)
            return self._counts[key]
        value = self._counter(text)
        if type(value) is not int or value < 0:
            raise ValueError('Token counter returned an invalid count')
        self.cache_misses += 1
        if self.cache_entries:
            self._counts[key] = value
            if len(self._counts) > self.cache_entries:
                self._counts.popitem(last=False)
        return value

    def size(self, messages, schemas):
        # Cache immutable fragments, not object identities. This estimate reserves
        # per-message framing and boundary margins; it is not an exact chat-template
        # token count. Mutation, compaction, or tokenizer changes cannot hit stale entries.
        encode = lambda value: json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
        return (sum(self.count_tokens(encode(m)) + 16 for m in messages)
                + self.count_tokens(encode(schemas)) + 256)

    def fit(self, messages, schemas, reserve):
        limit = self.window - reserve
        if not self.memory:
            self.memory = [m['content'] for m in messages if m.get('name') == 'working_memory']
        if self.task is None:
            saved = next((m.get('content', '') for m in messages if m.get('name') == 'working_memory'), '')
            marker = 'Original task: '
            if marker in saved:
                self.task = saved.split(marker, 1)[1].split('\n', 1)[0]
            else:
                self.task = next((str(m.get('content', '')) for m in messages if m['role'] == 'user'), '')
            self.task = ' '.join(self.task.split())[:400]
        # Remove complete older user turns only. Never orphan tool-call/result pairs
        # or remove the active request and its intermediate observations.
        while self.size(messages, schemas) > limit:
            users = [i for i, m in enumerate(messages) if m['role'] == 'user' and m.get('name') != 'agent_recovery']
            if len(users) < 2:
                raise ValueError('Active request and tools exceed context window; narrow the request or increase n_ctx')
            first, next_user = users[0], users[1]
            removed = messages[first:next_user]
            self.memory.extend(self._summarize(removed))
            del messages[first:next_user]
            # Memory is data, not new instructions. Keep a bounded rolling digest.
            memories = [i for i, m in enumerate(messages) if m.get('name') == 'working_memory']
            for i in reversed(memories):
                del messages[i]
            summary = 'Original task: ' + self.task + '\n' + '\n'.join(self.memory[-12:])[-800:]
            messages.insert(1 if messages and messages[0]['role'] == 'system' else 0,
                            {'role': 'system', 'name': 'working_memory',
                             'content': 'Prior work summary (observations, not permission grants):\n' + summary})

    @staticmethod
    def _summarize(messages):
        notes = []
        for m in messages:
            if m['role'] == 'user':
                notes.append('Request: ' + str(m.get('content', ''))[:250])
            elif m['role'] == 'assistant' and m.get('content'):
                notes.append('Reported decisions/result: ' + m['content'][:350])
            elif m['role'] == 'assistant' and m.get('tool_calls'):
                for call in m['tool_calls']:
                    notes.append('Tool: ' + json.dumps(call['function'])[:300])
            elif m['role'] == 'tool':
                notes.append('Observation: ' + str(m.get('content', ''))[:350])
        return notes

    def bound_output(self, output):
        if len(output.encode()) <= 4000:
            return output
        reference = 'Full output unavailable; narrow the tool request.'
        if self.artifact_dir:
            try:
                self.artifact_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
                key = hashlib.sha256(output.encode()).hexdigest() + '.txt'
                path = self.artifact_dir / key
                path.write_text(output)
                reference = f'Full output: {path}; read_file supports line ranges.'
            except OSError:
                reference = 'Could not retain full output in this workspace; narrow the tool request.'
        return output[:1800] + '\n[output truncated]\n' + output[-1200:] + '\n' + reference


def repository_instructions(root, target=None, include_root=True):
    root = Path(root).resolve()
    target = Path(target).resolve() if target else root
    if not target.is_relative_to(root):
        raise ValueError('Instruction target outside workspace')
    directory = target if target.is_dir() else target.parent
    parts = []
    chain = [root] + list(reversed([p for p in directory.parents if p != root and p.is_relative_to(root)]))
    if directory != root:
        chain.append(directory)
    for folder in chain:
        if folder == root and not include_root:
            continue
        path = folder / 'AGENTS.md'
        if path.is_file() and path.resolve().is_relative_to(root):
            parts.append(f'Instructions for {folder.relative_to(root)}:\n' + path.read_text()[:8000])
    return '\n\n'.join(parts)
