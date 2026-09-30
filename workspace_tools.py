"""Repository tools shared by the CLI and MCP entry points."""
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import threading
import time
import uuid


def schema(name, description, properties, required=()):
    return {'type': 'function', 'function': {'name': name, 'description': description,
        'parameters': {'type': 'object', 'properties': properties,
                       'required': list(required), 'additionalProperties': False}}}


STRING = {'type': 'string'}
INTEGER = {'type': 'integer', 'minimum': 1}
SCHEMAS = [
    schema('read_file', 'Read a bounded line range of a workspace file.',
           {'path': STRING, 'start_line': INTEGER, 'end_line': INTEGER}, ['path']),
    schema('list_directory', 'List workspace entries.', {'path': STRING}),
    schema('search_code', 'Search file contents using ripgrep; output contains line numbers.',
           {'pattern': STRING, 'path': STRING, 'glob': STRING}, ['pattern']),
    schema('apply_patch', 'Replace exactly one matching text block; empty old_text creates a new file only.',
           {'path': STRING, 'old_text': STRING, 'new_text': STRING}, ['path', 'old_text', 'new_text']),
    schema('run_command', 'Run an argv command in the workspace. Poll returned process_id until it exits.',
           {'argv': {'type': 'array', 'items': STRING, 'minItems': 1}, 'cwd': STRING,
            'timeout_seconds': {'type': 'integer', 'minimum': 1, 'maximum': 300}}, ['argv']),
    schema('poll_process', 'Get command output and exit status.', {'process_id': STRING}, ['process_id']),
    schema('cancel_process', 'Terminate a command process group.', {'process_id': STRING}, ['process_id']),
    schema('git_diff', 'Show tracked changes from HEAD and list untracked workspace files.', {}),
]


MODES = ('read-only', 'workspace-edit', 'execute')
READ_TOOLS = {'read_file', 'list_directory', 'search_code', 'git_diff'}
MCP_READ_TOOLS = {'read_text_file', 'read_multiple_files', 'directory_tree', 'get_file_info',
                  'list_allowed_directories', 'search_files', 'list_directory_with_sizes'}


class WorkspaceTools:
    def __init__(self, root=None, mcp_client=None, mode="read-only"):
        self.root = Path(root or os.getcwd()).resolve()
        if mode not in MODES:
            raise ValueError('Unknown permission mode')
        self.mode = mode
        self.mcp_client = mcp_client
        self.processes = {}
        self.is_connected = True

    def path(self, value='.'):
        p = (self.root / value).resolve()
        if not p.is_relative_to(self.root):
            raise ValueError('Path is outside the workspace')
        return p

    def authorize(self, name, args):
        if name not in self.tool_names:
            raise PermissionError(f'Tool {name} is unavailable in {self.mode} mode')
        for key in ('path', 'cwd'):
            if key in args:
                self.path(args[key])
        for value in args.get('paths', []):
            self.path(value)
        if name == 'apply_patch':
            relative = self.path(args['path']).relative_to(self.root)
            if any(part in ('.git', '.local-coder') for part in relative.parts):
                raise PermissionError('Harness metadata and Git internals cannot be edited')

    @property
    def tool_names(self):
        return {s['function']['name'] for s in self.get_openai_tool_schemas()}

    def get_openai_tool_schemas(self):
        extra = []
        if self.mcp_client and self.mcp_client.is_connected:
            names = {s['function']['name'] for s in SCHEMAS}
            extra = [s for s in self.mcp_client.get_openai_tool_schemas()
                     if s['function']['name'] not in names and s['function']['name'] in MCP_READ_TOOLS]
        allowed = READ_TOOLS | ({'apply_patch'} if self.mode != 'read-only' else set())
        if self.mode == 'execute':
            allowed |= {'run_command', 'poll_process', 'cancel_process'}
        return [s for s in SCHEMAS if s['function']['name'] in allowed] + extra

    def call_tool(self, name, args):
        try:
            self.authorize(name, args)
            if name == 'read_file':
                p = self.path(args['path'])
                start, end = args.get('start_line', 1), args.get('end_line', 200)
                if start < 1 or end < start or end - start > 1000:
                    raise ValueError('Use a range of at most 1001 lines')
                lines = []
                with p.open() as f:
                    for n, line in enumerate(f, 1):
                        if n > end:
                            break
                        if n >= start:
                            lines.append(f'{n}: {line}')
                from session import repository_instructions
                instructions = repository_instructions(self.root, p)
                result = ''.join(lines)[:32000] or '(empty file)'
                return result + ('\n\n' + instructions if instructions else '')
            if name == 'list_directory':
                return '\n'.join(p.name + ('/' if p.is_dir() else '')
                                 for p in sorted(self.path(args.get('path', '.')).iterdir())[:200])
            if name == 'search_code':
                command = ['rg', '-n', '--no-heading', '--color=never', '--max-count=50']
                if args.get('glob'):
                    command += ['--glob', args['glob']]
                command += ['--', args['pattern'], str(self.path(args.get('path', '.')))]
                result = self._capture(command)
                if result['exit_code'] == 1:
                    return 'No matches found.'
                return json.dumps(result)
            if name == 'apply_patch':
                p = self.path(args['path'])
                old, new = args['old_text'], args['new_text']
                before = p.read_text() if p.exists() else None
                mode = p.stat().st_mode if p.exists() else 0o644
                if not old:
                    with p.open('x') as f:
                        f.write(new)
                    updated = new
                else:
                    original = p.read_text()
                    before = original
                    if original.count(old) != 1:
                        raise ValueError('Expected text must match exactly once; reread the file')
                    updated = original.replace(old, new, 1)
                    # Atomic replacement, keeping executable mode.
                    with tempfile.NamedTemporaryFile(mode='w', dir=p.parent, delete=False) as f:
                        temp = Path(f.name)
                        f.write(updated)
                    try:
                        temp.chmod(p.stat().st_mode)
                        if p.read_text() != original:
                            raise ValueError('File changed during patch; reread it')
                        temp.replace(p)
                    finally:
                        temp.unlink(missing_ok=True)
                self._record_patch(p, before, updated, mode)
                return f'Patched {p.relative_to(self.root)}'
            if name == 'run_command':
                return json.dumps(self._start(args['argv'], self.path(args.get('cwd', '.')),
                                              args.get('timeout_seconds', 60)))
            if name in ('poll_process', 'cancel_process'):
                state = self.processes[args['process_id']]
                if name == 'cancel_process':
                    self._kill(state)
                return json.dumps(self._poll(args['process_id']))
            if name == 'git_diff':
                result = self._capture(['git', '--no-pager', '-c', 'core.fsmonitor=false', 'diff', '--no-ext-diff', '--no-textconv', 'HEAD', '--'])
                untracked = self._capture(['git', '-c', 'core.fsmonitor=false', 'ls-files', '--others', '--exclude-standard', '--exclude=.local-coder/', '-z'])
                result['untracked_files'] = [p for p in untracked['output'].split('\0') if p]
                return json.dumps(result)
            if self.mcp_client and self.mcp_client.is_connected:
                return self.mcp_client.call_tool(name, args)
            raise ValueError(f'Unknown tool: {name}')
        except (OSError, ValueError, KeyError, TypeError) as exc:
            return f'Error: {exc}'

    def _record_patch(self, path, before, after, mode):
        directory = self.path('.local-coder/undo')
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        record = {'path': str(path.relative_to(self.root)), 'before': before, 'after': after, 'mode': mode}
        key = f'{time.time_ns()}-{uuid.uuid4().hex}.json'
        with (directory / key).open('x') as f:
            os.chmod(f.name, 0o600)
            json.dump(record, f)

    def undo_last(self):
        if self.mode == 'read-only':
            raise PermissionError('Undo requires workspace-edit or execute mode')
        records = sorted(self.path('.local-coder/undo').glob('*.json'))
        if not records:
            return 'No harness edits to undo.'
        record_path = records[-1]
        if record_path.is_symlink():
            raise ValueError('Invalid undo record')
        record = json.loads(record_path.read_text())
        self.authorize('apply_patch', {'path': record['path']})
        path = self.path(record['path'])
        if not path.exists() or path.read_text() != record['after']:
            raise ValueError('File changed since the harness edit; undo refused to preserve your changes')
        if record['before'] is None:
            path.unlink()
        else:
            with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as f:
                temp = Path(f.name)
                f.write(record['before'])
            try:
                temp.chmod(record['mode'])
                if path.read_text() != record['after']:
                    raise ValueError('File changed during undo')
                temp.replace(path)
            finally:
                temp.unlink(missing_ok=True)
        record_path.unlink()
        return f'Undid harness edit to {record["path"]}'

    def _start(self, argv, cwd, timeout):
        if not isinstance(argv, list) or not argv or not all(isinstance(x, str) for x in argv):
            raise ValueError('argv must be a nonempty string array')
        if not 1 <= timeout <= 300:
            raise ValueError('timeout_seconds must be between 1 and 300')
        proc = subprocess.Popen(argv, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                start_new_session=True)
        key = uuid.uuid4().hex
        state = {'proc': proc, 'output': b'', 'deadline': time.monotonic() + timeout,
                 'timed_out': False, 'output_truncated': False, 'lock': threading.Lock()}
        self.processes[key] = state
        def drain():
            while chunk := proc.stdout.read1(4096):
                with state['lock']:
                    state['output_truncated'] |= len(state['output']) + len(chunk) > 32000
                    state['output'] = (state['output'] + chunk)[-32000:]
            proc.stdout.close()
        reader = threading.Thread(target=drain, daemon=True)
        state['reader'] = reader
        reader.start()
        def expire():
            if proc.poll() is None:
                state['timed_out'] = True
                self._kill(state)
        timer = threading.Timer(timeout, expire)
        timer.daemon = True
        timer.start()
        state['timer'] = timer
        return self._poll(key)

    def _poll(self, key):
        state = self.processes[key]
        code = state['proc'].poll()
        if code is not None:
            self._kill(state)  # A completed job must not leave background children.
            state['timer'].cancel()
            state['reader'].join(timeout=0.1)
        with state['lock']:
            output = state['output'].decode(errors='replace')
        return {'process_id': key, 'exit_code': code, 'running': code is None,
                'timed_out': state['timed_out'], 'output_truncated': state['output_truncated'], 'output': output}

    @staticmethod
    def _kill(state):
        try:
            os.killpg(state['proc'].pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        state['proc'].wait()

    def _capture(self, argv):
        result = self._start(argv, self.root, 30)
        state = self.processes[result['process_id']]
        state['proc'].wait()
        result = self._poll(result['process_id'])
        del self.processes[result['process_id']]
        return result

    def close(self):
        for state in self.processes.values():
            self._kill(state)
            state['timer'].cancel()
            state['reader'].join(timeout=0.5)
        if self.mcp_client:
            self.mcp_client.close()
