"""Immutable content-addressed text artifacts confined to the harness directory."""
import hashlib
import os
from pathlib import Path
import re
import tempfile


class ArtifactStore:
    def __init__(self, directory):
        self.directory = Path(directory).absolute()

    def check(self):
        if self.directory.resolve() != self.directory:
            raise ValueError('Artifact directory cannot be a symlink')

    def put(self, text):
        self.check()
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        data = text.encode('utf-8')
        key = hashlib.sha256(data).hexdigest() + '.txt'
        path = self.directory / key
        with tempfile.NamedTemporaryFile(dir=self.directory, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
        try:
            try:
                os.link(temporary, path)
            except FileExistsError:
                if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() + '.txt' != key:
                    raise ValueError('Artifact integrity check failed')
        finally:
            temporary.unlink(missing_ok=True)
        return key

    def read(self, key, offset=0, max_bytes=4000):
        if not isinstance(key, str) or not re.fullmatch(r'[a-f0-9]{64}\.txt', key):
            raise ValueError('Invalid artifact ID')
        if type(offset) is not int or offset < 0 or type(max_bytes) is not int or not 100 <= max_bytes <= 12000:
            raise ValueError('Use a nonnegative byte offset and max_bytes between 100 and 12000')
        self.check()
        path = self.directory / key
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, 'rb') as stream:
            stream.seek(offset)
            data = stream.read(max_bytes + 1)
        chunk = data[:max_bytes]
        return {'artifact_id': key, 'text': chunk.decode('utf-8', errors='replace'),
                'offset': offset, 'next_offset': offset + len(chunk), 'eof': len(data) <= max_bytes}
