"""Content-addressed artifacts under .local-coder/artifacts."""
import pytest

from artifact_store import ArtifactStore


def test_put_and_page_through_an_artifact(tmp_path):
    store = ArtifactStore(tmp_path / 'artifacts')
    key = store.put('a' * 250)
    assert store.put('a' * 250) == key  # Identical content is stored once.
    first = store.read(key, max_bytes=200)
    assert (first['text'], first['next_offset'], first['eof']) == ('a' * 200, 200, False)
    rest = store.read(key, offset=first['next_offset'], max_bytes=200)
    assert (rest['text'], rest['eof']) == ('a' * 50, True)


def test_reads_are_validated(tmp_path):
    store = ArtifactStore(tmp_path / 'artifacts')
    key = store.put('text')
    with pytest.raises(ValueError, match='Invalid artifact ID'):
        store.read('../secret.txt')
    for offset, max_bytes in ((-1, 200), (0, 50), (0, 20000), ('0', 200)):
        with pytest.raises(ValueError, match='nonnegative byte offset'):
            store.read(key, offset=offset, max_bytes=max_bytes)


def test_symlinked_directories_and_tampered_files_are_refused(tmp_path):
    (tmp_path / 'real').mkdir()
    (tmp_path / 'link').symlink_to(tmp_path / 'real')
    with pytest.raises(ValueError, match='cannot be a symlink'):
        ArtifactStore(tmp_path / 'link').put('text')
    store = ArtifactStore(tmp_path / 'artifacts')
    key = store.put('original')
    (tmp_path / 'artifacts' / key).write_text('tampered')
    with pytest.raises(ValueError, match='integrity'):
        store.put('original')
