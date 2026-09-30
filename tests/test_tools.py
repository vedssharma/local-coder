"""Tests for tools.py — tool schemas and filesystem tool implementations."""

import os
from pathlib import Path

import pytest

import tools


# ---------------------------------------------------------------------------
# TOOL_SCHEMAS structure
# ---------------------------------------------------------------------------

class TestToolSchemas:
    def test_schema_list_has_four_entries(self):
        assert len(tools.TOOL_SCHEMAS) == 4

    def test_all_schemas_have_type_function(self):
        for schema in tools.TOOL_SCHEMAS:
            assert schema["type"] == "function"

    def test_expected_tool_names_present(self):
        names = {s["function"]["name"] for s in tools.TOOL_SCHEMAS}
        assert names == {"read_file", "list_directory", "search_files", "write_file"}

    def test_each_schema_has_description(self):
        for schema in tools.TOOL_SCHEMAS:
            assert schema["function"]["description"]

    def test_each_schema_has_parameters(self):
        for schema in tools.TOOL_SCHEMAS:
            params = schema["function"]["parameters"]
            assert params["type"] == "object"
            assert "properties" in params


# ---------------------------------------------------------------------------
# execute_tool dispatch
# ---------------------------------------------------------------------------

class TestExecuteTool:
    def test_dispatches_read_file(self, tmp_path):
        f = tmp_path / "hello.txt"
        f.write_text("hi")
        result = tools.execute_tool("read_file", {"path": str(f)}, root=tmp_path)
        assert result == "hi"

    def test_dispatches_list_directory(self, tmp_path):
        (tmp_path / "sub").mkdir()
        result = tools.execute_tool("list_directory", {"path": str(tmp_path)}, root=tmp_path)
        assert "sub/" in result

    def test_dispatches_search_files(self, tmp_path):
        f = tmp_path / "code.py"
        f.write_text("def hello(): pass")
        result = tools.execute_tool("search_files", {"pattern": "hello", "path": str(tmp_path)}, root=tmp_path)
        assert "hello" in result

    def test_dispatches_write_file(self, tmp_path):
        dest = tmp_path / "out.txt"
        result = tools.execute_tool("write_file", {"path": str(dest), "content": "written"}, root=tmp_path, mode="workspace-edit")
        assert "Patched" in result
        assert dest.read_text() == "written"

    def test_returns_error_for_unknown_tool(self):
        result = tools.execute_tool("fly_to_the_moon", {})
        assert "Unknown tool" in result
        assert "fly_to_the_moon" in result


# ---------------------------------------------------------------------------
# _read_file
# ---------------------------------------------------------------------------

class TestReadFile:
    def test_reads_content(self, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("sample content")
        assert tools._read_file(str(f)) == "sample content"

    def test_error_on_missing_file(self):
        result = tools._read_file("/nonexistent/path/file.txt")
        assert result.startswith("Error")
        assert "not found" in result

    def test_error_on_directory(self, tmp_path):
        result = tools._read_file(str(tmp_path))
        assert result.startswith("Error")
        assert "Not a file" in result

    def test_truncates_large_file(self, tmp_path):
        f = tmp_path / "big.txt"
        big_content = "A" * (tools.MAX_FILE_SIZE + 1000)
        f.write_text(big_content)
        result = tools._read_file(str(f))
        assert len(result) < len(big_content)
        assert "truncated" in result

    def test_does_not_truncate_file_within_limit(self, tmp_path):
        f = tmp_path / "small.txt"
        content = "B" * 100
        f.write_text(content)
        assert tools._read_file(str(f)) == content


# ---------------------------------------------------------------------------
# _list_directory
# ---------------------------------------------------------------------------

class TestListDirectory:
    def test_lists_files_and_dirs(self, tmp_path):
        (tmp_path / "file.py").write_text("")
        (tmp_path / "subdir").mkdir()
        result = tools._list_directory(str(tmp_path))
        assert "file.py" in result
        assert "subdir/" in result

    def test_trailing_slash_for_dirs_only(self, tmp_path):
        (tmp_path / "f.txt").write_text("")
        (tmp_path / "d").mkdir()
        result = tools._list_directory(str(tmp_path))
        lines = result.splitlines()
        assert "d/" in lines
        assert "f.txt" in lines

    def test_error_on_missing_directory(self):
        result = tools._list_directory("/nonexistent/dir")
        assert result.startswith("Error")
        assert "not found" in result

    def test_error_on_file_path(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("")
        result = tools._list_directory(str(f))
        assert result.startswith("Error")
        assert "Not a directory" in result

    def test_skips_skip_dirs(self, tmp_path):
        for name in tools.SKIP_DIRS:
            (tmp_path / name).mkdir()
        (tmp_path / "visible").mkdir()
        result = tools._list_directory(str(tmp_path))
        for name in tools.SKIP_DIRS:
            assert name not in result
        assert "visible/" in result

    def test_truncates_at_max_dir_entries(self, tmp_path):
        for i in range(tools.MAX_DIR_ENTRIES + 5):
            (tmp_path / f"file{i:04d}.txt").write_text("")
        result = tools._list_directory(str(tmp_path))
        assert "more entries" in result

    def test_defaults_to_current_directory(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "myfile.txt").write_text("")
        result = tools._list_directory(".")
        assert "myfile.txt" in result


# ---------------------------------------------------------------------------
# _search_files
# ---------------------------------------------------------------------------

class TestSearchFiles:
    def test_finds_matching_lines(self, tmp_path):
        (tmp_path / "a.py").write_text("def foo():\n    pass\n")
        result = tools._search_files("def foo", str(tmp_path))
        assert "def foo" in result

    def test_includes_file_path_and_line_number(self, tmp_path):
        (tmp_path / "b.py").write_text("hello world\n")
        result = tools._search_files("hello", str(tmp_path))
        assert "b.py" in result
        assert ":1:" in result

    def test_returns_no_matches_message(self, tmp_path):
        (tmp_path / "c.py").write_text("nothing here\n")
        result = tools._search_files("zzznomatch", str(tmp_path))
        assert result == "No matches found."

    def test_invalid_regex_returns_error(self, tmp_path):
        result = tools._search_files("[invalid(regex", str(tmp_path))
        assert result.startswith("Error")
        assert "regex" in result.lower()

    def test_file_glob_filter(self, tmp_path):
        (tmp_path / "match.py").write_text("target pattern\n")
        (tmp_path / "skip.txt").write_text("target pattern\n")
        result = tools._search_files("target", str(tmp_path), file_glob="*.py")
        assert "match.py" in result
        assert "skip.txt" not in result

    def test_caps_at_max_results(self, tmp_path):
        for i in range(tools.MAX_SEARCH_RESULTS + 10):
            (tmp_path / f"f{i}.txt").write_text("match\n")
        result = tools._search_files("match", str(tmp_path))
        assert "capped at" in result

    def test_case_insensitive_search(self, tmp_path):
        (tmp_path / "d.py").write_text("HELLO World\n")
        result = tools._search_files("hello world", str(tmp_path))
        assert "HELLO World" in result

    def test_skips_skip_dirs(self, tmp_path):
        skip = tmp_path / "__pycache__"
        skip.mkdir()
        (skip / "cached.py").write_text("should be ignored\n")
        result = tools._search_files("should be ignored", str(tmp_path))
        assert result == "No matches found."


# ---------------------------------------------------------------------------
# _write_file
# ---------------------------------------------------------------------------

class TestWriteFile:
    def test_writes_content(self, tmp_path):
        dest = tmp_path / "output.txt"
        result = tools._write_file(str(dest), "hello")
        assert "Successfully wrote" in result
        assert dest.read_text() == "hello"

    def test_creates_parent_directories(self, tmp_path):
        dest = tmp_path / "nested" / "deep" / "file.txt"
        tools._write_file(str(dest), "data")
        assert dest.exists()
        assert dest.read_text() == "data"

    def test_overwrites_existing_file(self, tmp_path):
        dest = tmp_path / "file.txt"
        dest.write_text("old content")
        tools._write_file(str(dest), "new content")
        assert dest.read_text() == "new content"

    def test_confirm_fn_called_with_path_and_content(self, tmp_path):
        dest = tmp_path / "f.txt"
        calls = []

        def confirm(path, content):
            calls.append((path, content))
            return True

        tools._write_file(str(dest), "data", confirm_fn=confirm)
        assert len(calls) == 1
        assert calls[0] == (str(dest), "data")

    def test_confirm_fn_returning_false_cancels_write(self, tmp_path):
        dest = tmp_path / "f.txt"
        result = tools._write_file(str(dest), "data", confirm_fn=lambda p, c: False)
        assert "cancelled" in result.lower()
        assert not dest.exists()

    def test_result_includes_character_count(self, tmp_path):
        dest = tmp_path / "f.txt"
        content = "abc"
        result = tools._write_file(str(dest), content)
        assert str(len(content)) in result
