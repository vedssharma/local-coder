"""Tests for helpers.py — @file reference parsing."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

import helpers


class TestParseFileReferences:
    def test_no_references_returns_unchanged_prompt(self, tmp_path):
        prompt = "What is the time complexity of quicksort?"
        returned_prompt, files = helpers.parse_file_references(prompt)
        assert returned_prompt == prompt
        assert files == {}

    def test_single_valid_reference(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        target = tmp_path / "hello.txt"
        target.write_text("hello world")
        prompt = f"Explain @{target}"
        _, files = helpers.parse_file_references(prompt)
        assert str(target) in files
        assert files[str(target)] == "hello world"

    def test_multiple_valid_references(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        f1 = tmp_path / "a.py"
        f2 = tmp_path / "b.py"
        f1.write_text("# file a")
        f2.write_text("# file b")
        prompt = f"Compare @{f1} and @{f2}"
        _, files = helpers.parse_file_references(prompt)
        assert len(files) == 2
        assert files[str(f1)] == "# file a"
        assert files[str(f2)] == "# file b"

    def test_missing_file_emits_warning_and_skips(self, tmp_path, capsys):
        prompt = "@/nonexistent/missing.txt explain this"
        _, files = helpers.parse_file_references(prompt)
        assert files == {}
        captured = capsys.readouterr()
        assert "Warning" in captured.err

    def test_directory_reference_emits_warning_and_skips(self, tmp_path, capsys):
        prompt = f"@{tmp_path} what is this"
        _, files = helpers.parse_file_references(prompt)
        assert files == {}
        captured = capsys.readouterr()
        assert "Warning" in captured.err

    def test_original_prompt_returned_unchanged(self, tmp_path):
        prompt = "No refs here, just a question?"
        returned_prompt, _ = helpers.parse_file_references(prompt)
        assert returned_prompt == prompt

    def test_unreadable_file_emits_warning(self, tmp_path, monkeypatch, capsys):
        target = tmp_path / "locked.txt"
        target.write_text("secret")

        original_open = open

        def bad_open(path, *args, **kwargs):
            if str(target) in str(path):
                raise PermissionError("access denied")
            return original_open(path, *args, **kwargs)

        monkeypatch.setattr("builtins.open", bad_open)
        prompt = f"@{target}"
        _, files = helpers.parse_file_references(prompt)
        assert files == {}
        captured = capsys.readouterr()
        assert "Warning" in captured.err

    def test_reference_without_at_not_included(self, tmp_path):
        prompt = "just a plain path /tmp/something.txt"
        _, files = helpers.parse_file_references(prompt)
        assert files == {}

    def test_at_sign_followed_by_whitespace_not_included(self):
        prompt = "email @ user"
        _, files = helpers.parse_file_references(prompt)
        # "@" followed by a space produces no filename
        assert files == {}
