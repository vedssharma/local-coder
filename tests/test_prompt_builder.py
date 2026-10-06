"""Tests for prompt_builder.py — system/user message construction."""



import prompt_builder


# ---------------------------------------------------------------------------
# _load_context_md
# ---------------------------------------------------------------------------

class TestLoadContextMd:
    def test_returns_none_when_no_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert prompt_builder._load_context_md() is None

    def test_returns_content_when_file_exists(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "CONTEXT.md").write_text("Project overview here.")
        result = prompt_builder._load_context_md()
        assert result == "Project overview here."

    def test_truncates_large_content(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        large = "x" * 5000
        (tmp_path / "CONTEXT.md").write_text(large)
        result = prompt_builder._load_context_md()
        assert len(result) < 5000
        assert result.endswith("[... truncated ...]")

    def test_does_not_truncate_exactly_at_limit(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        content = "y" * 4000  # exactly at limit
        (tmp_path / "CONTEXT.md").write_text(content)
        result = prompt_builder._load_context_md()
        assert result == content

    def test_returns_none_on_read_error(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "CONTEXT.md").write_text("some content")

        original_open = open

        def bad_open(path, *args, **kwargs):
            if "CONTEXT.md" in str(path):
                raise PermissionError("denied")
            return original_open(path, *args, **kwargs)

        monkeypatch.setattr("builtins.open", bad_open)
        assert prompt_builder._load_context_md() is None


# ---------------------------------------------------------------------------
# build_system_message
# ---------------------------------------------------------------------------

class TestBuildSystemMessage:
    def test_returns_system_role(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        msg = prompt_builder.build_system_message()
        assert msg["role"] == "system"

    def test_content_is_string(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        msg = prompt_builder.build_system_message()
        assert isinstance(msg["content"], str)
        assert len(msg["content"]) > 0

    def test_includes_tool_instructions(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        msg = prompt_builder.build_system_message()
        content = msg["content"]
        assert "read" in content
        assert "edit" in content
        assert "bash" in content

    def test_injects_context_md_when_present(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "CONTEXT.md").write_text("My project is about testing.")
        msg = prompt_builder.build_system_message()
        assert "My project is about testing." in msg["content"]
        assert "<context>" in msg["content"]

    def test_no_context_tag_without_context_md(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        msg = prompt_builder.build_system_message()
        assert "<context>" not in msg["content"]


# ---------------------------------------------------------------------------
# build_edit_system_message
# ---------------------------------------------------------------------------

class TestBuildEditSystemMessage:
    def test_returns_system_role(self):
        msg = prompt_builder.build_edit_system_message()
        assert msg["role"] == "system"

    def test_mentions_read_and_write(self):
        msg = prompt_builder.build_edit_system_message()
        content = msg["content"]
        assert "read" in content
        assert "edit" in content

    def test_no_tool_injection(self):
        # Edit system message should NOT contain context from CONTEXT.md
        msg = prompt_builder.build_edit_system_message()
        assert "<context>" not in msg["content"]


# ---------------------------------------------------------------------------
# build_user_message
# ---------------------------------------------------------------------------

class TestBuildUserMessage:
    def test_plain_prompt_no_files(self):
        msg = prompt_builder.build_user_message("Hello", {})
        assert msg["role"] == "user"
        assert msg["content"] == "Hello"

    def test_none_file_contents(self):
        msg = prompt_builder.build_user_message("Hello", None)
        assert msg["role"] == "user"
        assert msg["content"] == "Hello"

    def test_single_file_injected(self):
        msg = prompt_builder.build_user_message(
            "Explain this code",
            {"main.py": "print('hello')"}
        )
        assert msg["role"] == "user"
        assert "main.py" in msg["content"]
        assert "print('hello')" in msg["content"]
        assert "Explain this code" in msg["content"]

    def test_multiple_files_all_injected(self):
        files = {"a.py": "# a", "b.py": "# b"}
        msg = prompt_builder.build_user_message("Compare", files)
        assert "a.py" in msg["content"]
        assert "b.py" in msg["content"]
        assert "# a" in msg["content"]
        assert "# b" in msg["content"]

    def test_file_wrapped_in_tags(self):
        msg = prompt_builder.build_user_message("Check", {"foo.py": "pass"})
        assert "<file path='foo.py'>" in msg["content"]
        assert "</file>" in msg["content"]


# ---------------------------------------------------------------------------
# build_messages
# ---------------------------------------------------------------------------

class TestBuildMessages:
    def test_first_message_is_system(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        msgs = prompt_builder.build_messages("Hello", {})
        assert msgs[0]["role"] == "system"

    def test_last_message_is_user(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        msgs = prompt_builder.build_messages("Hello", {})
        assert msgs[-1]["role"] == "user"

    def test_history_inserted_between_system_and_user(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        history = [
            {"role": "user", "content": "prev question"},
            {"role": "assistant", "content": "prev answer"},
        ]
        msgs = prompt_builder.build_messages("new question", {}, history=history)
        assert msgs[0]["role"] == "system"
        assert msgs[1] == history[0]
        assert msgs[2] == history[1]
        assert msgs[-1]["role"] == "user"
        assert msgs[-1]["content"] == "new question"

    def test_no_history_produces_two_messages(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        msgs = prompt_builder.build_messages("Q", {})
        assert len(msgs) == 2
