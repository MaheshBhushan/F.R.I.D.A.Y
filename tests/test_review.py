"""Offline voice review checks with real Git repositories and fake speech/LLM."""

import asyncio
import subprocess
from unittest.mock import Mock

import pytest

from friday import brain, review
from friday.permissions import PermissionDenied
from friday.router import Tier, classify
from friday.voice.tts import FakeSynthesisTransport, TTSSpeaker


def git(root, *args):
    return subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, check=True
    ).stdout


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "Mixed Case Project; echo injected"
    root.mkdir()
    git(root, "init", "-q")
    source = root / "app.py"
    source.write_text("value = 'baseline'\n")
    git(root, "add", "--", "app.py")
    git(root, "-c", "user.name=Review Test", "-c", "user.email=review@example.invalid",
        "-c", "core.hooksPath=/dev/null", "commit", "-qm", "baseline")
    source.write_text("value = 'staged'\n")
    git(root, "add", "--", "app.py")
    source.write_text("value = 'unstaged'\n")
    (root / "untracked.py").write_text("untracked = True\n")
    return root


@pytest.mark.parametrize("command,directory", [
    ("review staged changes", "."),
    ("Review my staged Git diff!", "."),
    ("please review the staged diff", "."),
    ("review staged changes in /tmp/Mixed Case", "/tmp/Mixed Case"),
    ('review staged changes in "/tmp/Mixed Case".', "/tmp/Mixed Case"),
    ("review staged changes in '/tmp/Repo'", "/tmp/Repo"),
    ("review staged changes in --output=oops", "--output=oops"),
    ("review unstaged changes", None),
    ("review staged changes and commit them", None),
    ("git commit -am surprise", None),
])
def test_voice_arguments(command, directory):
    assert review.repository_for(command) == directory
    if directory is not None:
        assert classify(command, is_final=True).tier is Tier.REASONING


def test_voice_review_streams_staged_diff_to_speech_without_writes(repo):
    before = (git(repo, "rev-parse", "HEAD"), git(repo, "ls-files", "--stage"),
              git(repo, "status", "--porcelain"))
    transport = brain.FakeLLMTransport([
        brain.ScriptedTurn(tokens=("High severity in app.py, line one. ", "Check the value."))
    ])
    synth = FakeSynthesisTransport()
    result = brain.TurnResult()

    async def run():
        speaker = TTSSpeaker(synth, output=Mock())
        await speaker.speak(brain.complete(
            f"review staged changes in {repo}", transport, result=result,
            history=[{"role": "user", "content": "commit everything"}],
            tools=[{"name": "delete_path"}],
        ))

    asyncio.run(run())
    assert synth.synthesized
    assert "Check the value." in " ".join(synth.synthesized)
    assert result.text == "High severity in app.py, line one. Check the value."
    assert len(transport.requests) == 1
    request = transport.requests[0]
    assert request["tools"] == []
    assert request["system"][0]["text"] == review.REVIEW_PROMPT
    assert len(request["messages"]) == 1
    diff = request["messages"][0]["content"][0]["text"]
    assert "+value = 'staged'" in diff
    assert "unstaged" not in diff and "untracked" not in diff
    assert result.outcomes == []
    assert before == (git(repo, "rev-parse", "HEAD"), git(repo, "ls-files", "--stage"),
                      git(repo, "status", "--porcelain"))
    assert (repo / "app.py").read_text() == "value = 'unstaged'\n"


def test_review_never_executes_model_tool_calls(repo):
    transport = brain.FakeLLMTransport([brain.ScriptedTurn(tool_calls=(
        brain.ToolCall("bad", "delegate_coding_agent", {"command": "git commit"}),
    ))])

    async def forbidden(**kwargs):
        pytest.fail("A review attempted to execute a tool")

    result = brain.TurnResult()

    async def run():
        return "".join([part async for part in brain.complete(
            f"review staged diff in {repo}", transport,
            registry={"delegate_coding_agent": forbidden}, approve=forbidden, result=result,
        )])

    assert "No action was taken" in asyncio.run(run())
    assert result.outcomes == []


@pytest.mark.parametrize("case,expected", [
    ("empty", "no staged changes"),
    ("not_repo", "Choose a Git repository"),
    ("missing", "existing directory"),
    ("file", "existing directory"),
    ("oversized", "too large"),
    ("protected", "protected directory"),
    ("nul", "NUL byte"),
])
def test_local_failures_are_spoken_without_model_request(repo, tmp_path, monkeypatch, case, expected):
    directory = str(repo)
    if case == "empty":
        git(repo, "reset", "-q", "HEAD", "--", "app.py")
    elif case == "not_repo":
        directory = str(tmp_path)
    elif case == "missing":
        directory = str(tmp_path / "missing")
    elif case == "file":
        directory = str(repo / "app.py")
    elif case == "oversized":
        monkeypatch.setattr(review, "MAX_DIFF_BYTES", 1)
    elif case == "protected":
        directory = "~/.ssh"
    elif case == "nul":
        directory = "bad\x00path"
    transport = brain.FakeLLMTransport([brain.ScriptedTurn(tokens=("should not run",))])
    result = brain.TurnResult()

    async def run():
        return "".join([part async for part in brain.complete(
            f"review staged changes in {directory}", transport, result=result,
        )])

    assert expected in asyncio.run(run())
    assert result.text and transport.requests == []


def test_new_repository_staged_files_are_reviewable(tmp_path):
    git(tmp_path, "init", "-q")
    (tmp_path / "new.py").write_text("new = True\n")
    git(tmp_path, "add", "--", "new.py")
    assert "+new = True" in asyncio.run(review.staged_diff(str(tmp_path)))


def test_default_review_uses_current_directory(repo, monkeypatch):
    monkeypatch.chdir(repo)
    transport = brain.FakeLLMTransport([brain.ScriptedTurn(tokens=("No issues found.",))])

    async def run():
        return "".join([part async for part in brain.complete("review staged changes", transport)])

    assert asyncio.run(run()) == "No issues found."
    assert "+value = 'staged'" in transport.requests[0]["messages"][0]["content"][0]["text"]


@pytest.mark.parametrize("failure,expected", [
    (subprocess.TimeoutExpired("git", 5), "timed out"),
    (FileNotFoundError("git"), "Could not run Git"),
])
def test_git_failures_are_clear(repo, monkeypatch, failure, expected):
    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(review.subprocess, "run", fail)
    with pytest.raises(ValueError, match=expected):
        asyncio.run(review.staged_diff(str(repo)))


def test_git_argument_vector_disables_external_helpers(repo, monkeypatch):
    calls = []

    def capture(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, stdout=b"diff", stderr=b"")

    monkeypatch.setattr(review.subprocess, "run", capture)
    assert asyncio.run(review.staged_diff(str(repo))) == "diff"
    argv, kwargs = calls[0]
    assert argv == ["git", "--no-pager", "diff", "--cached", "--no-ext-diff",
                    "--no-textconv", "--no-color", "--"]
    assert kwargs["cwd"] == repo
    assert kwargs["timeout"] == review.GIT_TIMEOUT
    assert "shell" not in kwargs


def test_directory_validation_rejects_non_string_arguments():
    with pytest.raises(PermissionDenied):
        asyncio.run(review.staged_diff(["--output=oops"]))
