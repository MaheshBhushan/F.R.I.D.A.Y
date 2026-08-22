"""The colouriser sits between the daemon and the user's eyes. Its job is to
add colour and change nothing else -- a reader that drops or mangles a line
hides exactly the output someone tailing the log is hunting for."""

import io

from friday.core import logfmt

STRIP = str.maketrans("", "", "")


def _plain(text: str) -> str:
    """Text with ANSI escapes removed."""
    out, i = [], 0
    while i < len(text):
        if text[i] == "\033":
            while i < len(text) and text[i] != "m":
                i += 1
            i += 1
            continue
        out.append(text[i])
        i += 1
    return "".join(out)


def test_content_survives_colouring():
    line = '22:41:07.412 [wake] model=alexa score=0.812'
    assert _plain(logfmt.colorize(line)) == line


def test_colour_is_actually_added():
    assert "\033[" in logfmt.colorize("22:41:07.412 [wake] model=alexa")


def test_non_event_lines_pass_through_untouched():
    # systemd's own lines and stray tracebacks share this stream.
    for line in ["Started FRIDAY voice assistant.",
                 "Traceback (most recent call last):",
                 "  File \"x.py\", line 1"]:
        assert logfmt.colorize(line) == line


def test_colour_can_be_disabled():
    line = "22:41:07.412 [wake] model=alexa"
    assert logfmt.colorize(line, color=False) == line


def test_subsystem_colour_is_stable():
    a = logfmt.colorize("22:41:07.412 [gateway] up")
    b = logfmt.colorize("22:41:09.999 [gateway] down")
    assert a.split("[gateway]")[0].split("\033[0m")[-1] == \
           b.split("[gateway]")[0].split("\033[0m")[-1]


def test_failures_are_red():
    assert "\033[31m" in logfmt.colorize("22:41:07.412 [turn] failed error=Boom")


def test_preemption_is_yellow_not_red():
    # A preemption is FRIDAY behaving correctly -- another app took the mic.
    # Colouring it as an error trains the user to ignore real errors.
    out = logfmt.colorize("22:41:07.412 [mic] paused who=zoom")
    assert "\033[33m" in out


def test_no_color_env_disables(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    assert logfmt.enabled(io.StringIO()) is False


def test_non_tty_is_not_coloured(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("FRIDAY_COLOR", raising=False)
    assert logfmt.enabled(io.StringIO()) is False


def test_stream_lines_writes_every_line():
    src = ["22:41:07.412 [wake] model=alexa\n", "plain line\n",
           "22:41:08.000 [stt] \"hello\"\n"]
    out = io.StringIO()
    logfmt.stream_lines(iter(src), out, color=False)
    assert out.getvalue().splitlines() == [s.rstrip("\n") for s in src]


def test_a_line_without_milliseconds_is_left_alone():
    # Guards the regex against half-matching and emitting a mangled line.
    assert logfmt.colorize("22:41:09 [route] tier=x") == "22:41:09 [route] tier=x"
