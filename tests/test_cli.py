import pytest

from flashtrace.cli import main, parse_span


def test_parse_span():
    assert parse_span("3:8") == (3, 8)
    assert parse_span(None) is None


@pytest.mark.parametrize("value", ["3", "8:3", "a:b"])
def test_parse_span_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        parse_span(value)


def test_cli_help_exits_successfully(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])

    assert exc.value.code == 0
    assert "trace" in capsys.readouterr().out
