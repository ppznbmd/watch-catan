from arena.env import load_env, parse


def test_parses_the_shapes_people_actually_write():
    values, complaints = parse(
        """
        # a comment
        DEEPSEEK_API_KEY=sk-plain
        export ANTHROPIC_API_KEY="sk-quoted"
        SINGLE='sk-single'
        WITH_EQUALS=a=b=c
        EMPTY=
        """
    )
    assert values == {
        "DEEPSEEK_API_KEY": "sk-plain",
        "ANTHROPIC_API_KEY": "sk-quoted",
        "SINGLE": "sk-single",
        "WITH_EQUALS": "a=b=c",
        "EMPTY": "",
    }
    assert complaints == []


def test_a_malformed_line_is_reported_not_swallowed():
    """A key that silently vanishes looks exactly like a key you never set."""
    values, complaints = parse("GOOD=1\nthis is not a setting\n")
    assert values == {"GOOD": "1"}
    assert len(complaints) == 1 and "line 2" in complaints[0]


def test_the_real_environment_wins(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("DEEPSEEK_API_KEY=from-file\nOTHER=from-file\n")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "from-export")
    load_env(env)
    import os
    assert os.environ["DEEPSEEK_API_KEY"] == "from-export"
    assert os.environ["OTHER"] == "from-file"


def test_a_missing_file_is_not_an_error(tmp_path):
    assert load_env(tmp_path / "nope.env") == []
