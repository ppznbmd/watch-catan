"""The awareness study is a document built from data. If a verdict file, an
answer file or the analysis changes, the numbers it reports change, and a reader
following it would find figures that no longer come out."""

import pytest

from scripts import awareness_study as study


@pytest.fixture(scope="module")
def built():
    return study.build()


def test_the_committed_report_is_what_the_data_produces_today(built):
    """A report edited by hand, or left behind after a verdict was corrected,
    would state numbers nobody can reproduce."""
    assert study.REPORT.read_text() == built[3], "rerun scripts/awareness_study.py"


def test_the_headline_counts_are_the_ones_the_journal_reports(built):
    """These are quoted in CLAUDE.md and the write-ups. If one moves,
    those documents are now wrong and have to be corrected with it."""
    t, m, f, _ = built
    assert (t["test"], t["n"]) == (26, 273)
    assert (m["test"], m["n"]) == (8, 362)
    assert (m["long_match"], m["long_trap"]) == ([8, 81], [20, 137])
    table = f["table"]
    assert (table["none"]["reveals"], table["none"]["offers"]) == (0, 61)
    assert (table["honesty"]["reveals"], table["honesty"]["offers"]) == (39, 54)
    assert (table["none"]["lie"], table["honesty"]["lie"]) == (10, 2)
    assert f["within"][:2] == (17, 25)


def test_the_circular_split_is_shown_beside_the_one_that_replaced_it(built):
    """The study keeps its own mistake. If the two splits ever agreed, the
    paragraph explaining why one is circular would be describing nothing."""
    f = built[2]
    circular_long = f["circular"][2]
    independent = [k / n for k, n in f["independent"]]
    assert circular_long == (7, 23)
    assert max(independent) - min(independent) < 0.1
