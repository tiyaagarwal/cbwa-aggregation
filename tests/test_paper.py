"""Every number the paper derives from the CBWA rule is reproduced exactly (at its rounding)."""

import pytest

from cbwa.paper import checks

CHECKS = checks()


@pytest.mark.parametrize("check", CHECKS, ids=[c.where for c in CHECKS])
def test_value_matches_the_paper(check):
    assert check.ok, f"{check.where}: paper {check.paper}, code {check.computed}"


def test_number_of_checks():
    assert len(CHECKS) == 44
