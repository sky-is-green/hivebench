"""Unit tests for sieve.vocabulary."""

import pytest

from sieve.vocabulary import Vocabulary


def test_load_code_and_general_merge():
    v = Vocabulary.load("code", "general")
    assert v.size > 0
    assert v.matches("the class uses a foreign key")  # code structural
    assert v.matches("we agreed on a decision")        # general


def test_matches_case_insensitive():
    v = Vocabulary.load("code")
    assert v.matches("import torch") is True
    assert v.matches("IMPORT torch") is True


def test_matched_terms_returns_found_terms():
    v = Vocabulary(["JWT", "schema"])
    terms = v.matched_terms("The JWT schema is defined")
    assert set(t.lower() for t in terms) == {"jwt", "schema"}


def test_no_match_returns_empty():
    v = Vocabulary(["tensorflow"])
    assert v.matched_terms("nothing relevant here") == []
    assert v.matches("nothing relevant here") is False


def test_missing_domain_raises():
    with pytest.raises(FileNotFoundError):
        Vocabulary.load("does_not_exist")


def test_empty_vocabulary():
    v = Vocabulary([])
    assert v.size == 0
    assert v.matches("anything") is False
    assert v.matched_terms("anything") == []