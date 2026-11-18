"""guard/aliases.py: Schwartz & Hearst abbreviation definitions and the corpus dictionary."""

from __future__ import annotations

from rag_thesis.guard.aliases import AbbreviationDictionary, extract_abbreviations, find_expansion


def test_finds_classic_definitions():
    text = (
        "Lewy bodies contain alpha-synuclein. The heat shock protein (HSP) family and adenosine "
        "triphosphate (ATP) were measured, as was tumor necrosis factor-alpha (TNF-alpha)."
    )
    assert extract_abbreviations(text) == [
        ("HSP", "heat shock protein"),
        ("ATP", "adenosine triphosphate"),
        ("TNF-alpha", "tumor necrosis factor-alpha"),
    ]


def test_the_long_form_is_the_shortest_match_starting_a_word():
    # "the" must not be pulled in; the first letter of the short form starts a word.
    assert extract_abbreviations("We studied the methicillin-resistant Staphylococcus aureus (MRSA).") == [
        ("MRSA", "methicillin-resistant Staphylococcus aureus")
    ]


def test_ignores_brackets_that_are_not_definitions():
    text = "Mice (n = 12) were treated (2010) with drug (see below) and controls (p<0.05)."
    assert extract_abbreviations(text) == []
    assert extract_abbreviations("No match here (XYZ).") == []  # letters not found in order


def test_dictionary_counts_long_forms_per_short_form():
    texts = [
        "adenosine triphosphate (ATP) levels",
        "Adenosine  triphosphate (ATP) again",
        "amyloid transport protein (ATP) is rare",
        "heat shock protein (HSP)",
    ]
    dictionary = AbbreviationDictionary.from_texts(texts)
    assert dictionary.expansions("atp") == ["adenosine triphosphate", "amyloid transport protein"]
    assert dictionary.expansions("HSP") == ["heat shock protein"]
    assert dictionary.expansions("XYZ") == []
    assert len(dictionary) == 2


def test_find_expansion_spots_a_long_form_written_out_anywhere():
    assert find_expansion("DAT1", "Where is the human dopamine transporter 1 gene located?") == "dopamine transporter 1"
    assert find_expansion("ChIP-seq", "data from Chromatin Immunoprecipitation Sequencing runs") == (
        "Chromatin Immunoprecipitation Sequencing"
    )
    assert find_expansion("DAT1", "Where is the gene located?") is None
    assert find_expansion("[18F]altanserin-x", "anything") is None  # longer than a short form


def test_expand_adds_the_most_frequent_long_form_and_keeps_the_short_form():
    dictionary = AbbreviationDictionary.from_texts(
        ["dopamine transporter 1 (DAT1) gene", "dopamine transporter 1 (DAT1)", "the insertion sequence (IS)"]
    )
    assert dictionary.expand("Where is DAT1 located?") == "Where is DAT1 (dopamine transporter 1) located?"
    assert dictionary.expand("Where is the dopamine transporter 1 (DAT1)?") == (
        "Where is the dopamine transporter 1 (DAT1)?"  # already spelled out
    )
    assert dictionary.expand("Is it located here?") == "Is it located here?"  # "Is" isn't identifier-like
    assert dictionary.expand("Where is XYZ1?") == "Where is XYZ1?"  # unknown short form
