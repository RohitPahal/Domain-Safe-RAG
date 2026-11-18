from rag_thesis.ids import deterministic_hash


def test_same_input_produces_same_hash() -> None:
    value = "deterministic text"
    assert deterministic_hash(value) == deterministic_hash(value)


def test_different_input_produces_different_hash() -> None:
    assert deterministic_hash("first") != deterministic_hash("second")


def test_hash_matches_expected_sha256_digest() -> None:
    assert deterministic_hash("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
