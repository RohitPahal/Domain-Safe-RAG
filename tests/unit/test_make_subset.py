import json
from pathlib import Path

from typer.testing import CliRunner

from rag_thesis.cli.make_subset import app

RUNNER = CliRunner()


def _build_subset_fixture() -> dict:
    return {
        "version": "1.0",
        "data": [
            {
                "title": "Doc A",
                "paragraphs": [
                    {
                        "context": "Acetylsalicylic acid is aspirin. 5% glucose is used in labs.",
                        "qas": [
                            {
                                "id": "qa-synonym",
                                "question": "What is a synonym for aspirin?",
                                "answers": [{"text": "Acetylsalicylic acid", "answer_start": 0}],
                            },
                            {
                                "id": "qa-numeric",
                                "question": "What does 5% glucose indicate?",
                                "answers": [{"text": "5%", "answer_start": 35}],
                            },
                            {
                                "id": "qa-empty-answers",
                                "question": "Which answer is missing?",
                                "answers": [],
                            },
                        ],
                    }
                ],
            },
            {
                "title": "Doc B",
                "paragraphs": [
                    {
                        "context": "Adenosine triphosphate is ATP. The p53 pathway regulates apoptosis.",
                        "qas": [
                            {
                                "id": "qa-abbrev",
                                "question": "What is ATP?",
                                "answers": [{"text": "Adenosine triphosphate", "answer_start": 0}],
                            },
                            {
                                "id": "qa-other",
                                "question": "Which signaling pathway regulates apoptosis in human cells today?",
                                "answers": [{"text": "p53 pathway", "answer_start": 39}],
                            },
                            {
                                "id": "qa-synonym",
                                "question": "Duplicate should be skipped",
                                "answers": [{"text": "duplicate", "answer_start": 0}],
                            },
                        ],
                    }
                ],
            },
        ],
    }


def _write_dataset(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _load_dataset(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _collect_qas(payload: dict) -> list[dict]:
    collected = []
    for item in payload["data"]:
        for paragraph in item["paragraphs"]:
            collected.extend(paragraph["qas"])
    return collected


def test_make_subset_preserves_structure_and_limits_to_valid_unique_qas(tmp_path):
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "subset.json"
    _write_dataset(input_path, _build_subset_fixture())

    result = RUNNER.invoke(
        app,
        [
            "--in",
            str(input_path),
            "--out",
            str(output_path),
            "--n_qas",
            "10",
            "--mode",
            "first",
            "--seed",
            "42",
        ],
    )

    assert result.exit_code == 0
    subset_payload = _load_dataset(output_path)
    subset_qas = _collect_qas(subset_payload)

    assert subset_payload["version"] == "1.0"
    assert list(subset_payload.keys()) == ["version", "data"]
    assert [item["title"] for item in subset_payload["data"]] == ["Doc A", "Doc B"]
    assert list(subset_payload["data"][0].keys()) == ["title", "paragraphs"]
    assert list(subset_payload["data"][0]["paragraphs"][0].keys()) == ["context", "qas"]
    assert len(subset_qas) == 4
    assert [qa["id"] for qa in subset_qas] == [
        "qa-synonym",
        "qa-numeric",
        "qa-abbrev",
        "qa-other",
    ]
    assert all(qa["answers"] for qa in subset_qas)
    assert len({qa["id"] for qa in subset_qas}) == len(subset_qas)
    assert "Before selection:" in result.output
    assert "After selection:" in result.output


def test_make_subset_random_mode_is_deterministic_for_same_seed(tmp_path):
    input_path = tmp_path / "input.json"
    output_path_a = tmp_path / "subset_a.json"
    output_path_b = tmp_path / "subset_b.json"
    _write_dataset(input_path, _build_subset_fixture())

    args = [
        "--in",
        str(input_path),
        "--out",
        str(output_path_a),
        "--n_qas",
        "3",
        "--mode",
        "random",
        "--seed",
        "7",
    ]
    result_a = RUNNER.invoke(app, args)
    result_b = RUNNER.invoke(
        app,
        [
            "--in",
            str(input_path),
            "--out",
            str(output_path_b),
            "--n_qas",
            "3",
            "--mode",
            "random",
            "--seed",
            "7",
        ],
    )

    assert result_a.exit_code == 0
    assert result_b.exit_code == 0
    assert _load_dataset(output_path_a) == _load_dataset(output_path_b)


def test_make_subset_stratified_mode_selects_balanced_buckets(tmp_path):
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "subset.json"
    _write_dataset(input_path, _build_subset_fixture())

    result = RUNNER.invoke(
        app,
        [
            "--in",
            str(input_path),
            "--out",
            str(output_path),
            "--n_qas",
            "4",
            "--mode",
            "stratified",
            "--seed",
            "42",
        ],
    )

    assert result.exit_code == 0
    subset_payload = _load_dataset(output_path)
    subset_qids = [qa["id"] for qa in _collect_qas(subset_payload)]

    assert len(subset_qids) == 4
    assert set(subset_qids) == {"qa-synonym", "qa-numeric", "qa-abbrev", "qa-other"}
    assert "synonym_entity: 1" in result.output
    assert "numeric: 1" in result.output
    assert "abbrev_short: 1" in result.output
    assert "other: 1" in result.output


def test_make_subset_stratified_mode_is_deterministic_for_same_seed(tmp_path):
    input_path = tmp_path / "input.json"
    output_path_a = tmp_path / "subset_a.json"
    output_path_b = tmp_path / "subset_b.json"
    _write_dataset(input_path, _build_subset_fixture())

    args = [
        "--in",
        str(input_path),
        "--out",
        str(output_path_a),
        "--n_qas",
        "4",
        "--mode",
        "stratified",
        "--seed",
        "42",
    ]
    result_a = RUNNER.invoke(app, args)
    result_b = RUNNER.invoke(
        app,
        [
            "--in",
            str(input_path),
            "--out",
            str(output_path_b),
            "--n_qas",
            "4",
            "--mode",
            "stratified",
            "--seed",
            "42",
        ],
    )

    assert result_a.exit_code == 0
    assert result_b.exit_code == 0
    assert _load_dataset(output_path_a) == _load_dataset(output_path_b)


def test_make_subset_help_exposes_expected_cli_flags():
    result = RUNNER.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "--in" in result.output
    assert "--out" in result.output
    assert "--n_qas" in result.output
    assert "--mode" in result.output
    assert "--seed" in result.output
