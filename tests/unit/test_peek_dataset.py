import json
from pathlib import Path

from tools.peek_dataset import app
from typer.testing import CliRunner

RUNNER = CliRunner()
ASSET_PATH = Path(__file__).resolve().parents[1] / "assets" / "sample_squad.json"


def test_peek_dataset_reports_expected_counts_for_valid_sample():
    result = RUNNER.invoke(app, ["--in", str(ASSET_PATH)])

    assert result.exit_code == 0
    assert "data_items: 1" in result.output
    assert "paragraphs: 1" in result.output
    assert "total_qas: 2" in result.output
    assert "- q1: What is ATP?" in result.output
    assert "- q2: What is BRCA1?" in result.output
    assert "empty_questions: 0" in result.output
    assert "empty_answers: 0" in result.output
    assert "invalid_answer_start: 0" in result.output
    assert "answer_text_mismatch: 0" in result.output


def test_peek_dataset_counts_malformed_records_without_crashing(tmp_path):
    malformed_payload = {
        "data": [
            {
                "title": "Malformed Doc",
                "paragraphs": [
                    {
                        "context": "Protein is useful.",
                        "qas": [
                            {
                                "id": "q-bad-1",
                                "question": "",
                                "answers": [],
                            },
                            {
                                "id": "q-bad-2",
                                "question": "What is Protein?",
                                "answers": [
                                    {
                                        "text": "Protein",
                                        "answer_start": -1,
                                    },
                                    {
                                        "text": "Mismatch",
                                        "answer_start": 0,
                                    },
                                    {
                                        "text": 123,
                                        "answer_start": 0,
                                    },
                                ],
                            },
                            "not-a-qa",
                        ],
                    },
                    {
                        "context": 123,
                        "qas": [],
                    },
                ],
            },
            "not-a-data-item",
        ]
    }
    dataset_path = tmp_path / "malformed.json"
    dataset_path.write_text(json.dumps(malformed_payload), encoding="utf-8")

    result = RUNNER.invoke(app, ["--in", str(dataset_path)])

    assert result.exit_code == 0
    assert "data_items: 2" in result.output
    assert "paragraphs: 2" in result.output
    assert "total_qas: 2" in result.output
    assert "empty_questions: 1" in result.output
    assert "empty_answers: 1" in result.output
    assert "invalid_answer_start: 1" in result.output
    assert "answer_text_mismatch: 1" in result.output
    assert "malformed_data_items: 1" in result.output
    assert "malformed_paragraphs: 1" in result.output
    assert "malformed_qas: 1" in result.output
    assert "malformed_answers: 1" in result.output


def test_peek_dataset_help_exposes_expected_cli_flag():
    result = RUNNER.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "--in" in result.output
