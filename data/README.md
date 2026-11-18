# Data

No dataset files are included in this repository. BioASQ data is distributed by the BioASQ challenge organisers under their own terms; obtain it yourself and place it here.

## 1. Get the data

1. Register at <http://bioasq.org> and download the factoid question-answering data.
2. Convert it to **SQuAD v1.1 format** (`{"data": [{"paragraphs": [{"context": ..., "qas": [{"id", "question", "answers": [{"text", "answer_start"}]}]}]}]}`).
3. Save it as `data/raw/bioasq.json`.

The experiments used a file with 2,511 unique contexts and 4,772 valid factoid QA pairs.

## 2. Prepare the artifacts

Run from the repository root:

```bash
# Frozen 120-question evaluation subset (stratified, seed 42).
# Re-running this on the same input reproduces the original subset byte-for-byte.
rag-thesis make-subset --in data/raw/bioasq.json --out data/raw/bioasq_subset.json \
    --n_qas 120 --mode stratified --seed 42

# Documents, chunks (180 MiniLM wordpiece tokens, 40 overlap), questions, gold evidence
rag-thesis build-corpus --in data/raw/bioasq_subset.json

# Retrieval indexes
rag-thesis build-index --bm25
rag-thesis build-index --dense

# Schema validation of every artifact
rag-thesis validate
```

## 3. Optional: the full corpus

The default corpus is closed: it holds only the contexts of the 120 evaluation questions. The full
corpus holds every context of `bioasq.json`, so the same 120 questions are searched among all the
other passages too. It lives in its own folders, next to the default corpus:

```bash
# Documents and chunks of every context (+ questions and gold evidence for all QA pairs)
rag-thesis build-corpus --in data/raw/bioasq.json \
    --processed-dir data/full/processed --eval-dir data/full/eval --report-dir outputs/reports/full_corpus

# Indexes of the full corpus (one at a time; the dense build uses the GPU if there is one)
rag-thesis build-index --bm25 --chunks-path data/full/processed/chunks.jsonl --index-dir indexes/full/bm25
rag-thesis build-index --dense --chunks-path data/full/processed/chunks.jsonl --index-dir indexes/full/chroma

# Retrieval of the 120 evaluation questions against the full corpus. Documents and chunks have
# content-hash ids, so the questions and gold evidence of the default build (data/eval) are valid
# for the full corpus too.
rag-thesis eval-retrieval --bm25-dir indexes/full/bm25 --vector-dir indexes/full/chroma \
    --output-dir outputs/reports/full_corpus/none
```

Add `--queries-from-log <run log>` to `eval-retrieval` to retrieve each question with the
(rewritten, guard-checked) queries a run recorded instead of the question itself; no LLM call is made.

## 4. Optional: a dev split for tuning

Tuning (e.g. the fusion weights of rewritten queries) must never see the 120 evaluation questions.
This draws 120 other BioASQ questions (one QA pair each; every snippet of an evaluation question is
excluded) from the full-corpus build, so it needs section 3 first:

```bash
python tools/make_dev_split.py --out-dir data/dev
```

## Layout

```
data/
  raw/         bioasq.json, bioasq_subset.json        (you provide / generate)
  processed/   documents.jsonl, chunks.jsonl          (generated)
  eval/        questions.jsonl, gold_evidence.jsonl   (generated)
  full/        processed/ and eval/ of the full corpus (optional, generated)
  dev/         questions.jsonl, gold_evidence.jsonl of the dev split (optional, generated)
```

Everything under `data/` except this README is ignored by git.
