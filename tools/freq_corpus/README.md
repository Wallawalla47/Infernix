# Token-frequency rankings

A draft head that holds only a model's most frequent tokens needs a ranking of its vocabulary. This folder measures one
through the public Engine, by the method of the Qwen ranking in `fixtures/ranking` (from NInfer): the model's own most
probable token at every assistant position of a chat corpus, teacher-forced on the corpus's text, counted per stratum
(`en_chat`, `zh_chat`, `other_chat`, `code`, `math`). The Qwen corpus was not published; a ranking belongs to one
tokenizer.

## 1. Prepare a corpus

```powershell
py -3.11 tools/freq_corpus/prepare_corpus.py --raw RAW_DIR --tokenizer tokenizer.json --out CORPUS_DIR --train-tokens 12000000
```

`RAW_DIR` holds the parquet shards in one folder per dataset (`allenai__WildChat-1M`, `CohereLabs__aya_dataset`,
`OpenAssistant__oasst2`, `nvidia__OpenCodeInstruct`, `nvidia__OpenMathInstruct-2`, `openai__gsm8k`) and a `MANIFEST.json`
of their revisions and licences. The script writes `train.jsonl`, `heldout.jsonl` (one conversation in ten, by the hash of
its id), `groundtruth.counts.i64` (each token's count in all usable assistant text) and `corpus.manifest.json`. Budgets
keep NInfer's stratum shares, and a run is deterministic.

## 2. Measure

Build the `infernix_token_ranking` target, then for each split:

```powershell
infernix_token_ranking MODEL.infernix --corpus CORPUS_DIR/train.jsonl --output OUT/ranking.train --vocab V --context 4096
```

`V` is the model's vocabulary size. The tool scores with `ScoreOptions::argmax_only`, which finds each position's top
token on the GPU, and writes `PREFIX.counts.i64` (int64 LE; row 0 the totals, rows 1 to 5 the strata) and
`PREFIX.manifest.json`. It saves every five minutes and resumes from `PREFIX.state.json` when run again.

Where a chat template writes the reasoning block only on the last assistant turn (Qwen3.6's), the tool counts only the
turns that open with it. The earlier turns are history in a form the model never writes, and its top token there is
mostly the end of the turn. On six test conversations this took the Qwen3.6 35B's agreement with the text from 30 % to
77 %, and left the counts of the Qwen3.8 27B, Flash-Next and Inkling 2 unchanged.

## 3. Build the rank order

```powershell
py -3.11 tools/freq_corpus/build_rank.py --train OUT/ranking.train.counts.i64 --heldout OUT/ranking.heldout.counts.i64 --groundtruth CORPUS_DIR/groundtruth.counts.i64 --tokenizer tokenizer.json --out OUT
```

Each stratum's counts become frequencies, mixed with more weight on English and code (English 50 %, Chinese 20 %, code
15 %, other languages 12 %, maths 3 %). Tokens are ordered by that mix, then by their corpus-text counts, then by id, with
the tokenizer's special tokens first. `--train` takes several counts files, so `extend_corpus.py` can add a stratum to a
finished measurement. The script writes `rank.order.i32` and `coverage.report.json`, the share of the held-out split's
positions each shortlist length holds. On Inkling 2, `python -m tools.convert --token-rank rank.order.i32` stores the order for
`--lm-head-draft`.
