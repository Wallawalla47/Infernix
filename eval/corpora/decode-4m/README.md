# Infernix Decode Corpus

`infernix-decode-v1` is the text of the teacher-forced decode gate
([model quality §6](../../../docs/maintainer/model-quality.md#6-numerical-acceptance-policy-and-fast-paths)).
It shares no text with the prefill corpus `ninfer-ppl-1m-v1`, so the two gates measure independent
positions. It holds four UTF-8 streams, one per domain:

| Domain | Source | Size |
|---|---|---:|
| `pg19` | PG-19 test and validation books the prefill corpus does not use | 5.7 MB |
| `wikitext` | WikiText-103 raw validation (the prefill corpus uses the test split) | 1.1 MB |
| `zhwiki` | Chinese Wikipedia articles of at least 2,000 characters the prefill corpus does not use | 4.3 MB |
| `sglang` | SGLang runtime sources (`python/sglang/srt`) | 4.6 MB |

Each stream is sized to about 1.3 million Qwen3.8 tokens. It was the corpus of the decode gate
replaced on 2026-10-22 (1,024 segments of 512-, 2,048- and 8,192-token prompts with 64 forced
tokens, about 3.7 million tokens); its SGLang stream is one of the IQS-1 corpus's code sources.

`manifest.json` records the streams. `provenance/*.jsonl` records each stream's source and pinned
revision, the selection rule, and every part in order with its byte length. See
`THIRD_PARTY_NOTICES.md` before redistributing the corpus. Tokenizer identity and exact token
counts are not part of the corpus contract; a changed corpus uses a new corpus identity.
