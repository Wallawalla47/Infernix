# IQS-1 certification corpus

The text the Infernix Quality Standard measures every execution path on
([model quality §6](../../../docs/maintainer/model-quality.md#6-numerical-acceptance-policy-and-fast-paths)).
Built deterministically by `tools/quality/iqs_corpus.py` from the repository's existing corpora and the
public datasets listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md); `provenance.json` records each
document's source, slice and licence. `manifest.json` is what `infernix-certify` reads.

| Set | Documents | Window | Decode segments per window |
|---|---|---|---|
| `main` | 32: English reference, English long form, Chinese, German, code, mathematics, chat, tool calls (4 each) | 4,096 tokens, every position scored | 64 tokens at 256, 1,024, 2,048 and 3,840 |
| `long` | 4: a book, encyclopaedic English, Chinese, a code repository | 32,768 tokens | 64 tokens at 8,192, 16,384, 24,576 and 32,640 |

Text documents are tokenized raw by each artifact. Chat documents (`kind: chat`, JSON) are rendered with each
model's own chat template, their final assistant message continued: the oasst2 conversations, the reasoning
chats (each turn's worked solution as reasoning, kept in every turn) and the tool-call sessions (tools as
the template declares them, each call a structured tool call, each result a tool message).

Rebuild (the downloaded sources under `--sources`, one folder per dataset):

```bash
python -I tools/quality/iqs_corpus.py --sources .local/iqs-sources --existing eval/corpora --out eval/corpora/iqs-1
```

A change to any document changes the corpus content hash every reference and certificate records, so stored
references stop matching and are regenerated.
