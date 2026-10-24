# IQS-1 certification corpus

The text the Infernix Quality Standard measures every execution path on
([model quality §6](../../../docs/maintainer/model-quality.md#6-numerical-acceptance-policy-and-fast-paths)).
Built deterministically by `tools/quality/iqs_corpus.py` from the repository's existing corpora and the
public datasets listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md); `provenance.json` records each
document's source, slice and licence. `manifest.json` is what `infernix-certify` reads.

| Set | Documents | Window | Decode segments per window |
|---|---|---|---|
| `main` | 64: English reference, English long form, Chinese, German, code, mathematics, chat, tool calls (8 each) | 4,096 tokens, every position scored | 64 tokens at 256, 1,024, 2,048 and 3,840 |
| `long` | 4: a book, encyclopaedic English, Chinese, a code repository | 32,768 tokens | 64 tokens at 8,192, 16,384, 24,576 and 32,640 |

Text documents are tokenized raw by each artifact. Chat documents (`kind: chat`, JSON) are rendered with each
model's own chat template, their final assistant message continued: the oasst2 conversations, the reasoning
chats (each turn's worked solution as reasoning, kept in every turn) and the tool-call sessions (tools as
the template declares them, each call a structured tool call, each result a tool message).

Rebuild (the downloaded sources under `--sources`, one folder per dataset):

```bash
python -I tools/quality/iqs_corpus.py --sources .local/iqs-sources --existing eval/corpora --out eval/corpora/iqs-1
```

The main set grows in layers of 32 windows, four documents per domain, appended in window order: windows 0-31
are the original corpus, 32-63 the second layer (`--per-domain 8`, the default). A layer never changes the
documents before it, and `infernix-certify` matches each stored window to its tokens, so references, noise
twins and reproducible arms stored for earlier windows stay valid and a larger corpus scores only its new
windows. The sources of layers 2 and 3 (`--per-domain 16`) are listed in the builder; they need eight more
German books and more competition mathematics than MATH-500 holds.

A change to an existing document changes that window's tokens: stored windows no longer match it and are
regenerated.
