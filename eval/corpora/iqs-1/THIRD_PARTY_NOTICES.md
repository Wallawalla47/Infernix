# Third-Party Notices

The files under `data/` are evaluation inputs. They keep the licences and attribution requirements of their
sources and are not relicensed as Infernix source code. `provenance.json` maps every document to its source,
slice or item range.

## WikiText (English reference, long set)

Excerpts of the WikiText dataset (Salesforce Research), derived from Wikipedia Good and Featured articles,
Creative Commons Attribution-ShareAlike 4.0 International, as carried by `eval/corpora/perplexity-1m` and
`eval/corpora/decode-4m` (see their notices for the pinned revision).

- <https://huggingface.co/datasets/Salesforce/wikitext>
- <https://creativecommons.org/licenses/by-sa/4.0/>

## PG-19 (English long form, long set)

Excerpts of books in DeepMind's PG-19 test and validation splits (Apache License 2.0; Project Gutenberg books
published before 1919), as carried by the existing corpora. Project Gutenberg's trademark terms remain applicable.

- <https://huggingface.co/datasets/deepmind/pg19>
- <https://www.gutenberg.org/policy/terms_of_use.html>

## Chinese Wikipedia (Chinese, long set)

Article text from the `20231101.zh` Wikimedia Wikipedia snapshot, as carried by the existing corpora:
Creative Commons Attribution-ShareAlike 3.0 Unported and, where applicable, the GNU Free Documentation License.

- <https://huggingface.co/datasets/wikimedia/wikipedia>
- <https://creativecommons.org/licenses/by-sa/3.0/>

## Project Gutenberg German books (German)

Franz Kafka, *Die Verwandlung* (ebook 22367); Thomas Mann, *Der Tod in Venedig* (12108); Hermann Hesse,
*Siddhartha* (2499); Theodor Fontane, *Effi Briest* (5323); Alexander von Humboldt, *Reise in die
Aequinoctial-Gegenden des neuen Continents*, Band 2 (24746); Joseph von Eichendorff, *Aus dem Leben eines
Taugenichts* (35312); Sigmund Freud, *Die Traumdeutung* (40739); Friedrich Nietzsche, *Also sprach
Zarathustra* (7205). Public domain in the United States; the Project Gutenberg licence header and footer are
removed and the trademark terms remain applicable.

- <https://www.gutenberg.org/policy/terms_of_use.html>

## Source code (code, long set)

NInfer source excerpts (Apache License 2.0, as carried by `eval/corpora/perplexity-1m`) and SGLang source
excerpts (Apache License 2.0, as carried by `eval/corpora/decode-4m`).

## GSM8K (mathematics, reasoning chats)

The `main` test split of `openai/gsm8k`, MIT License; calculator annotations (`<<...>>`) removed.

- <https://huggingface.co/datasets/openai/gsm8k>

## MATH-500 (mathematics, reasoning chats)

`HuggingFaceH4/MATH-500`, the 500 MATH problems of OpenAI's *Let's Verify Step by Step* (prm800k, MIT
License).

- <https://huggingface.co/datasets/HuggingFaceH4/MATH-500>
- <https://github.com/openai/prm800k>

## OpenAssistant Conversations 2 (chat)

English threads of the `OpenAssistant/oasst2` validation split, Apache License 2.0.

- <https://huggingface.co/datasets/OpenAssistant/oasst2>

## Hermes Function-Calling V1 (tool calls)

`func-calling.json` and `json-mode-agentic.json` of `NousResearch/hermes-function-calling-v1`, Apache License
2.0, restructured as chat sessions (tool calls and responses as structured messages).

- <https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1>

These datasets include historical language and synthetic content. Inclusion is solely for measuring numerical
fidelity and implies no endorsement.
