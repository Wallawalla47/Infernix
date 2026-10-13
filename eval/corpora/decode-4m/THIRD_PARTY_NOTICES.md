# Third-Party Notices

The files below are evaluation inputs. They retain the licenses and attribution requirements of
their sources and are not relicensed as Infernix source code.

## PG-19

`data/pg19/` contains books from DeepMind's PG-19 test and validation splits. PG-19 is distributed
under the Apache License 2.0 and was prepared from books in Project Gutenberg published before 1919.
The selected book files are recorded in `provenance/pg19.jsonl`. Project Gutenberg's trademark and
local-jurisdiction terms remain applicable independently of the source texts' public-domain status
in the United States.

- <https://huggingface.co/datasets/deepmind/pg19>
- <https://www.apache.org/licenses/LICENSE-2.0>
- <https://www.gutenberg.org/policy/terms_of_use.html>

PG-19 includes historical language and biases. Inclusion is solely for language-model evaluation
and does not imply endorsement.

## WikiText-103 raw validation

`data/wikitext/` contains the WikiText-103 validation split distributed by Salesforce Research at
the revision recorded in `provenance/wikitext.jsonl`. The dataset is derived from Wikipedia Good and
Featured articles and is distributed under the Creative Commons Attribution-ShareAlike 4.0
International license:

- <https://huggingface.co/datasets/Salesforce/wikitext>
- <https://creativecommons.org/licenses/by-sa/4.0/>

## Chinese Wikipedia

`data/zhwiki/` contains article text from the `20231101.zh` Wikimedia Wikipedia dataset snapshot
recorded in `provenance/zhwiki.jsonl`. Wikipedia text is available under the Creative Commons
Attribution-ShareAlike 3.0 Unported license and, where applicable, the GNU Free Documentation
License. Provenance segments identify each article as `article/<id>`; its stable source URL is
`https://zh.wikipedia.org/?curid=<id>`.

- <https://huggingface.co/datasets/wikimedia/wikipedia>
- <https://creativecommons.org/licenses/by-sa/3.0/>
- <https://www.gnu.org/licenses/fdl-1.3.html>
- <https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use>

## SGLang source

`data/sglang/` contains source files from the SGLang revision recorded in
`provenance/sglang.jsonl` (a fork of <https://github.com/sgl-project/sglang>), distributed under
the Apache License 2.0:

- <https://www.apache.org/licenses/LICENSE-2.0>
