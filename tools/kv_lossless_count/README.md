# KV lossless block count (PB-01, step 1)

TEMPORARY, with the `INFERNIX_TMP_KV_DUMP` switch: measures whether a 12-bit lossless form of
BF16 KV pages (ZipFusion, arXiv 2610.33719; candidates §23.2 PB-01) would apply to real Qwen3.5-family
KV before anything is built. PB-01 is built only if at least 80 % of the K bytes and of the V bytes
qualify. Remove this directory with the switch once that is decided.

## The rule

A block of 512 BF16 values is stored in 12 bits a value (each value's low byte, holding its seven
fraction bits and lowest exponent bit, plus a 4-bit code: the sign and a 3-bit offset of the exponent
from the block base, the block's smallest exponent rounded down to even) when all its exponents lie
within 16 binades of that base; otherwise it stays raw. `count.py` counts the blocks that qualify in
the page's storage order (a block is two consecutive slots of one KV head at head dim 256), under the
paper's per-block base and under a per-channel base (one base per head-dim channel per page and KV
head), with the average bits per value of each, metadata included. Its docstring has the details.

K is stored as BF16 (after QK-norm and partial RoPE). V is stored as FP16 on the default route; the
form would store V's BF16 source and convert in registers, so V is counted on that source, recovered
from the FP16 values (exact for 2^-17 <= |v| < 2^16; the report gives the share outside). A dump taken
under `--lossless` stores V as BF16 directly.

## Run

1. Build long prompts (32K+ tokens): code, documentation and an agent trace. `make_messages.py` writes
   `--messages` files from repository text (about 160,000 characters, 35-40K tokens, by default):

   ```bash
   python -I tools/kv_lossless_count/make_messages.py text  kv-code.json  src/ops/softmax_attention src/ops/kv_cache
   python -I tools/kv_lossless_count/make_messages.py text  kv-docs.json  docs
   python -I tools/kv_lossless_count/make_messages.py agent kv-agent.json src/models/qwen3_5/program src/runtime/engine
   ```

2. Prefill each prompt with the dump switch on a BF16 KV cache (one generated token is enough; the
   pages are written after the prefill, outside its reported time):

   ```bash
   INFERNIX_TMP_KV_DUMP=kv-dump ./build/apps/infernix MODEL.ninfer --messages kv-code.json \
     --kv-dtype bf16 --max-context 65536 --max-new 1
   ```

   Each prompt writes `kv-<process>-<n>.json` and, per attention layer, `...-LNN-K.bin` and
   `...-LNN-V.bin` into the directory (the 27B: 64 KiB a token, 2 GiB at 32K; the 35B-A3B: 20 KiB a
   token). Only the main text KV is written, and only on a BF16 cache.

3. Count (needs numpy; one dump or a whole directory, every dump of a model also summed):

   ```bash
   python -I tools/kv_lossless_count/count.py kv-dump [--csv kv-count.csv]
   ```

   Per layer and plane: complete blocks, the share holding a zero or subnormal (`zero%`), percentiles
   of the block's exponent span, and the qualifying share and bits per value under each base; then the
   K, V and K+V totals and the 80 % verdict for each base.
