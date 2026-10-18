# The PLE n-gram volume is in another repository

This model needs a Qwen3.8-Flash-Next PLE n-gram volume, which this repository does not contain.
Three volumes of the table exist, one per row format, and any of them runs this model. The
recommended one is the FP8 volume: the abliteration left the n-gram table untouched as far as FP8
can tell (quantized to FP8, it equals NVIDIA's table byte for byte; all 320,001,536 rows checked),
so it is the volume this model was converted and measured with.

**Download one of:**

- **Recommended:** `Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix.ngram` (FP8, 52,429,058,048 bytes) from
  https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix/blob/main/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix.ngram
  or the identical copy in https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix
- `Qwen3.8-Flash-Next-BF16.ngram` (BF16, Qwen's unquantized table, 109,227,196,416 bytes) from
  https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-FP8-Infernix/blob/main/Qwen3.8-Flash-Next-BF16.ngram
- `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix.ngram` (IQ4_NL, 4.5 bits per value, 29,127,258,112
  bytes) from
  https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix/blob/main/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix.ngram

```text
hf download Wallawalla47/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix.ngram --local-dir <nvme dir>
hf download Wallawalla47/Qwen3.8-Flash-Next-FP8-Infernix Qwen3.8-Flash-Next-BF16.ngram --local-dir <nvme dir>
hf download Wallawalla47/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix.ngram --local-dir <nvme dir>
```

Older Infernix builds run this model with the FP8 and BF16 volumes only.

Put the volume on an NVMe drive (it is read in random 4 KiB blocks) and pass it to Infernix:

```text
infernix-serve Qwen3.8-Flash-Next-Uncensored-NVFP4-Infernix-00001-of-00003.infernix \
  --ngram-volume <nvme dir>/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix.ngram ...
```

If you already have a volume for another Flash-Next conversion, use it: one copy serves all of
them.
