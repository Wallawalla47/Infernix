# The PLE n-gram volume is in another repository

This model needs a Qwen3.8-Flash-Next PLE n-gram volume, which this repository does not contain.
Four volumes of the table exist, one per row format, and any of them runs this model. The
recommended one is the FP8 volume of NVIDIA's NVFP4 release. NVIDIA kept that table frozen through
QAD, so the volume holds exactly the table this checkpoint stores. This artifact was converted,
verified and measured with that volume.

**Download one of:**

- **Recommended:** `Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix.ngram` (FP8, NVIDIA's table,
  52,429,058,048 bytes) from
  https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix/blob/main/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix.ngram
  or the identical copy in https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix
- `Qwen3.8-Flash-Next-BF16.ngram` (BF16, Qwen's unquantized table, 109,227,196,416 bytes) from
  https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-FP8-Infernix/blob/main/Qwen3.8-Flash-Next-BF16.ngram
- `Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix.ngram` (NVFP4, NVIDIA FP4-Local's rows,
  29,127,258,112 bytes) from
  https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix/blob/main/Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix.ngram
- `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix.ngram` (IQ4_NL, 4.5 bits per value, 29,127,258,112
  bytes) from
  https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix/blob/main/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix.ngram

```text
hf download Wallawalla47/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix.ngram --local-dir <nvme dir>
hf download Wallawalla47/Qwen3.8-Flash-Next-FP8-Infernix Qwen3.8-Flash-Next-BF16.ngram --local-dir <nvme dir>
hf download Wallawalla47/Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix.ngram --local-dir <nvme dir>
hf download Wallawalla47/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Infernix.ngram --local-dir <nvme dir>
```

Put the volume on an NVMe drive, where it is read in random 4 KiB blocks, and pass it to Infernix:

```text
infernix-serve Qwen3.8-Flash-Next-NVIDIA-NVFP4-QAD-Dense8-Infernix-00001-of-00003.infernix \
  --ngram-volume <nvme dir>/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix.ngram ...
```

If you already have a volume for another Flash-Next conversion, use it: one copy serves all of them.
If you already run NVIDIA's NVFP4 or NVFP4 Dense8 conversion with Infernix, its volume is the
recommended one.
