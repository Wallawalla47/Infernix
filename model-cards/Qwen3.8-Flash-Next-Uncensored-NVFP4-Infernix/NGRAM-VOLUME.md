# The PLE n-gram volume is in another repository

This model needs Qwen3.8-Flash-Next's 52 GB PLE n-gram volume, which this repository does not
contain. The abliteration left the n-gram table untouched: quantized to FP8, it equals NVIDIA's table
byte for byte (all 320,001,536 rows checked), so the file is the same one the NVIDIA conversions use.

**Download:** `Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix.ngram` (52,429,058,048 bytes) from

- https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix/blob/main/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix.ngram
- or the identical copy in https://huggingface.co/Wallawalla47/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix

```text
hf download Wallawalla47/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix.ngram --local-dir <nvme dir>
```

Put it on an NVMe drive (it is read in random 4 KiB blocks) and pass it to Infernix:

```text
infernix-serve Qwen3.8-Flash-Next-Uncensored-NVFP4-Infernix-00001-of-00003.infernix \
  --ngram-volume <nvme dir>/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix.ngram ...
```

If you already run either NVIDIA conversion, use the volume you have: one copy serves all three
Flash-Next models.
