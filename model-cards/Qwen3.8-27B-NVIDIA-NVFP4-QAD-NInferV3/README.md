---
library_name: ninfer
pipeline_tag: image-text-to-text
license: other
license_name: nvidia-open-model-license
license_link: https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/
url: https://github.com/Wallawalla47/ninfer-custom
base_model: nvidia/Qwen3.8-27B-NVFP4-QAD
base_model_relation: quantized
tags:
  - Qwen3.8
  - nvidia
  - ModelOpt
  - nvfp4
  - fp4
  - w4a4
  - 4bit
  - fp8
  - QAD
  - quantization
  - quantized
  - blackwell
  - ninfer
  - qwen3.8-27b
  - 4-bit precision
  - multimodal
  - vision
  - reasoning
  - thinking
  - long-context
  - agentic
  - mtp
  - speculative-decoding
---

# Qwen3.8-27B-NVIDIA-NVFP4-QAD-NInferV3

**A single-file [NInfer V3](https://github.com/Wallawalla47/ninfer-custom) engine artifact of
[nvidia/Qwen3.8-27B-NVFP4-QAD](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4-QAD), NVIDIA's
quantization-aware-distilled NVFP4 release of Qwen3.8-27B, built for an NVIDIA RTX 5090 (sm_120a). No conversion
step at load time: point `ninfer-serve` at the `.ninfer` file and run.**

This is the NInfer form of the checkpoint above, made with the same recipe (with a Q8 head, below) as my
[Qwen3.8-27B-NVIDIA-NVFP4-NInferV3](https://huggingface.co/wallawalla47/Qwen3.8-27B-NVIDIA-NVFP4-NInferV3): the
Model Optimizer NVFP4 and FP8 codes and block scales of every transformer linear except the language-model head
are carried over **bit-exactly** (the per-tensor scales in NInfer's stored forms, below), and the MTP companion,
the DFlash2 draft model and the indexed proposal head are included so the artifact runs with NInfer's fastest
decode paths.

**How it differs from that artifact:** NVIDIA's QAD checkpoint stores the linear-attention (Gated Delta-Net)
projections of 38 of its 48 GDN layers in NVFP4, where the artifact above (made from an earlier revision of
nvidia/Qwen3.8-27B-NVFP4) stores every GDN projection in FP8. Those layers run on NInfer's NVFP4 kernels.

**Recommended engine:** I recommend running this artifact with [my fork of NInfer](https://github.com/Wallawalla47/ninfer-custom),
which adds much better prefix caching, tool-calling fixes, speed improvements and various other enhancements over
upstream. For the bit-exact Infernix form of the same checkpoint, see
[Qwen3.8-27B-NVIDIA-NVFP4-QAD-Infernix-Exact](https://huggingface.co/wallawalla47/Qwen3.8-27B-NVIDIA-NVFP4-QAD-Infernix-Exact).

## What is in the file

| Component | Contents |
|---|---|
| `text` | Qwen3.5-family hybrid backbone: 64 layers (Gated Delta-Net linear attention + full attention every 4th layer), hidden 5120, vocab 248,320, native context 262,144 |
| `vision` | qwen3_5_vision tower (depth 27) for image/video input |
| `mtp` | Qwen3.5 MTP companion for MTP speculative decoding |
| `dflash2` | 5-layer sliding-attention draft model (window 2048, selector rank 256 / top-16): the `--spec dflash2` path |
| `proposal` | **Indexed 131,072-row proposal head** gathered from the QAD output head: cheap copy/proposal drafting via `--lm-head-draft` |

Format census of the 1,454 tensor objects: **204 NVFP4** (all 128 MLP projections and the 76 NVFP4 GDN
projections, imported bit-exact), **53 FP8** (e4m3fn, row scale: the 32 full-attention and 20 FP8 GDN projections
imported bit-exact, plus the re-quantised embedding), 579 BF16, 478 FP32 (norms, convolutions and the 382 stored
NVFP4 activation divisors), 55 Q4, 54 Q5, 29 Q8 (the output head among them) and 1 Q6 (grouped, FP16 scales),
and 1 INT32 proposal index table. The file is 20,943,081,216 bytes.

## How it was created

The artifact was converted on CPU with the v3 converter's `qwen3_8_27b_nvfp4_nvidia` recipe, which picks each
matrix's form from its stored encoding. `qwen3_8_27b_nvfp4-nvidia-qad.ninfer.conversion.json`, included here, is
the full per-tensor conversion report. Sources and treatment:

1. **nvidia/Qwen3.8-27B-NVFP4-QAD** (revision `8158ac56`, Model Optimizer 0.46 mixed NVFP4/FP8): the dominant
   source.
   - **NVFP4 projections** (MLP gate/up/down in every layer; GDN query/key/value/z and output in 38 layers) are
     **imported in their encoded form, bit-exact**: the packed FP4 weights and E4M3 block scales are carried over
     unchanged. The FP32 `weight_scale_2` and `input_scale` are stored as NInfer's FP32 divisors (their
     reciprocals, rounded once to FP32); dividing by the stored divisor reproduces NVIDIA's weight scale exactly
     in 271 of 307 cases and within one unit in the last place in the rest. At runtime these drive NInfer's NVFP4
     kernels.
   - **FP8 projections** (full-attention q/k/v/o in all 16 attention layers; GDN projections in layers 0, 1, 2, 4,
     5, 57, 58, 60, 61 and 62) are **imported in their encoded per-row FP8 form**: the E4M3 weights are carried
     over unchanged, with the per-tensor FP32 scale stored as the format's BF16 row scale (exactly in 5 of 94
     cases; rounding changes the others by at most 0.33 %).
   - **Output head**: stored as NVFP4 in the source, but NInfer registers the vocabulary projection only for
     Q8/Q6/FP8, so it is dequantised and re-quantised to grouped Q8 (`grouped_absmax`, one FP16 scale per 32
     values, 16-bit activations). The artifact above used row-scale FP8 here; on this checkpoint Q8 lowers perplexity
     from 4.3537 to 4.3423 (-0.0026 nats per token, 95 % interval -0.0033 to -0.0019) for 0.5-0.7 % of NInfer's
     plain decode speed (DFlash2 rounds on the same text 0.3 % slower).
   - **Token embedding**: quantised from its source BF16 to row-scale FP8.
   - **GDN a/b projections, norms, convolutions, `dt_bias`/`a_log`** keep their source BF16/FP32 values.
   - **Vision tower**: carried over from the checkpoint in the same mixed BF16 + grouped Q4/Q5/Q6/Q8 storage as
     the artifact above.
2. **MTP companion**: the QAD checkpoint ships the MTP decoder layer but not its input projection (`mtp.fc`) and
   three norms. Its MTP decoder layer is byte-identical to nvidia/Qwen3.8-27B-NVFP4's, so those four tensors are
   taken from nvidia/Qwen3.8-27B-NVFP4 (revision `482ca0f3`).
3. **DFlash2 draft**: copied byte for byte from
   [Qwen3.8-27B-NVIDIA-NVFP4-NInferV3](https://huggingface.co/wallawalla47/Qwen3.8-27B-NVIDIA-NVFP4-NInferV3) (the
   converter checks that its bindings and configuration match this backbone).
4. **Indexed proposal head**: 131,072 rows gathered from the QAD output head, ranked by a frequency corpus, stored
   as grouped Q4 plus an INT32 token-id table; `--lm-head-draft` serves copy-heavy proposals from it.
5. **Resources**: the checkpoint's own chat template (as in the artifact above), tokenizer and generation config,
   with `tokenizer_config.json` normalised to what the C++ tokenizer requires.

No transformer projection is dequantised and re-quantised: the 204 NVFP4 and 52 FP8 projection objects are direct
imports of the checkpoint's stored codes and block scales, and only their per-tensor scales are rounded as above.
The only re-quantisations are the output head (NVFP4 to Q8, an engine requirement) and the token embedding (BF16
to FP8). Measured on my RTX 5090:

- **Perplexity** (Infernix's perplexity tool on this artifact, its quick corpus of 261,167 tokens, INT8 KV): 4.3423
  against 4.3472 for my earlier NVIDIA artifact (paired difference -0.0011 nats per token, 95 % interval -0.0035 to
  +0.0014: not significant); with the FP8 head the same conversion measures 4.3537, so the Q8 head is worth -0.0026
  nats per token. With 16-bit activations the QAD weights are the better ones (4.2983 against 4.3048, both with
  FP8 heads), but they lose more to NVIDIA's 4-bit activations (+0.0128 against +0.0098 nats per token).
- **Speed** (NInfer, 256 greedy tokens, INT8 KV): plain decode 84.7-85.1 tok/s (my earlier NVIDIA artifact
  79 tok/s: the 38 NVFP4 GDN layers read fewer bytes than FP8 ones), DFlash2 with 7 drafts and the proposal head
  230-398 tok/s over five prompts, MTP with 3 drafts 172 tok/s on prose; prefill 12.4-12.9k tok/s on prompts of
  about 2,000 tokens.

## How to run

Requires the NInfer V3 engine and an NVIDIA RTX 50-series GPU (built and tuned on an RTX 5090; the weights take
about 16.4 GiB of VRAM, 18.8 GiB with the DFlash2 drafter and the proposal head). I recommend [my fork](https://github.com/Wallawalla47/ninfer-custom) of NInfer, but it
is not required. The launch command below uses options from the fork; if you run upstream NInfer, drop any option
its `ninfer-serve --help` does not list. Prebuilt Windows executables of the fork are on its
[releases page](https://github.com/Wallawalla47/ninfer-custom/releases/latest) and need an NVIDIA driver of 580 or
later; on other systems, build from source ([quick start](https://github.com/Wallawalla47/ninfer-custom#quick-start-windows)).

```bat
ninfer-serve.exe qwen3_8_27b_nvfp4-nvidia-qad.ninfer --host 127.0.0.1 --port 8080 --max-context 240000 --max-concurrency 2 --spec dflash2 --draft-tokens 7 --lm-head-draft --ngram-draft-tokens 15 --ngram-min-match 12 --kv-dtype int8 --preserve-thinking --host-cache-mib 52000 --pending-timeout-ms 900000 --prefill-chunk 4096 --kv-capacity auto --vram-headroom-mib 0 --log-colours on --ngram-archive-mib 2048 --ngram-session-mib 256 --ngram-native-sessions --request-log-jsonl log.json --default-thinking-budget 16384 --thinking-budget-message "Considering the limited time available to the user, I must stop thinking now. Time to act:" --tolerant-tool-calls
```

It runs DFlash2 speculative decoding with 7 draft tokens, the indexed proposal head (`--lm-head-draft`) and
n-gram copy drafting, int8 KV, and the hybrid prefix cache with a 52,000 MiB tier in system RAM
(`--host-cache-mib`). `--kv-capacity auto` sizes the KV pool to the free VRAM. Lower `--host-cache-mib` on a
machine with less system RAM, and on a single 32 GB GPU stop any other resident model first.
`ninfer-serve.exe --help` lists every option.

- **Images and video:** add `--vision`. `--vision-offload on` keeps the vision tower in system RAM instead of
  VRAM, which frees about 0.8 GiB of VRAM for the KV cache.
- **Longer context:** beyond the native 262,144 tokens with `--rope-yarn-factor F` (F from 1 to 4, up to 1M
  tokens).

## Licence

Use of the weights is governed by the
[NVIDIA Open Model Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/),
as for the source checkpoint; the Qwen3.8-27B base model is licensed under Apache 2.0 (`LICENSE`, copied from the
source checkpoint). See `NOTICE`.

---

## Original model card (copied verbatim from nvidia/Qwen3.8-27B-NVFP4-QAD)

> The following is the README of the source checkpoint
> [`nvidia/Qwen3.8-27B-NVFP4-QAD`](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4-QAD),
> copied from its source repository for reference.

---
pipeline_tag: text-generation
base_model:
- Qwen/Qwen3.8-27B
license: other
license_name: nvidia-open-model-license
library_name: Model Optimizer
tags:
- nvidia
- ModelOpt
- Qwen3.8
- quantized
- NVFP4
- nvfp4
- QAD
---

# Model Overview

## Description:
The NVIDIA Qwen3.8-27B-NVFP4-QAD model is the quantized version of Alibaba's Qwen3.8-27B model, which is an auto-regressive language model that uses an optimized transformer architecture. For more information, please check [here](https://huggingface.co/Qwen/Qwen3.8-27B). The NVIDIA Qwen3.8-27B-NVFP4-QAD model is quantized with [Model Optimizer](https://github.com/NVIDIA/Model-Optimizer) using [quantization-aware distillation (QAD)](https://developer.nvidia.com/blog/how-quantization-aware-training-enables-low-precision-accuracy-recovery/).

This model is ready for commercial or non-commercial use. <br>

## License/Terms of Use:
**GOVERNING DOWNLOAD TERMS:** Use of this model is governed by the [NVIDIA Open Model Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/).

**ADDITIONAL INFORMATION** : The Qwen3.8-27B base model is governed by the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0). **Qwen3.8-27B**.

## Deployment Geography:
Global <br>

## Use Case:
Developers looking to take off-the-shelf, pre-quantized models for deployment in AI Agent systems, chatbots, RAG systems, and other AI-powered applications. <br>

## Release Date:
Huggingface **11/04/2026** via https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4-QAD <br>

## References
NVIDIA Model Optimizer: https://github.com/NVIDIA/Model-Optimizer

## Model Architecture:
**Architecture Type:** Transformer <br>
**Network Architecture:** Qwen3.8-27B (`Qwen3_5ForConditionalGeneration`) <br>
**Number of Model Parameters:** 27B <br>
**This model was developed based on [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B).** <br>

<!--
## Computational Load (Internal Only: For NVIDIA Models Only; please add as an HTML comment and remove the fields below from the published model card)

**Cumulative Compute:** **[XXX]** <br>

**Estimated Energy and Emissions for Model Training:** **[XXX]** <br>
-->

## Input:
**Input Type(s):** Text, Image, Video <br>
**Input Format(s):** String, Red, Green, Blue (RGB), Video (MP4/WebM) <br>
**Input Parameters:** Text: One-Dimensional (1D); Image: Two-Dimensional (2D); Video: Three-Dimensional (3D) <br>
**Other Properties Related to Input:** Context length up to 262,144 tokens <br>

## Output:
**Output Type(s):** Text <br>
**Output Format:** String <br>
**Output Parameters:** 1D (One-Dimensional): Sequences <br>
**Other Properties Related to Output:** Maximum completion length is configurable up to 262,144 tokens. <br>

Our AI models are designed and/or optimized to run on NVIDIA GPU-accelerated systems. By leveraging NVIDIA's hardware (e.g. GPU cores) and software frameworks (e.g., CUDA libraries), the model achieves faster training and inference times compared to CPU-only solutions. <br>

## Software Integration:
**Runtime Engine(s):** <br>
* vLLM <br>

**Supported Hardware Microarchitecture Compatibility:** <br>
* NVIDIA Blackwell <br>

**Supported Operating System(s):** <br>
* Linux <br>

The integration of foundation and fine-tuned models into AI systems requires additional testing using use-case-specific data to ensure safe and effective deployment. Following the V-model methodology, iterative testing and validation at both unit and system levels are essential to mitigate risks, meet technical and functional requirements, and ensure compliance with safety and ethical standards before deployment.

## Model Version(s):
The model version is NVFP4 QAD 1.0 and is quantized with NVIDIA-ModelOpt **v0.48.0** <br>

## Calibration Dataset:
** Link: [Nemotron-Post-Training-Dataset-v3](https://huggingface.co/collections/nvidia/nemotron-post-training-v3) <br>
** Data Collection Method by dataset: Automated. <br>
** Labeling Method by dataset: Automated. <br>
** Properties: The Nemotron-Post-Training-Dataset-v3 is a post-training dataset curated by NVIDIA containing multi-turn conversations across diverse topics. <br>

## Training Dataset:
** Link: [Nemotron-Cascade-2-SFT-Data](https://huggingface.co/datasets/nvidia/Nemotron-Cascade-2-SFT-Data) <br>
** Data Modality: Text <br>
** Text Training Data Size: 1.26B tokens <br>
** Data Collection Method by dataset: Hybrid: Automated, Manually-Collected, Synthetic <br>
** Labeling Method by dataset: Hybrid: Automated, Manually-Labeled, Synthetic <br>
** Properties: English-language multi-turn conversations spanning math, science, chat, instruction following, safety, conversational agent, software engineering, and terminal agent tasks. The data does not include personal data. <br>

## Inference:
**Acceleration Engine:** vLLM <br>
**Test Hardware:** NVIDIA Blackwell GB300 <br>

## Evaluation Dataset:
* Datasets: GPQA Diamond, IFBench, MMMU Pro, SciCode, Terminal-Bench 2.1, AA-LCR <br>
** Data Collection Method by dataset: Hybrid: Automated, Manually-Collected <br>
** Labeling Method by dataset: Hybrid: Manually-Labeled, Automated <br>
** Properties: We evaluated the model on English-language reasoning, coding, agentic, long-context, and multimodal benchmarks: GPQA Diamond contains 198 graduate-level multiple-choice questions written by domain experts in biology, physics, and chemistry; IFBench is a benchmark for evaluating instruction-following capabilities across diverse and structured task constraints; MMMU Pro is the more challenging version of the Massive Multi-discipline Multimodal Understanding benchmark, measuring college-level multimodal reasoning across diverse disciplines with expanded answer choices and a vision-only input setting; SciCode evaluates scientific coding capabilities; Terminal-Bench 2.1 evaluates agentic task completion in terminal environments; AA-LCR (Artificial Analysis Long Context Recall) evaluates a model's ability to accurately retrieve and recall information from long input contexts. <br>

## Quantization-Aware Distillation
This model was obtained by quantization-aware distillation of Qwen3.8-27B to NVFP4, ready for inference with vLLM. The BF16 teacher supplies token-level probability targets. The trainable NVFP4 weights are quantized and reconstructed for each forward pass. Gradients update the underlying weights so their low-precision representations better match the teacher. Only the weights and activations of the linear operators within transformer blocks are quantized. This optimization reduces the disk size and GPU memory requirements by approximately 2.8x.

## Usage

To serve this checkpoint with [vLLM](https://github.com/vllm-project/vllm), you can start the docker `vllm/vllm-openai:nightly` and run the sample command below:

```sh
vllm serve nvidia/Qwen3.8-27B-NVFP4-QAD --port 8000 --quantization modelopt --max-model-len 262144 --reasoning-parser qwen3  
```

## Evaluation
<table>
  <tr>
    <td><strong>Benchmark</strong></td>
    <td><strong>BF16</strong></td>
    <td><strong>NVFP4 QAD</strong></td>
  </tr>
  <tr>
    <td>GPQA Diamond</td>
    <td>88.19</td>
    <td>87.94</td>
  </tr>
  <tr>
    <td>IFBench</td>
    <td>79.25</td>
    <td>79.04</td>
  </tr>
  <tr>
    <td>MMMU Pro</td>
    <td>75.78</td>
    <td>75.61</td>
  </tr>
  <tr>
    <td>SciCode</td>
    <td>48.34</td>
    <td>48.82</td>
  </tr>
  <tr>
    <td>Terminal-Bench 2.1</td>
    <td>77.25</td>
    <td>76.69</td>
  </tr>
  <tr>
    <td>AA-LCR</td>
    <td>78.63</td>
    <td>79.13</td>
  </tr>
</table>

> Baseline: [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B).

## Model Limitations:
The base model was trained on data that contains toxic language and societal biases originally crawled from the internet. Therefore, the model may amplify those biases and return toxic responses especially when prompted with toxic prompts. The model may generate answers that may be inaccurate, omit key information, or include irrelevant or redundant text producing socially unacceptable or undesirable text, even if the prompt itself does not include anything explicitly offensive.

## Ethical Considerations

NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. Developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.

For more detailed information on ethical considerations for this model, please see the Model Card++ Explainability, Bias, Safety & Security, and Privacy Subcards below.

Please report model quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).

SUBCARDS:

## Explainability

| Field | Response |
| :---- | :---- |
| Intended Task/Domain: | Text generation, reasoning, summarization, and question answering. |
| Model Type: | Text, Image, and Video-to-text transformer |
| Intended Users: | This model is intended for developers, researchers, and customers building/utilizing LLMs, while balancing accuracy and efficiency. |
| Output: | Text String(s) |
| Describe how the model works: | Generates text by predicting the next word or token based on the context provided in the input sequence using multiple self-attention layers. Quantization-aware distillation trains the NVFP4 weights so their low-precision representations match a BF16 teacher. |
| Name the adversely impacted groups this has been tested to deliver comparable outcomes regardless of: | Not Applicable |
| Technical Limitations & Mitigation: | The model was trained on data that contains toxic language and societal biases originally crawled from the internet. Therefore, the model may amplify those biases and return toxic responses especially when prompted with toxic prompts. Therefore, before deploying any applications of this model, developers should perform safety testing and tuning tailored to their specific applications of the model. |
| Verified to have met prescribed NVIDIA quality standards: | Yes |
| Performance Metrics: | Accuracy, Throughput, and user-side throughput |
| Potential Known Risk | The model may generate answers that may be inaccurate, omit key information, or include irrelevant or redundant text producing socially unacceptable or undesirable text, even if the prompt itself does not include anything explicitly offensive. |
| Licensing: | **GOVERNING DOWNLOAD TERMS:** Use of this model is governed by the [NVIDIA Open Model Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/).

**ADDITIONAL INFORMATION** : The Qwen3.8-27B base model is governed by the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0). **Qwen3.8-27B**. |

## Bias

| Field | Response |
| :---- | :---- |
| Participation considerations from adversely impacted groups [protected classes](https://www.senate.ca.gov/protected-classes) in model design and testing: | None |
| Measures taken to mitigate against unwanted bias: | None |
| Bias Metric (If Measured): | Not Applicable |

## Safety & Security

| Field | Response |
| :---- | :---- |
| Model Application Field(s): | Chat, Instruction Following, Chatbot Development, Code Generation, Reasoning, Customer Service |
| Describe the life critical impact (if present). | Not Applicable |
| Use Case Restrictions: | **GOVERNING DOWNLOAD TERMS:** Use of this model is governed by the [NVIDIA Open Model Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/).

**ADDITIONAL INFORMATION** : The Qwen3.8-27B base model is governed by the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0). **Qwen3.8-27B**. |
| Model and dataset restrictions: | The Principle of least privilege (PoLP) is applied limiting access for dataset generation and model development.  Restrictions enforce dataset access during training, and dataset license constraints adhered to. |

## Privacy

The datasets used to train this model were reviewed before release. Please review NVIDIA’s Privacy Policy (https://www.nvidia.com/en-us/about-nvidia/privacy-policy/) for more information.
