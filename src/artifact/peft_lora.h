#pragma once

#include "artifact/safetensors.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace infernix::artifact {

// One adapted module of a Hugging Face PEFT LoRA adapter: W x becomes W x + scale * B (A x), with A [rank, in] and
// B [out, rank] stored as the tensors named here.
struct PeftLoraModule {
    std::string module;   // the module's path without PEFT's "base_model.model." prefix, e.g.
                          // "model.language_model.layers.3.self_attn.q_proj"
    std::string a_tensor; // "<prefix><module>.lora_A.weight"
    std::string b_tensor; // "<prefix><module>.lora_B.weight"
    std::uint32_t rank  = 0;
    std::uint64_t in    = 0; // A's columns
    std::uint64_t out   = 0; // B's rows
    float alpha         = 0;
    float scale         = 0; // alpha / rank, or alpha / sqrt(rank) with rsLoRA (PEFT's `scaling`)
};

// A PEFT LoRA adapter as Hugging Face stores it: `adapter_model.safetensors` with `adapter_config.json` beside it.
// The constructor validates the configuration and pairs the tensors; it refuses every feature that is not plain LoRA
// on linear weights (DoRA, LoRA biases, modules_to_save, trainable tokens, layer replication, fan_in_fan_out,
// aLoRA, QALoRA, target_parameters, ...) and every tensor it cannot place, naming it. Tensor data is read on request.
class PeftLoraAdapter {
public:
    // `path`: the adapter's directory, or its .safetensors file (adapter_config.json must lie beside it). A GGUF
    // adapter (llama.cpp's export) is refused by name.
    explicit PeftLoraAdapter(const std::filesystem::path& path);
    ~PeftLoraAdapter();
    PeftLoraAdapter(const PeftLoraAdapter&)            = delete;
    PeftLoraAdapter& operator=(const PeftLoraAdapter&) = delete;

    [[nodiscard]] const std::filesystem::path& weights_path() const noexcept { return file_->path(); }
    [[nodiscard]] const std::filesystem::path& config_path() const noexcept { return config_path_; }
    [[nodiscard]] const SafetensorsFile& file() const noexcept { return *file_; }
    [[nodiscard]] const std::vector<PeftLoraModule>& modules() const noexcept { return modules_; }
    [[nodiscard]] const std::string& base_model() const noexcept { return base_model_; }
    [[nodiscard]] const std::string& peft_version() const noexcept { return peft_version_; }
    [[nodiscard]] bool rslora() const noexcept { return rslora_; }
    // The weights file's size and last-write time (seconds since the epoch), which name its contents for the saved
    // prefix cache's fingerprint as the artifact's do.
    [[nodiscard]] std::uint64_t weights_bytes() const noexcept;
    [[nodiscard]] std::int64_t weights_mtime() const noexcept { return weights_mtime_; }
    // The sum of every module's A and B elements.
    [[nodiscard]] std::uint64_t parameter_count() const noexcept;
    // The stored element type of the factors ("F32", "BF16" or "F16"; one type for every factor).
    [[nodiscard]] const std::string& dtype() const noexcept { return dtype_; }

private:
    std::filesystem::path config_path_;
    std::unique_ptr<SafetensorsFile> file_;
    std::vector<PeftLoraModule> modules_;
    std::string base_model_, peft_version_, dtype_;
    bool rslora_                = false;
    std::int64_t weights_mtime_ = 0;
};

} // namespace infernix::artifact
