#pragma once

// Finds Infernix/NInfer v3 artifacts, Qwen3.8-Flash-Next n-gram volumes and LoRA adapters in the
// model folders by reading their headers only: models (a `text` component), companion drafters (no
// `text`), n-gram volumes and adapter folders (adapter_config.json), with the facts the launcher
// shows, matches and configures by.

#include "launcher/util.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace infernix::launcher {

struct NgramTableFacts {
    std::string format; // the artifact's own table: "fp8", "iq4_nl", "bf16", "nvfp4", "q8_0" or "unknown"
    std::uint64_t rows      = 0;
    std::uint32_t row_bytes = 0;
    std::string volume_id; // 32 hex digits
};

struct ScannedModel {
    std::string key; // artifact_id hex
    std::string name;
    std::string file_name;
    std::filesystem::path path;
    std::vector<std::filesystem::path> locations;
    std::string format; // "infernix" | "ninfer"
    std::string family;
    std::string family_label;
    std::string model_type;
    std::string architecture;
    std::optional<std::string> recipe;
    std::optional<std::string> quantization; // "NVFP4 + FP8": the weight formats holding most parameters
    std::uint64_t bytes   = 0;
    std::size_t file_count = 0;
    std::vector<std::filesystem::path> part_files; // the continuation files present, beside `path`
    std::vector<std::string> missing_files;
    std::vector<std::string> components;
    bool has_vision          = false;
    bool has_mtp             = false;
    bool has_dflash          = false;
    bool has_dflash2         = false;
    bool has_proposal        = false;
    bool offloaded_experts   = false;
    bool needs_ngram_volume  = false;
    bool accepts_draft_model = false;
    std::uint64_t native_context = 0;
    std::uint64_t hidden_size    = 0;
    std::uint64_t num_layers     = 0;
    std::uint64_t vocab_size     = 0;
    std::optional<std::uint64_t> num_experts;
    std::optional<NgramTableFacts> ngram_table;
    std::optional<std::filesystem::path> expert_state;
    std::optional<std::filesystem::path> own_ngram_volume;
    std::int64_t modified = 0;
};

struct ScannedDrafter {
    std::string key;
    std::string name;
    std::string file_name;
    std::filesystem::path path;
    std::uint64_t bytes = 0;
    std::string kind; // components joined with "+"
    std::optional<std::string> target_architecture;
    std::uint64_t target_hidden_size = 0;
    std::uint64_t target_layers      = 0;
    std::uint64_t target_vocab       = 0;
    std::vector<std::string> compatible_models;
    std::int64_t modified = 0;
};

struct ScannedNgramVolume {
    std::string key; // volume_id hex
    std::string name;
    std::string file_name;
    std::filesystem::path path;
    std::uint64_t bytes          = 0;
    std::uint64_t expected_bytes = 0; // header + blocks from the header's geometry
    std::string format;     // "fp8" | "iq4_nl" | "bf16" | "nvfp4" | "q8_0" | "bf16_p12" | "unknown"
    std::uint16_t tag       = 0;
    std::uint16_t fp8_scale = 0;
    std::uint64_t rows      = 0;
    std::uint32_t row_bytes = 0;
    std::vector<std::string> compatible_models;
    std::vector<std::string> own_for;
    std::int64_t modified = 0;
};

// What a PEFT LoRA adapter's adapter_config.json says (each field absent when the file does not say it).
struct AdapterConfigFacts {
    std::optional<std::string> peft_type;
    std::optional<std::string> base_model; // base_model_name_or_path
    std::optional<std::uint64_t> rank;     // r
    std::optional<std::uint64_t> max_rank; // the largest of r and the rank_pattern's ranks
    std::optional<double> alpha;           // lora_alpha
    std::vector<std::string> target_modules; // when the config lists them (not a pattern)
    bool rslora = false;
};

// A Hugging Face PEFT LoRA adapter (`--lora`): a folder holding adapter_config.json and adapter_model.safetensors.
// The launcher shows what its config says; infernix-serve decides whether it can apply it (--plan).
struct ScannedAdapter {
    std::string key;            // 32 hex digits of its folder's path's SHA-256 (an adapter has no id of its own)
    std::string name;           // the folder's name: the repository's, when the launcher downloaded it
    std::filesystem::path path; // the folder
    std::uint64_t bytes = 0;    // adapter_model.safetensors
    AdapterConfigFacts config;
    std::vector<std::string> compatible_models; // the dense models of the base model it names
    std::int64_t modified = 0;
};

struct ScannedDir {
    std::filesystem::path path;
    bool exists               = false;
    std::size_t models        = 0;
    std::size_t drafters      = 0;
    std::size_t ngram_volumes = 0;
    std::size_t adapters      = 0;
    // The room on its drive when it was scanned (nullopt when the drive did not say).
    std::optional<std::uint64_t> free_bytes;
    std::optional<std::uint64_t> total_bytes;
    // Why it was not scanned, or scanned only in part.
    std::optional<std::string> error;
};

struct ScanIssue {
    std::filesystem::path path;
    std::string message;
};

struct LibraryScan {
    std::int64_t scanned_at = 0;
    std::vector<ScannedDir> dirs;
    std::vector<ScannedModel> models;
    std::vector<ScannedDrafter> drafters;
    std::vector<ScannedNgramVolume> ngram_volumes;
    std::vector<ScannedAdapter> adapters;
    std::vector<ScanIssue> issues;

    [[nodiscard]] const ScannedModel* model(std::string_view key) const;
    [[nodiscard]] const ScannedDrafter* drafter(std::string_view key) const;
};

// Scans each folder to `depth` levels of subfolders (0: the folder itself), reading headers only.
[[nodiscard]] LibraryScan scan_library(const std::vector<std::filesystem::path>& dirs, int depth);

// The rank of an n-gram volume for a model (lower is better): its own table first, then FP8, NVFP4,
// IQ4_NL, Q8_0, BF16 P12 and BF16; nullopt when the model cannot read it.
[[nodiscard]] std::optional<int> ngram_rank(const ScannedModel& model, const ScannedNgramVolume& volume);

// Parses one artifact entry's directory JSON into a model or drafter (exposed for tests). `path` is
// the entry; its continuation files are checked beside it.
struct ParsedEntry {
    std::optional<ScannedModel> model;
    std::optional<ScannedDrafter> drafter;
};
[[nodiscard]] ParsedEntry parse_artifact_directory(const std::filesystem::path& path,
                                                   const std::array<std::uint8_t, 16>& artifact_id,
                                                   const Json& directory, std::uint64_t entry_payload_offset);

// Parses the first bytes of an n-gram volume (nullopt when it is not one).
[[nodiscard]] std::optional<ScannedNgramVolume> parse_ngram_header(const std::filesystem::path& path,
                                                                   std::string_view header);

// The facts of an adapter_config.json (fields of the wrong type are left out).
[[nodiscard]] AdapterConfigFacts parse_adapter_config(const Json& config);
[[nodiscard]] Json to_json(const AdapterConfigFacts& facts);

// Whether an adapter was trained on this model's base: a dense Qwen3.5-architecture model (the --lora models) whose
// family (Qwen3.8-27B, Qwen3.6-27B) the adapter's base_model_name_or_path names.
[[nodiscard]] bool adapter_matches(const ScannedModel& model, const ScannedAdapter& adapter);

// The API's Library object without each model's `config` (the launcher adds it).
[[nodiscard]] Json to_json(const LibraryScan& scan);

} // namespace infernix::launcher
