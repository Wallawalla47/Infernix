#include "models/qwen3_5/program/kv_dump.h"

#include "core/device.h"
#include "core/dtype.h"
#include "core/temporary_switches.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace infernix::models::qwen3_5::detail {
namespace {

const char* dtype_name(DType dtype) {
    switch (dtype) {
    case DType::BF16:
        return "bf16";
    case DType::FP16:
        return "fp16";
    default:
        return "other";
    }
}

std::string json_string(std::string_view text) {
    std::string out = "\"";
    for (const char c : text) {
        if (c == '"' || c == '\\') {
            out += '\\';
            out += c;
        } else if (static_cast<unsigned char>(c) < 0x20) {
            char escaped[8];
            std::snprintf(escaped, sizeof(escaped), "\\u%04x", static_cast<unsigned>(c));
            out += escaped;
        } else {
            out += c;
        }
    }
    return out + "\"";
}

// One plane's pages, in logical order (`table`: their physical page ids), each its page-major payload
// [kv head][slot][dim].
void write_plane(const Tensor& plane, const std::vector<std::int32_t>& table,
                 const std::filesystem::path& path, cudaStream_t stream) {
    const auto element = static_cast<std::int64_t>(dtype_size(plane.dtype));
    const std::int64_t page_bytes =
        static_cast<std::int64_t>(plane.ne[0]) * plane.ne[1] * plane.ne[2] * element;
    if (plane.ne[1] != kPagedKVPageSize || plane.nb[0] != element ||
        plane.nb[1] != plane.ne[0] * element || plane.nb[2] != plane.ne[1] * plane.nb[1] ||
        plane.nb[3] < page_bytes) {
        throw std::logic_error("INFERNIX_TMP_KV_DUMP: the KV plane is not page-major");
    }
    std::vector<unsigned char> host(static_cast<std::size_t>(page_bytes) * table.size());
    for (std::size_t page = 0; page < table.size(); ++page) {
        const auto* source =
            static_cast<const unsigned char*>(plane.data) + static_cast<std::int64_t>(table[page]) * plane.nb[3];
        CUDA_CHECK(cudaMemcpyAsync(host.data() + page * static_cast<std::size_t>(page_bytes), source,
                                   static_cast<std::size_t>(page_bytes), cudaMemcpyDeviceToHost, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(host.data()), static_cast<std::streamsize>(host.size()));
    if (!out) { throw std::runtime_error("INFERNIX_TMP_KV_DUMP: cannot write " + path.string()); }
}

} // namespace

const std::string& kv_dump_directory() {
    static const std::string directory = temporary_switches::value("INFERNIX_TMP_KV_DUMP").value_or(std::string());
    return directory;
}

void dump_prompt_kv(const PagedKVCache& cache, const PagedKVCacheView& view, std::uint32_t tokens,
                    const TextConfig& text, std::string_view model_name, cudaStream_t stream) {
    const std::string& directory = kv_dump_directory();
    if (directory.empty() || tokens == 0 || cache.layers() == 0) { return; }
    const PagedKVLayerView first = view.layer_view(0);
    if (first.storage != KvCacheStorage::BFloat16 || first.k_pages.dtype != DType::BF16) {
        static std::atomic<bool> reported{false};
        if (!reported.exchange(true)) {
            std::fprintf(stderr, "INFERNIX_TMP_KV_DUMP: the KV cache is not BF16 (--kv-dtype bf16); nothing written\n");
        }
        return;
    }

    const std::uint32_t pages = (tokens + kPagedKVPageSize - 1) / kPagedKVPageSize;
    if (first.block_table.data == nullptr || static_cast<std::uint32_t>(first.block_table.ne[0]) < pages) {
        throw std::logic_error("INFERNIX_TMP_KV_DUMP: the sequence's block table is shorter than its prompt");
    }
    std::vector<std::int32_t> table(pages);
    CUDA_CHECK(cudaMemcpyAsync(table.data(), first.block_table.data, pages * sizeof(std::int32_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // kv-<process start, epoch seconds>-<prompt ordinal of this process>.
    static const long long process = std::chrono::duration_cast<std::chrono::seconds>(
                                         std::chrono::system_clock::now().time_since_epoch())
                                         .count();
    static std::atomic<unsigned> ordinal{0};
    const std::string tag = "kv-" + std::to_string(process) + "-" + std::to_string(ordinal++);
    const std::filesystem::path root(directory);
    std::filesystem::create_directories(root);

    std::vector<std::uint32_t> model_layers;
    for (std::size_t layer = 0; layer < text.layer_types.size(); ++layer) {
        if (text.layer_types[layer] == MixerKind::FullAttention) {
            model_layers.push_back(static_cast<std::uint32_t>(layer));
        }
    }

    std::ostringstream meta;
    meta << "{\n  \"format\": \"infernix-kv-dump-1\",\n"
         << "  \"model\": " << json_string(model_name) << ",\n"
         << "  \"tokens\": " << tokens << ",\n"
         << "  \"page_size\": " << kPagedKVPageSize << ",\n"
         << "  \"pages\": " << pages << ",\n"
         << "  \"kv_heads\": " << first.num_kv_heads << ",\n"
         << "  \"head_dim\": " << first.head_dim << ",\n"
         << "  \"rotary_dim\": " << (text.rope_parameters ? text.rope_parameters->rotary_dim : 0) << ",\n"
         << "  \"key_dtype\": \"" << dtype_name(first.k_pages.dtype) << "\",\n"
         << "  \"value_dtype\": \"" << dtype_name(first.v_pages.dtype) << "\",\n"
         << "  \"page_layout\": \"[kv_head][slot][dim]\",\n"
         << "  \"layers\": [\n";
    for (std::uint32_t layer = 0; layer < cache.layers(); ++layer) {
        const PagedKVLayerView kv = view.layer_view(layer);
        char suffix[32];
        std::snprintf(suffix, sizeof(suffix), "-L%02u", layer);
        const std::string key   = tag + suffix + "-K.bin";
        const std::string value = tag + suffix + "-V.bin";
        write_plane(kv.k_pages, table, root / key, stream);
        write_plane(kv.v_pages, table, root / value, stream);
        meta << "    {\"ordinal\": " << layer << ", \"model_layer\": "
             << (layer < model_layers.size() ? static_cast<long long>(model_layers[layer]) : -1LL)
             << ", \"key\": " << json_string(key) << ", \"value\": " << json_string(value) << "}"
             << (layer + 1 < cache.layers() ? ",\n" : "\n");
    }
    meta << "  ]\n}\n";
    const std::filesystem::path meta_path = root / (tag + ".json");
    std::ofstream out(meta_path, std::ios::trunc);
    out << meta.str();
    if (!out) { throw std::runtime_error("INFERNIX_TMP_KV_DUMP: cannot write " + meta_path.string()); }
    std::fprintf(stderr, "INFERNIX_TMP_KV_DUMP: %u tokens, %u layers -> %s\n", tokens, cache.layers(),
                 meta_path.string().c_str());
}

} // namespace infernix::models::qwen3_5::detail
