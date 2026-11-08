// The prefix cache file (docs/maintainer/hybrid-prefix-cache-spec.md §5.5): one layout for every model
// binding (candidates RE-03; it replaced Qwen3.5's version 4 and Qwen3.8-Flash-Next's version 2 copies of
// the same code). Nothing in it is trusted unless the fingerprint (artifact, KV and state formats,
// binary) and the binding's Host geometry match; a damaged or foreign file loads nothing.
//
// Layout (version 5): the header (the family's magic, version, fingerprint, geometry, the SSD section's
// flag and the SSD tier's slab count), the block table (parents first; each with its family word, whether
// the file holds its Host slab, and its SSD slab and hash), the snapshot table (retained snapshots first,
// then superseded ones oldest first; each with its kind, hits, flags, SSD slabs and hashes and its family
// meta), then the Host slab bytes of every block and snapshot that has them, in table order, then the
// footer. The tables come first so a Host tier smaller than the file can choose what to restore before
// reading any slab.

#include "runtime/prefix_cache/persist_file.h"

#include "core/content_hash.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unordered_map>
#include <utility>

namespace infernix::runtime::prefix_cache {
namespace {

constexpr std::uint32_t kVersion      = 5;
constexpr std::array<char, 8> kFooter = {'N', 'I', 'N', 'F', 'P', 'C', 'E', 'D'};
constexpr std::int32_t kRootIndex     = -1;
constexpr std::uint32_t kNone         = kNoId;
constexpr std::uint32_t kFlagSsd      = 1U;
constexpr std::uint64_t kFooterBytes  = kFooter.size() + 2U * sizeof(std::uint32_t);
constexpr const char* kSaveAbandoned  = "abandoned; the previous file is kept";
// Smallest table entries: a block (parent, hash, extra, word, tokens, host, SSD slab, SSD hash) and a
// snapshot without a tail, SSD slabs or meta. They bound a damaged header's counts before allocating.
constexpr std::uint64_t kMinBlockBytes    = 4 + 8 + 8 + 4 + kBlockTokens * sizeof(TokenId) + 1 + 4 + 8;
constexpr std::uint64_t kMinSnapshotBytes = 4 + 4 + 4 + 1 + 4 + 1 + 1 + 4 + 1;

class Writer {
public:
    explicit Writer(const std::filesystem::path& path) : out_(path, std::ios::binary | std::ios::trunc) {
        if (!out_) { throw std::runtime_error("cannot create " + path.string()); }
    }

    template <class T>
    void value(const T& value) {
        bytes(&value, sizeof(T));
    }

    void bytes(const void* data, std::size_t count) {
        out_.write(static_cast<const char*>(data), static_cast<std::streamsize>(count));
        if (!out_) { throw std::runtime_error("prefix cache file write failed"); }
        written_ += count;
    }

    void finish() {
        out_.flush();
        if (!out_) { throw std::runtime_error("prefix cache file flush failed"); }
        out_.close();
    }

    [[nodiscard]] std::uint64_t written() const noexcept { return written_; }

private:
    std::ofstream out_;
    std::uint64_t written_ = 0;
};

class Reader {
public:
    explicit Reader(const std::filesystem::path& path) : in_(path, std::ios::binary) {}

    [[nodiscard]] bool open() const noexcept { return static_cast<bool>(in_); }

    template <class T>
    [[nodiscard]] T value() {
        T out{};
        bytes(&out, sizeof(T));
        return out;
    }

    void bytes(void* data, std::size_t count) {
        in_.read(static_cast<char*>(data), static_cast<std::streamsize>(count));
        if (!in_) { throw std::runtime_error("prefix cache file is truncated"); }
        read_ += count;
        position_ += count;
    }

    void seek(std::uint64_t offset) {
        if (offset == position_) { return; }
        in_.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
        if (!in_) { throw std::runtime_error("prefix cache file is truncated"); }
        position_ = offset;
    }

    // Bytes read, not counting what seeks passed over.
    [[nodiscard]] std::uint64_t read() const noexcept { return read_; }
    [[nodiscard]] std::uint64_t position() const noexcept { return position_; }

private:
    std::ifstream in_;
    std::uint64_t read_     = 0;
    std::uint64_t position_ = 0;
};

struct SavedBlock {
    std::int32_t parent = kRootIndex;
    std::uint64_t hash  = 0;
    std::uint64_t extra = 0;
    std::int32_t word   = -1;
    std::array<TokenId, kBlockTokens> tokens{};
    bool host              = false;
    std::uint32_t ssd_slab = kNone;
    std::uint64_t ssd_hash = 0;
    std::uint64_t host_slab_index = 0; // its slab's position among the file's Host slabs
};

struct SavedSnapshot {
    std::int32_t anchor    = kRootIndex;
    std::uint32_t frontier = 0;
    std::uint32_t tail_len = 0;
    std::array<TokenId, kBlockTokens> tail{};
    SnapshotKind kind = SnapshotKind::Tap;
    std::uint32_t hits = 0;
    bool superseded    = false;
    bool host          = false;
    bool ssd_packed    = false;
    std::vector<std::uint32_t> ssd_slabs;
    std::vector<std::uint64_t> ssd_hashes;
    std::vector<std::byte> meta;
    std::uint32_t host_slabs      = 0;
    std::uint64_t host_slab_index = 0; // its first slab's position among the file's Host slabs
};

double seconds_since(std::chrono::steady_clock::time_point started) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
}

std::uint32_t ssd_capacity_slabs(const PersistBinding& binding) {
    return binding.ssd == nullptr ? 0U : static_cast<std::uint32_t>(binding.ssd->capacity_bytes() / binding.slab_bytes);
}

} // namespace

std::filesystem::path kept_ssd_file(const std::filesystem::path& ssd_directory,
                                    const std::filesystem::path& prefix_cache_file) {
    const std::string name = std::filesystem::absolute(prefix_cache_file).generic_string();
    char hex[17]{};
    std::snprintf(hex, sizeof(hex), "%016llx",
                  static_cast<unsigned long long>(content_hash(reinterpret_cast<const std::byte*>(name.data()), name.size())));
    return ssd_directory / ("infernix-context-kept-" + std::string(hex) + ".bin");
}

PersistResult save_prefix_cache(const PrefixCacheIndex& index, const PersistBinding& binding,
                                const std::filesystem::path& path, std::string_view fingerprint,
                                const CancellationView& abandoned) {
    PersistResult out;
    const auto started = std::chrono::steady_clock::now();
    if (abandoned.requested()) {
        out.message = kSaveAbandoned;
        return out;
    }
    const auto check_abandoned = [&] {
        if (abandoned.requested()) { throw std::runtime_error(kSaveAbandoned); }
    };
    const bool ssd = binding.ssd != nullptr;
    std::filesystem::path temporary = path;
    temporary += ".tmp";
    try {
        // The tables name SSD slabs only once their bytes are durable.
        if (ssd) { binding.ssd->flush(); }
        std::vector<NodeRef> nodes;
        std::vector<SnapshotRef> snapshots;
        index.collect_persistable(nodes, snapshots, ssd);

        Writer writer(temporary);
        writer.bytes(binding.magic.data(), binding.magic.size());
        writer.value(kVersion);
        writer.value(static_cast<std::uint32_t>(fingerprint.size()));
        writer.bytes(fingerprint.data(), fingerprint.size());
        writer.value(static_cast<std::uint32_t>(binding.geometry.size()));
        writer.bytes(binding.geometry.data(), binding.geometry.size());
        writer.value(ssd ? kFlagSsd : 0U);
        writer.value(ssd_capacity_slabs(binding));
        writer.value(static_cast<std::uint32_t>(nodes.size()));
        writer.value(static_cast<std::uint32_t>(snapshots.size()));

        std::unordered_map<std::uint32_t, std::int32_t> file_index;
        file_index.reserve(nodes.size());
        for (const NodeRef node : nodes) {
            const BlockIdentity identity = index.block_identity(node);
            const NodeView view          = index.node(node);
            const bool host              = view.host == CopyState::Resident;
            const std::uint32_t ssd_slab = ssd && view.ssd == CopyState::Resident ? view.ssd_slab : kNone;
            writer.value(identity.parent.valid() ? file_index.at(identity.parent.index) : kRootIndex);
            writer.value(identity.lookup_hash);
            writer.value(identity.extra);
            writer.value(binding.block_word ? binding.block_word(node) : std::int32_t{-1});
            writer.bytes(identity.tokens.data(), identity.tokens.size_bytes());
            writer.value(static_cast<std::uint8_t>(host ? 1U : 0U));
            writer.value(ssd_slab);
            writer.value(ssd_slab != kNone ? binding.ssd->slab_hash(ssd_slab) : std::uint64_t{0});
            file_index.emplace(node.index, static_cast<std::int32_t>(file_index.size()));
            out.ssd_blocks += ssd_slab != kNone ? 1U : 0U;
        }
        std::vector<std::byte> meta(binding.meta_bytes);
        for (const SnapshotRef snapshot : snapshots) {
            const SnapshotView view = index.snapshot(snapshot);
            const bool host         = view.host == CopyState::Resident;
            if (host && view.host_slabs.size() != binding.image_slabs + (view.tail_len != 0 ? 1U : 0U)) {
                throw std::logic_error("a saved snapshot's Host slabs do not match its tail");
            }
            const bool on_ssd = ssd && view.ssd == CopyState::Resident;
            writer.value(view.anchor.valid() ? file_index.at(view.anchor.index) : kRootIndex);
            writer.value(view.frontier);
            writer.value(view.tail_len);
            writer.bytes(view.tail.data(), view.tail.size_bytes());
            writer.value(static_cast<std::uint8_t>(view.kind));
            writer.value(view.hits);
            writer.value(static_cast<std::uint8_t>(view.superseded ? 1U : 0U));
            writer.value(static_cast<std::uint8_t>(host ? 1U : 0U));
            writer.value(static_cast<std::uint32_t>(on_ssd ? view.ssd_slabs.size() : 0U));
            writer.value(static_cast<std::uint8_t>(on_ssd && view.ssd_packed ? 1U : 0U));
            if (on_ssd) {
                for (const std::uint32_t slab : view.ssd_slabs) {
                    writer.value(slab);
                    writer.value(binding.ssd->slab_hash(slab));
                }
                ++out.ssd_snapshots;
            }
            if (binding.meta_bytes != 0) {
                binding.snapshot_meta(snapshot, meta.data());
                writer.bytes(meta.data(), meta.size());
            }
        }
        for (const NodeRef node : nodes) {
            const NodeView view = index.node(node);
            if (view.host != CopyState::Resident) { continue; }
            check_abandoned();
            writer.bytes(binding.slab(view.host_slab), binding.slab_bytes);
        }
        for (const SnapshotRef snapshot : snapshots) {
            const SnapshotView view = index.snapshot(snapshot);
            if (view.host != CopyState::Resident) { continue; }
            for (const std::uint32_t slab_id : view.host_slabs) {
                check_abandoned();
                writer.bytes(binding.slab(slab_id), binding.slab_bytes);
            }
        }
        writer.bytes(kFooter.data(), kFooter.size());
        writer.value(static_cast<std::uint32_t>(nodes.size()));
        writer.value(static_cast<std::uint32_t>(snapshots.size()));
        writer.finish();
        out.bytes = writer.written();
        // Abandoned after the last slab: the previous file still stays.
        check_abandoned();
        std::filesystem::rename(temporary, path);
        out.blocks    = nodes.size();
        out.snapshots = snapshots.size();
    } catch (const std::exception& error) {
        // The writer is closed by now, so the temporary file can be deleted on every platform.
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        out         = PersistResult{};
        out.message = error.what();
        return out;
    }
    out.ok      = true;
    out.seconds = seconds_since(started);
    return out;
}

PersistResult load_prefix_cache(PrefixCacheIndex& index, const PersistBinding& binding,
                                const std::filesystem::path& path, std::string_view fingerprint,
                                const StartupObserver& observer) {
    PersistResult out;
    const auto started = std::chrono::steady_clock::now();
    Reader reader(path);
    if (!reader.open()) {
        out.message = "no saved prefix cache at " + path.string() + " yet; it is written at shutdown";
        return out;
    }
    // Opened once the header matches, so a missing or foreign file shows only the reason it was
    // not restored. A file damaged part way through completes it with the bytes that were read.
    std::optional<StartupPhaseScope> phase;
    try {
        std::array<char, 8> magic{};
        reader.bytes(magic.data(), magic.size());
        if (magic != binding.magic || reader.value<std::uint32_t>() != kVersion) {
            out.message = "not a prefix cache file of this format version";
            return out;
        }
        const auto fingerprint_bytes = reader.value<std::uint32_t>();
        if (fingerprint_bytes > (1U << 20U)) {
            out.message = "prefix cache file header is damaged";
            return out;
        }
        std::string saved(fingerprint_bytes, '\0');
        reader.bytes(saved.data(), saved.size());
        if (saved != fingerprint) {
            out.message = "saved prefix cache belongs to a different model, KV format or build";
            return out;
        }
        const auto geometry_bytes = reader.value<std::uint32_t>();
        if (geometry_bytes != binding.geometry.size()) {
            out.message = "saved prefix cache has a different Host geometry";
            return out;
        }
        std::vector<std::byte> geometry(geometry_bytes);
        reader.bytes(geometry.data(), geometry.size());
        if (geometry != binding.geometry) {
            out.message = "saved prefix cache has a different Host geometry";
            return out;
        }
        const auto flags         = reader.value<std::uint32_t>();
        const auto ssd_capacity  = reader.value<std::uint32_t>();
        const std::uint64_t file_bytes = std::filesystem::file_size(path);
        const std::uint64_t slab_bytes = binding.slab_bytes;
        const auto node_count          = reader.value<std::uint32_t>();
        const auto snapshot_count      = reader.value<std::uint32_t>();
        if (static_cast<std::uint64_t>(node_count) * kMinBlockBytes +
                static_cast<std::uint64_t>(snapshot_count) * kMinSnapshotBytes > file_bytes) {
            throw std::runtime_error("prefix cache file header is damaged");
        }
        // The file's SSD copies count only in the kept SSD tier's file of the run that saved them.
        bool use_ssd = false;
        if ((flags & kFlagSsd) != 0U) {
            if (binding.ssd == nullptr) {
                out.ssd_message = "this start has no kept SSD tier";
            } else if (!binding.ssd->kept_content()) {
                out.ssd_message = "the SSD tier's file " + binding.ssd->path().string() + " was not kept";
            } else if (ssd_capacity != ssd_capacity_slabs(binding)) {
                out.ssd_message = "the SSD tier has a different size";
            } else {
                use_ssd = true;
            }
        }

        std::vector<SavedBlock> blocks(node_count);
        std::vector<std::int32_t> parents(node_count);
        const std::unique_ptr<bool[]> block_host = std::make_unique<bool[]>(node_count); // all false
        std::uint64_t host_slab_count = 0;
        std::uint64_t ssd_saved       = 0;
        for (std::uint32_t i = 0; i < node_count; ++i) {
            SavedBlock& block = blocks[i];
            block.parent      = reader.value<std::int32_t>();
            block.hash        = reader.value<std::uint64_t>();
            block.extra       = reader.value<std::uint64_t>();
            block.word        = reader.value<std::int32_t>();
            reader.bytes(block.tokens.data(), sizeof(block.tokens));
            const auto host = reader.value<std::uint8_t>();
            block.ssd_slab  = reader.value<std::uint32_t>();
            block.ssd_hash  = reader.value<std::uint64_t>();
            if (block.parent >= static_cast<std::int32_t>(i) || block.parent < kRootIndex || host > 1U ||
                (host == 0U && block.ssd_slab == kNone)) {
                throw std::runtime_error("prefix cache file block table is damaged");
            }
            block.host = host != 0U;
            if (block.host) { block.host_slab_index = host_slab_count++; }
            ssd_saved += block.ssd_slab != kNone ? 1U : 0U;
            if (!use_ssd) { block.ssd_slab = kNone; }
            parents[i]    = block.parent;
            block_host[i] = block.host;
        }
        std::vector<SavedSnapshot> snapshots(snapshot_count);
        std::vector<SavedSnapshotShape> shapes(snapshot_count);
        for (std::uint32_t i = 0; i < snapshot_count; ++i) {
            SavedSnapshot& snapshot = snapshots[i];
            snapshot.anchor         = reader.value<std::int32_t>();
            snapshot.frontier       = reader.value<std::uint32_t>();
            snapshot.tail_len       = reader.value<std::uint32_t>();
            if (snapshot.tail_len >= kBlockTokens || snapshot.anchor >= static_cast<std::int32_t>(node_count) ||
                snapshot.anchor < kRootIndex) {
                throw std::runtime_error("prefix cache file snapshot is damaged");
            }
            reader.bytes(snapshot.tail.data(), static_cast<std::size_t>(snapshot.tail_len) * sizeof(TokenId));
            const auto kind_byte = reader.value<std::uint8_t>();
            snapshot.hits        = reader.value<std::uint32_t>();
            const auto superseded = reader.value<std::uint8_t>();
            const auto host       = reader.value<std::uint8_t>();
            const auto ssd_count  = reader.value<std::uint32_t>();
            const auto packed     = reader.value<std::uint8_t>();
            if (kind_byte > static_cast<std::uint8_t>(SnapshotKind::Boundary) || superseded > 1U || host > 1U ||
                packed > 1U || ssd_count > binding.image_slabs + 1U || (host == 0U && ssd_count == 0U)) {
                throw std::runtime_error("prefix cache file snapshot is damaged");
            }
            snapshot.kind       = static_cast<SnapshotKind>(kind_byte);
            snapshot.superseded = superseded != 0U;
            snapshot.host       = host != 0U;
            snapshot.ssd_packed = packed != 0U;
            snapshot.ssd_slabs.resize(ssd_count);
            snapshot.ssd_hashes.resize(ssd_count);
            for (std::uint32_t k = 0; k < ssd_count; ++k) {
                snapshot.ssd_slabs[k]  = reader.value<std::uint32_t>();
                snapshot.ssd_hashes[k] = reader.value<std::uint64_t>();
            }
            ssd_saved += ssd_count != 0U ? 1U : 0U;
            if (!use_ssd) { snapshot.ssd_slabs.clear(); }
            snapshot.meta.resize(binding.meta_bytes);
            reader.bytes(snapshot.meta.data(), snapshot.meta.size());
            snapshot.host_slabs = binding.image_slabs + (snapshot.tail_len != 0 ? 1U : 0U);
            if (snapshot.host) {
                snapshot.host_slab_index = host_slab_count;
                host_slab_count += snapshot.host_slabs;
            }
            shapes[i] = SavedSnapshotShape{.anchor     = snapshot.anchor,
                                           .frontier   = snapshot.frontier,
                                           .slabs      = snapshot.host_slabs,
                                           .hits       = snapshot.hits,
                                           .superseded = snapshot.superseded,
                                           .host       = snapshot.host};
        }
        const std::uint64_t payload = reader.position();
        if (file_bytes != payload + host_slab_count * slab_bytes + kFooterBytes) {
            throw std::runtime_error("prefix cache file is truncated or damaged");
        }
        if (ssd_saved == 0U) { out.ssd_message.clear(); }
        out.saved_blocks        = node_count;
        out.saved_snapshots     = snapshot_count;
        out.required_host_bytes = host_slab_count * slab_bytes;
        out.host_bytes          = static_cast<std::uint64_t>(binding.host_slabs) * slab_bytes;

        const HostRestorePlan plan = plan_host_restore(parents, shapes, binding.host_slabs, index.config(),
                                                       std::span<const bool>(block_host.get(), node_count));
        const std::uint64_t planned = reader.read() + plan.slabs * slab_bytes + kFooterBytes;
        phase.emplace(observer, StartupPhase::PrefixCacheLoad, StartupProgressUnit::Bytes, planned);
        phase->progress(reader.read(), planned);

        // Restored index ids of the file's blocks; absent when a block was neither chosen for the Host
        // tier nor kept on the SSD tier, or did not restore, so its descendants and the snapshots on it
        // are skipped too.
        std::vector<std::optional<NodeRef>> restored(node_count);
        for (std::uint32_t i = 0; i < node_count; ++i) {
            const SavedBlock& saved_block = blocks[i];
            const bool host               = plan.blocks[i];
            if ((!host && saved_block.ssd_slab == kNone) ||
                (saved_block.parent != kRootIndex && !restored[static_cast<std::size_t>(saved_block.parent)])) {
                continue;
            }
            const NodeRef parent =
                saved_block.parent == kRootIndex ? NodeRef{} : *restored[static_cast<std::size_t>(saved_block.parent)];
            const std::optional<RestoredBlock> block =
                index.restore_block(parent, saved_block.hash, saved_block.tokens, saved_block.extra, host, saved_block.ssd_slab);
            if (!block) { continue; }
            if (host) {
                reader.seek(payload + saved_block.host_slab_index * slab_bytes);
                reader.bytes(binding.slab(block->slab), slab_bytes);
            }
            if (saved_block.ssd_slab != kNone) {
                binding.ssd->expect(saved_block.ssd_slab, saved_block.ssd_hash);
                ++out.ssd_blocks;
            }
            if (binding.set_block_word) { binding.set_block_word(block->node, saved_block.word); }
            restored[i] = block->node;
            ++out.blocks;
            phase->progress(reader.read(), planned);
        }
        for (std::uint32_t i = 0; i < snapshot_count; ++i) {
            const SavedSnapshot& saved_snapshot = snapshots[i];
            const bool host                     = plan.snapshots[i];
            if ((!host && saved_snapshot.ssd_slabs.empty()) ||
                (saved_snapshot.anchor != kRootIndex && !restored[static_cast<std::size_t>(saved_snapshot.anchor)])) {
                continue;
            }
            const NodeRef anchor =
                saved_snapshot.anchor == kRootIndex ? NodeRef{} : *restored[static_cast<std::size_t>(saved_snapshot.anchor)];
            const std::optional<SnapshotRef> snapshot = index.restore_snapshot(
                anchor, saved_snapshot.frontier, std::span<const TokenId>(saved_snapshot.tail.data(), saved_snapshot.tail_len),
                saved_snapshot.kind, saved_snapshot.hits, saved_snapshot.superseded, host, saved_snapshot.ssd_slabs,
                saved_snapshot.ssd_packed);
            if (!snapshot) { continue; }
            if (host) {
                reader.seek(payload + saved_snapshot.host_slab_index * slab_bytes);
                for (const std::uint32_t slab_id : index.snapshot(*snapshot).host_slabs) {
                    reader.bytes(binding.slab(slab_id), slab_bytes);
                }
            }
            for (std::size_t k = 0; k < saved_snapshot.ssd_slabs.size(); ++k) {
                binding.ssd->expect(saved_snapshot.ssd_slabs[k], saved_snapshot.ssd_hashes[k]);
            }
            out.ssd_snapshots += saved_snapshot.ssd_slabs.empty() ? 0U : 1U;
            if (binding.restore_meta && !binding.restore_meta(*snapshot, saved_snapshot.meta.data(), host)) {
                throw std::runtime_error("prefix cache file snapshot meta is damaged");
            }
            ++out.snapshots;
            phase->progress(reader.read(), planned);
        }
        if (use_ssd) { index.finish_ssd_restore(); }
        reader.seek(payload + host_slab_count * slab_bytes);
        std::array<char, 8> footer{};
        reader.bytes(footer.data(), footer.size());
        if (footer != kFooter || reader.value<std::uint32_t>() != node_count ||
            reader.value<std::uint32_t>() != snapshot_count) {
            throw std::runtime_error("prefix cache file footer is damaged");
        }
        phase->complete(reader.read());
    } catch (const std::exception& error) {
        if (phase) { phase->complete(reader.read()); }
        // Partially restored entries may hold unread slab bytes: the caller drops them all.
        out         = PersistResult{};
        out.message = error.what();
        return out;
    }
    out.ok      = true;
    out.bytes   = reader.read();
    out.seconds = seconds_since(started);
    return out;
}

} // namespace infernix::runtime::prefix_cache
