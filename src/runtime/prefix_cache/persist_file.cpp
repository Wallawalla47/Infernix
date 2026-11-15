// The prefix cache file (docs/maintainer/hybrid-prefix-cache-spec.md §5.5): one layout for every model
// binding (candidates RE-03; it replaced Qwen3.5's version 4 and Qwen3.8-Flash-Next's version 2 copies of
// the same code). Nothing in it is trusted unless the fingerprint (artifact, KV and state formats,
// binary) and the binding's Host geometry match; a damaged or foreign file loads nothing.
//
// Layout (version 6): the head, a whole number of 4 KiB blocks, then the Host slab bytes of every block
// and snapshot that has them, in table order, so the file is a whole number of 4 KiB blocks and every
// slab starts on one. The head holds the header (the family's magic, version, the head's size,
// fingerprint, geometry, the SSD section's flag and the SSD tier's slab count, the counts), the block
// table (parents first; each with its family word, whether the file holds its Host slab and that slab's
// content hash, and its SSD slab and hash), the snapshot table (retained snapshots first, then superseded
// ones oldest first; each with its kind, hits, flags, SSD slabs and hashes, its family meta and its Host
// slabs' content hashes), the end mark, and zeros up to the next 4 KiB. The tables come first so a Host
// tier smaller than the file can choose what to restore before reading any slab.
//
// Both directions move the slabs between the drive and the Host tier's page-locked memory unbuffered
// (core::DirectScratchFile: no page cache, no copy), consecutive slabs of one chunk as one transfer of up
// to 8 MiB, several in flight. Each slab's content hash (core::content_hash, xxHash64, as the SSD tier's
// slabs) is computed on the transfer's thread, of the memory before a write and of the bytes a read
// delivered: the only pass over the bytes besides the device's. A save writes the head last, once the
// hashes are known; a load compares them, and an entry with a slab whose bytes changed is dropped.

#include "runtime/prefix_cache/persist_file.h"

#include "core/content_hash.h"
#include "core/direct_scratch_file.h"

#ifndef _WIN32
#    include <fcntl.h>
#    include <unistd.h>
#endif

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <new>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unordered_map>
#include <utility>

namespace infernix::runtime::prefix_cache {
namespace {

constexpr std::uint32_t kVersion      = 6;
constexpr std::array<char, 8> kFooter = {'N', 'I', 'N', 'F', 'P', 'C', 'E', 'D'};
constexpr std::int32_t kRootIndex     = -1;
constexpr std::uint32_t kNone         = kNoId;
constexpr std::uint32_t kFlagSsd      = 1U;
constexpr const char* kSaveAbandoned  = "abandoned; the previous file is kept";
constexpr const char* kKeptPrefix     = "infernix-context-kept-";
constexpr std::size_t kBlock          = DirectScratchFile::kAlignment;
// The head's size sits right after the magic and the version, inside the file's first block.
constexpr std::size_t kHeadBytesOffset = 8 + sizeof(std::uint32_t);
// Consecutive slabs join one transfer up to this size (several in flight reach a drive's sequential rate),
// and this many transfers are issued between two looks at the caller (an abandoned save, a load's progress).
constexpr std::size_t kMaxIoBytes = 8U << 20U;
constexpr std::size_t kBatchIos   = 32;
// Smallest table entries: a block (parent, hash, extra, word, tokens, host, Host hash, SSD slab, SSD hash)
// and a snapshot without a tail, SSD slabs, meta or Host slabs. They bound a damaged header's counts
// before allocating.
constexpr std::uint64_t kMinBlockBytes    = 4 + 8 + 8 + 4 + kBlockTokens * sizeof(TokenId) + 1 + 8 + 4 + 8;
constexpr std::uint64_t kMinSnapshotBytes = 4 + 4 + 4 + 1 + 4 + 1 + 1 + 4 + 1;

struct AlignedFree {
    void operator()(std::byte* p) const noexcept { ::operator delete(p, std::align_val_t{kBlock}); }
};
using AlignedBytes = std::unique_ptr<std::byte, AlignedFree>;

AlignedBytes aligned_bytes(std::size_t bytes) {
    return AlignedBytes(static_cast<std::byte*>(::operator new(bytes, std::align_val_t{kBlock})));
}

std::uint64_t round_up_block(std::uint64_t bytes) { return (bytes + kBlock - 1U) / kBlock * kBlock; }

// The head being built: values appended as their bytes, and 64-bit fields filled in later.
class Table {
public:
    template <class T>
    void value(const T& value) {
        bytes(&value, sizeof(T));
    }

    void bytes(const void* data, std::size_t count) {
        const auto* begin = static_cast<const std::byte*>(data);
        bytes_.insert(bytes_.end(), begin, begin + count);
    }

    void patch(std::size_t offset, std::uint64_t value) { std::memcpy(bytes_.data() + offset, &value, sizeof(value)); }

    [[nodiscard]] std::size_t size() const noexcept { return bytes_.size(); }
    [[nodiscard]] const std::byte* data() const noexcept { return bytes_.data(); }

private:
    std::vector<std::byte> bytes_;
};

// The head as it was read.
class Cursor {
public:
    Cursor(const std::byte* data, std::size_t size) : data_(data), size_(size) {}

    template <class T>
    [[nodiscard]] T value() {
        T out{};
        bytes(&out, sizeof(T));
        return out;
    }

    void bytes(void* data, std::size_t count) {
        if (count > size_ - at_) { throw std::runtime_error("prefix cache file is truncated"); }
        std::memcpy(data, data_ + at_, count);
        at_ += count;
    }

private:
    const std::byte* data_;
    std::size_t size_;
    std::size_t at_ = 0;
};

// POSIX: the rename itself is durable only once its directory is (Windows' rename journals it).
void make_rename_durable([[maybe_unused]] const std::filesystem::path& path) noexcept {
#ifndef _WIN32
    const std::filesystem::path directory = path.has_parent_path() ? path.parent_path() : std::filesystem::path(".");
    const int fd = ::open(directory.c_str(), O_RDONLY);
    if (fd >= 0) {
        (void)::fsync(fd);
        ::close(fd);
    }
#endif
}

// Issues one batch and waits for it; a failed transfer throws.
void run_batch(DirectScratchFile& file, DirectScratchFile::Batch&& batch, const char* what) {
    file.submit(std::move(batch));
    file.wait_idle();
    std::vector<DirectScratchFile::Completion> done;
    file.poll(done);
    for (const DirectScratchFile::Completion& completion : done) {
        if (!completion.ok) {
            throw std::runtime_error(std::string("prefix cache file ") + what + " failed: " + completion.error.message());
        }
    }
}

// `bytes` at `offset` between the file and `memory` (both 4 KiB-aligned), as transfers of up to 8 MiB.
void transfer_range(DirectScratchFile& file, DirectScratchFile::Op op, std::uint64_t offset, std::byte* memory,
                    std::size_t bytes, const char* what) {
    DirectScratchFile::Batch batch{.tag = 0, .op = op, .ios = {}};
    for (std::size_t at = 0; at < bytes; at += kMaxIoBytes) {
        batch.ios.push_back(DirectScratchFile::Io{.offset      = offset + at,
                                                  .bytes       = std::min(kMaxIoBytes, bytes - at),
                                                  .memory      = memory + at,
                                                  .hashes      = nullptr,
                                                  .hash_stride = 0});
    }
    run_batch(file, std::move(batch), what);
}

// One Host slab of the file: its position among the file's slabs and its address in the Host tier.
struct SlabIo {
    std::uint64_t file_slab = 0;
    std::byte* memory       = nullptr;
};

// Moves `slabs` (ascending file positions) between the file's payload at `payload` and their memory, and
// leaves slab k's content hash in hashes[k]. Slabs consecutive in the file and in memory (one chunk of the
// Host tier) join one transfer. `between(bytes)` runs after every batch with the bytes moved so far.
template <class Between>
void transfer_slabs(DirectScratchFile& file, DirectScratchFile::Op op, std::uint64_t payload, std::size_t slab_bytes,
                    std::span<const SlabIo> slabs, std::uint64_t* hashes, const char* what, const Between& between) {
    DirectScratchFile::Batch batch{.tag = 0, .op = op, .ios = {}};
    std::uint64_t moved = 0;
    for (std::size_t k = 0; k < slabs.size(); ++k) {
        const SlabIo& slab = slabs[k];
        if (!batch.ios.empty()) {
            DirectScratchFile::Io& last = batch.ios.back();
            if (last.offset + last.bytes == payload + slab.file_slab * slab_bytes && last.memory + last.bytes == slab.memory &&
                last.bytes + slab_bytes <= kMaxIoBytes) {
                last.bytes += slab_bytes;
                continue;
            }
            if (batch.ios.size() == kBatchIos) {
                for (const DirectScratchFile::Io& io : batch.ios) { moved += io.bytes; }
                run_batch(file, std::exchange(batch, DirectScratchFile::Batch{.tag = 0, .op = op, .ios = {}}), what);
                between(moved);
            }
        }
        batch.ios.push_back(DirectScratchFile::Io{.offset      = payload + slab.file_slab * slab_bytes,
                                                  .bytes       = slab_bytes,
                                                  .memory      = slab.memory,
                                                  .hashes      = hashes + k,
                                                  .hash_stride = slab_bytes});
    }
    if (!batch.ios.empty()) {
        for (const DirectScratchFile::Io& io : batch.ios) { moved += io.bytes; }
        run_batch(file, std::move(batch), what);
        between(moved);
    }
}

struct SavedBlock {
    std::int32_t parent = kRootIndex;
    std::uint64_t hash  = 0;
    std::uint64_t extra = 0;
    std::int32_t word   = -1;
    std::array<TokenId, kBlockTokens> tokens{};
    bool host               = false;
    std::uint64_t host_hash = 0;
    std::uint32_t ssd_slab  = kNone;
    std::uint64_t ssd_hash  = 0;
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
    std::vector<std::uint64_t> host_hashes; // one per Host slab the file holds of it
    std::uint32_t host_slabs      = 0;
    std::uint64_t host_slab_index = 0; // its first slab's position among the file's Host slabs
};

// A Host slab a load reads: where it goes, what it must hash to, and the restored entry it belongs to.
struct Fill {
    std::uint64_t expected = 0;
    std::uint32_t entry    = 0; // the saved block's or snapshot's index
    bool snapshot          = false;
};

double seconds_since(std::chrono::steady_clock::time_point started) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
}

std::uint32_t ssd_capacity_slabs(const PersistBinding& binding) {
    return binding.ssd == nullptr ? 0U : static_cast<std::uint32_t>(binding.ssd->capacity_bytes() / binding.slab_bytes);
}

std::filesystem::path directory_of(const std::filesystem::path& file) {
    return std::filesystem::absolute(file).parent_path();
}

} // namespace

std::filesystem::path kept_ssd_file(const std::filesystem::path& ssd_directory,
                                    const std::filesystem::path& prefix_cache_file) {
    const std::string name = std::filesystem::absolute(prefix_cache_file).generic_string();
    char hex[17]{};
    std::snprintf(hex, sizeof(hex), "%016llx",
                  static_cast<unsigned long long>(content_hash(reinterpret_cast<const std::byte*>(name.data()), name.size())));
    return ssd_directory / (kKeptPrefix + std::string(hex) + ".bin");
}

std::vector<KeptSsdFile> other_kept_ssd_files(const std::filesystem::path& ssd_directory,
                                              const std::filesystem::path& current) noexcept {
    std::vector<KeptSsdFile> out;
    try {
        std::error_code ec;
        const std::string own = current.filename().string();
        for (std::filesystem::directory_iterator entry(ssd_directory, ec), end; !ec && entry != end; entry.increment(ec)) {
            const std::string name = entry->path().filename().string();
            if (name.rfind(kKeptPrefix, 0) != 0 || entry->path().extension() != ".bin" || name == own) { continue; }
            std::error_code size_error;
            const std::uintmax_t bytes = entry->file_size(size_error);
            if (size_error) { continue; }
            out.push_back(KeptSsdFile{.path = entry->path(), .bytes = static_cast<std::uint64_t>(bytes)});
        }
    } catch (...) {} // a directory that cannot be listed names nothing
    return out;
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
        if (binding.slab_bytes == 0 || binding.slab_bytes % kBlock != 0) {
            throw std::logic_error("prefix cache file slabs must be whole 4 KiB blocks");
        }
        // The tables name SSD slabs only once their bytes are durable.
        if (ssd) { binding.ssd->flush(); }
        std::vector<NodeRef> nodes;
        std::vector<SnapshotRef> snapshots;
        index.collect_persistable(nodes, snapshots, ssd);

        // The head, with a zero where each Host slab's hash goes (hash_at, in file order) and where its own
        // size and the slab count go.
        Table table;
        std::vector<SlabIo> slabs;
        std::vector<std::size_t> hash_at;
        const auto host_slab = [&](std::uint32_t slab_id) {
            hash_at.push_back(table.size());
            table.value(std::uint64_t{0});
            slabs.push_back(SlabIo{.file_slab = slabs.size(), .memory = binding.slab(slab_id)});
        };
        table.bytes(binding.magic.data(), binding.magic.size());
        table.value(kVersion);
        table.value(std::uint64_t{0}); // the head's size (kHeadBytesOffset)
        table.value(static_cast<std::uint32_t>(fingerprint.size()));
        table.bytes(fingerprint.data(), fingerprint.size());
        table.value(static_cast<std::uint32_t>(binding.geometry.size()));
        table.bytes(binding.geometry.data(), binding.geometry.size());
        table.value(ssd ? kFlagSsd : 0U);
        table.value(ssd_capacity_slabs(binding));
        table.value(static_cast<std::uint32_t>(nodes.size()));
        table.value(static_cast<std::uint32_t>(snapshots.size()));
        const std::size_t slab_count_at = table.size();
        table.value(std::uint64_t{0}); // the file's Host slabs

        std::unordered_map<std::uint32_t, std::int32_t> file_index;
        file_index.reserve(nodes.size());
        for (const NodeRef node : nodes) {
            const BlockIdentity identity = index.block_identity(node);
            const NodeView view          = index.node(node);
            const bool host              = view.host == CopyState::Resident;
            const std::uint32_t ssd_slab = ssd && view.ssd == CopyState::Resident ? view.ssd_slab : kNone;
            table.value(identity.parent.valid() ? file_index.at(identity.parent.index) : kRootIndex);
            table.value(identity.lookup_hash);
            table.value(identity.extra);
            table.value(binding.block_word ? binding.block_word(node) : std::int32_t{-1});
            table.bytes(identity.tokens.data(), identity.tokens.size_bytes());
            table.value(static_cast<std::uint8_t>(host ? 1U : 0U));
            if (host) {
                host_slab(view.host_slab);
            } else {
                table.value(std::uint64_t{0});
            }
            table.value(ssd_slab);
            table.value(ssd_slab != kNone ? binding.ssd->slab_hash(ssd_slab) : std::uint64_t{0});
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
            table.value(view.anchor.valid() ? file_index.at(view.anchor.index) : kRootIndex);
            table.value(view.frontier);
            table.value(view.tail_len);
            table.bytes(view.tail.data(), view.tail.size_bytes());
            table.value(static_cast<std::uint8_t>(view.kind));
            table.value(view.hits);
            table.value(static_cast<std::uint8_t>(view.superseded ? 1U : 0U));
            table.value(static_cast<std::uint8_t>(host ? 1U : 0U));
            table.value(static_cast<std::uint32_t>(on_ssd ? view.ssd_slabs.size() : 0U));
            table.value(static_cast<std::uint8_t>(on_ssd && view.ssd_packed ? 1U : 0U));
            if (on_ssd) {
                for (const std::uint32_t slab : view.ssd_slabs) {
                    table.value(slab);
                    table.value(binding.ssd->slab_hash(slab));
                }
                ++out.ssd_snapshots;
            }
            if (binding.meta_bytes != 0) {
                binding.snapshot_meta(snapshot, meta.data());
                table.bytes(meta.data(), meta.size());
            }
            if (host) {
                for (const std::uint32_t slab_id : view.host_slabs) { host_slab(slab_id); }
            }
        }
        table.bytes(kFooter.data(), kFooter.size());
        table.value(static_cast<std::uint32_t>(nodes.size()));
        table.value(static_cast<std::uint32_t>(snapshots.size()));
        const std::uint64_t head_bytes = round_up_block(table.size());
        const std::uint64_t total      = head_bytes + static_cast<std::uint64_t>(slabs.size()) * binding.slab_bytes;
        table.patch(kHeadBytesOffset, head_bytes);
        table.patch(slab_count_at, slabs.size());

        // A temporary file an earlier save left (a process ended inside it) starts over.
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        {
            // Two writes in flight: the second's hashing runs under the first's write. More would only queue: a
            // new file's writes extend it in order (Windows runs a write past the valid data one at a time).
            DirectScratchFile file(directory_of(temporary), total,
                                   DirectScratchFile::Options{.read_depth = 1, .write_depth = 2, .retries = 2, .kept_file = temporary});
            std::vector<std::uint64_t> hashes(slabs.size(), 0U);
            transfer_slabs(file, DirectScratchFile::Op::Write, head_bytes, binding.slab_bytes, slabs, hashes.data(), "write",
                           [&](std::uint64_t) { check_abandoned(); });
            for (std::size_t k = 0; k < hashes.size(); ++k) { table.patch(hash_at[k], hashes[k]); }
            const AlignedBytes head = aligned_bytes(static_cast<std::size_t>(head_bytes));
            std::memcpy(head.get(), table.data(), table.size());
            std::memset(head.get() + table.size(), 0, static_cast<std::size_t>(head_bytes) - table.size());
            transfer_range(file, DirectScratchFile::Op::Write, 0, head.get(), static_cast<std::size_t>(head_bytes), "write");
            // Abandoned after the last slab: the previous file still stays.
            check_abandoned();
            // On the drive before the rename that replaces the previous file, so a power loss or an
            // operating-system crash leaves the old file or the new one, never a torn one.
            file.flush();
        }
        std::filesystem::rename(temporary, path);
        make_rename_durable(path);
        out.bytes     = total;
        out.blocks    = nodes.size();
        out.snapshots = snapshots.size();
    } catch (const std::exception& error) {
        // The file is closed by now, so the temporary file can be deleted on every platform.
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
    std::error_code missing;
    const std::uintmax_t size = std::filesystem::file_size(path, missing);
    if (missing) {
        out.message = "no saved prefix cache at " + path.string() + " yet; it is written at shutdown";
        return out;
    }
    const auto file_bytes = static_cast<std::uint64_t>(size);
    // Every file of this format is whole 4 KiB blocks (an earlier format's is not, save by chance: its
    // version then says so).
    if (file_bytes < kBlock || file_bytes % kBlock != 0) {
        out.message = "not a prefix cache file of this format version";
        return out;
    }
    // Opened once the header matches, so a missing or foreign file shows only the reason it was
    // not restored. A file damaged part way through completes it with the bytes that were read.
    std::optional<StartupPhaseScope> phase;
    std::uint64_t bytes_read = 0;
    try {
        if (binding.slab_bytes == 0 || binding.slab_bytes % kBlock != 0) {
            throw std::logic_error("prefix cache file slabs must be whole 4 KiB blocks");
        }
        DirectScratchFile file(directory_of(path), file_bytes,
                               DirectScratchFile::Options{
                                   .read_depth = 8, .write_depth = 1, .retries = 2, .kept_file = path, .read_only = true});
        const AlignedBytes first = aligned_bytes(kBlock);
        transfer_range(file, DirectScratchFile::Op::Read, 0, first.get(), kBlock, "read");
        Cursor prelude(first.get(), kBlock);
        std::array<char, 8> magic{};
        prelude.bytes(magic.data(), magic.size());
        if (magic != binding.magic || prelude.value<std::uint32_t>() != kVersion) {
            out.message = "not a prefix cache file of this format version";
            return out;
        }
        const auto head_bytes = prelude.value<std::uint64_t>();
        if (head_bytes < kBlock || head_bytes % kBlock != 0 || head_bytes > file_bytes) {
            out.message = "prefix cache file header is damaged";
            return out;
        }
        const AlignedBytes head = aligned_bytes(static_cast<std::size_t>(head_bytes));
        transfer_range(file, DirectScratchFile::Op::Read, 0, head.get(), static_cast<std::size_t>(head_bytes), "read");
        bytes_read = head_bytes;
        Cursor reader(head.get(), static_cast<std::size_t>(head_bytes));
        reader.bytes(magic.data(), magic.size());
        (void)reader.value<std::uint32_t>(); // the version
        (void)reader.value<std::uint64_t>(); // the head's size
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
        const auto flags               = reader.value<std::uint32_t>();
        const auto ssd_capacity        = reader.value<std::uint32_t>();
        const std::uint64_t slab_bytes = binding.slab_bytes;
        const auto node_count          = reader.value<std::uint32_t>();
        const auto snapshot_count      = reader.value<std::uint32_t>();
        const auto saved_slab_count    = reader.value<std::uint64_t>();
        if (static_cast<std::uint64_t>(node_count) * kMinBlockBytes +
                static_cast<std::uint64_t>(snapshot_count) * kMinSnapshotBytes > head_bytes) {
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
            block.host_hash = reader.value<std::uint64_t>();
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
                snapshot.host_hashes.resize(snapshot.host_slabs);
                reader.bytes(snapshot.host_hashes.data(), snapshot.host_hashes.size() * sizeof(std::uint64_t));
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
        std::array<char, 8> footer{};
        reader.bytes(footer.data(), footer.size());
        if (footer != kFooter || reader.value<std::uint32_t>() != node_count ||
            reader.value<std::uint32_t>() != snapshot_count) {
            throw std::runtime_error("prefix cache file tables are damaged");
        }
        if (host_slab_count != saved_slab_count || file_bytes != head_bytes + host_slab_count * slab_bytes) {
            throw std::runtime_error("prefix cache file is truncated or damaged");
        }
        if (ssd_saved == 0U) { out.ssd_message.clear(); }
        out.saved_blocks        = node_count;
        out.saved_snapshots     = snapshot_count;
        out.required_host_bytes = host_slab_count * slab_bytes;
        out.host_bytes          = static_cast<std::uint64_t>(binding.host_slabs) * slab_bytes;

        if (!use_ssd) {
            // Without the SSD copies a block the file holds no Host bytes of cuts off everything below it: those
            // blocks and the snapshots on them cannot come back, so the plan gives them no Host slab (it would
            // otherwise choose them ahead of snapshots that can).
            for (std::uint32_t i = 0; i < node_count; ++i) {
                block_host[i] = blocks[i].host &&
                                (blocks[i].parent == kRootIndex || block_host[static_cast<std::size_t>(blocks[i].parent)]);
            }
            for (std::uint32_t i = 0; i < snapshot_count; ++i) {
                if (snapshots[i].anchor != kRootIndex && !block_host[static_cast<std::size_t>(snapshots[i].anchor)]) {
                    shapes[i].host = false;
                }
            }
        }
        const HostRestorePlan plan = plan_host_restore(parents, shapes, binding.host_slabs, index.config(),
                                                       std::span<const bool>(block_host.get(), node_count));
        const std::uint64_t planned = head_bytes + plan.slabs * slab_bytes;
        phase.emplace(observer, StartupPhase::PrefixCacheLoad, StartupProgressUnit::Bytes, planned);
        phase->progress(bytes_read, planned);

        // The index entries first, each taking its Host slabs, then their bytes in file order.
        // Restored index ids of the file's blocks; absent when a block was neither chosen for the Host
        // tier nor kept on the SSD tier, or did not restore, so its descendants and the snapshots on it
        // are skipped too.
        std::vector<std::optional<NodeRef>> restored(node_count);
        std::vector<std::optional<SnapshotRef>> restored_snapshots(snapshot_count);
        std::vector<SlabIo> slabs;
        std::vector<Fill> fills;
        slabs.reserve(static_cast<std::size_t>(plan.slabs));
        fills.reserve(static_cast<std::size_t>(plan.slabs));
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
                slabs.push_back(SlabIo{.file_slab = saved_block.host_slab_index, .memory = binding.slab(block->slab)});
                fills.push_back(Fill{.expected = saved_block.host_hash, .entry = i, .snapshot = false});
            }
            if (saved_block.ssd_slab != kNone) { binding.ssd->expect(saved_block.ssd_slab, saved_block.ssd_hash); }
            if (binding.set_block_word) { binding.set_block_word(block->node, saved_block.word); }
            restored[i] = block->node;
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
                const std::span<const std::uint32_t> host_slabs = index.snapshot(*snapshot).host_slabs;
                if (host_slabs.size() != saved_snapshot.host_slabs) {
                    throw std::logic_error("a restored snapshot's Host slabs do not match its tail");
                }
                for (std::uint32_t k = 0; k < saved_snapshot.host_slabs; ++k) {
                    slabs.push_back(SlabIo{.file_slab = saved_snapshot.host_slab_index + k, .memory = binding.slab(host_slabs[k])});
                    fills.push_back(Fill{.expected = saved_snapshot.host_hashes[k], .entry = i, .snapshot = true});
                }
            }
            for (std::size_t k = 0; k < saved_snapshot.ssd_slabs.size(); ++k) {
                binding.ssd->expect(saved_snapshot.ssd_slabs[k], saved_snapshot.ssd_hashes[k]);
            }
            restored_snapshots[i] = *snapshot;
        }
        // The SSD free list is settled before anything is dropped below: a dropped entry frees its SSD slabs.
        if (use_ssd) { index.finish_ssd_restore(); }

        std::vector<std::uint64_t> hashes(slabs.size(), 0U);
        transfer_slabs(file, DirectScratchFile::Op::Read, head_bytes, static_cast<std::size_t>(slab_bytes), slabs, hashes.data(),
                       "read", [&](std::uint64_t bytes) {
                           bytes_read = head_bytes + bytes;
                           phase->progress(bytes_read, planned);
                       });

        // An entry with a slab whose bytes are not the saved ones goes: a block with everything that resumes
        // through it, a snapshot alone. The rest of the file is as good as it was.
        Withdrawal damaged;
        for (std::size_t k = 0; k < fills.size(); ++k) {
            if (hashes[k] == fills[k].expected) { continue; }
            if (fills[k].snapshot) {
                const SnapshotRef snapshot = *restored_snapshots[fills[k].entry];
                if (damaged.snapshots.empty() || damaged.snapshots.back() != snapshot) { damaged.snapshots.push_back(snapshot); }
            } else {
                damaged.nodes.push_back(*restored[fills[k].entry]);
            }
        }
        const std::size_t damaged_entries = damaged.nodes.size() + damaged.snapshots.size();
        if (damaged_entries != 0) {
            (void)index.withdraw(damaged);
            if (!damaged.empty()) { throw std::logic_error("a damaged prefix cache entry could not be dropped"); }
        }
        for (std::uint32_t i = 0; i < node_count; ++i) {
            if (!restored[i] || !index.valid(*restored[i])) { continue; }
            ++out.blocks;
            out.ssd_blocks += blocks[i].ssd_slab != kNone ? 1U : 0U;
        }
        for (std::uint32_t i = 0; i < snapshot_count; ++i) {
            if (!restored_snapshots[i] || !index.valid(*restored_snapshots[i])) { continue; }
            // The family's meta, checked against the image's header slab when its Host bytes were read.
            if (binding.restore_meta &&
                !binding.restore_meta(*restored_snapshots[i], snapshots[i].meta.data(), plan.snapshots[i])) {
                throw std::runtime_error("prefix cache file snapshot meta is damaged");
            }
            ++out.snapshots;
            out.ssd_snapshots += snapshots[i].ssd_slabs.empty() ? 0U : 1U;
        }
        if (damaged_entries != 0) {
            out.message = std::to_string(damaged_entries) +
                          (damaged_entries == 1 ? " entry's" : " entries'") +
                          " Host bytes did not match their saved hash; dropped with what resumed through them";
        }
        phase->complete(bytes_read);
    } catch (const std::exception& error) {
        if (phase) { phase->complete(bytes_read); }
        // Partially restored entries may hold unread slab bytes: the caller drops them all.
        out         = PersistResult{};
        out.message = error.what();
        return out;
    }
    out.ok      = true;
    out.bytes   = bytes_read;
    out.seconds = seconds_since(started);
    return out;
}

} // namespace infernix::runtime::prefix_cache
