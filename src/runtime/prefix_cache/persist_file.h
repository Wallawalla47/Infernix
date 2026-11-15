#pragma once

#include "core/startup.h"
#include "infernix/types.h"
#include "runtime/prefix_cache/persist_result.h"
#include "runtime/prefix_cache/prefix_index.h"
#include "runtime/prefix_cache/ssd_tier.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace infernix::runtime::prefix_cache {

// What a model's prefix cache binding gives the file (§5.5, §17): one layout for every family. The
// family supplies its file kind, its Host geometry (compared byte for byte), the slab addresses, and
// what it keeps beside the index: one word per block (Qwen3.8-Flash-Next's MTP continuation token) and
// fixed-size meta per snapshot (its state image header, kept beside an SSD copy).
struct PersistBinding {
    std::array<char, 8> magic{};
    std::vector<std::byte> geometry;
    std::size_t slab_bytes    = 0;
    std::uint32_t image_slabs = 0; // Host slabs of one state image
    std::uint32_t host_slabs  = 0; // the Host tier's
    std::function<std::byte*(std::uint32_t)> slab;
    std::function<std::int32_t(NodeRef)> block_word;          // empty: none (-1 saved)
    std::function<void(NodeRef, std::int32_t)> set_block_word; // empty: ignored
    std::size_t meta_bytes = 0;
    std::function<void(SnapshotRef, std::byte*)> snapshot_meta;
    // A restored snapshot's saved meta; `host_read`: its Host image bytes were read too. False when the
    // meta does not belong to the image (a damaged file).
    std::function<bool(SnapshotRef, const std::byte*, bool host_read)> restore_meta;
    // A kept SSD tier (SsdTier::kept): the copies it holds are saved by reference, with their hashes,
    // and taken back at a load when the tier reopened the previous run's file.
    SsdTier* ssd = nullptr;
};

// Every Host-backed (and with a kept SSD tier, SSD-backed) snapshot and the block path it anchors on,
// written to a temporary file renamed into place: the header, the block table (parents first), the
// snapshot table (retained snapshots first, then superseded ones oldest first) and the end mark, padded
// to 4 KiB, then the Host slab bytes of the entries that have them. The slabs go from the Host tier's
// memory (binding.slab: 4 KiB-aligned) to the drive unbuffered, consecutive ones in one chunk as one
// write, each hashed on its way (the tables carry the hashes and are written last). The SSD tier must be
// drained (no transfer in flight); its file is made durable before the tables name its slabs. An
// abandoned save keeps the previous file.
[[nodiscard]] PersistResult save_prefix_cache(const PrefixCacheIndex& index, const PersistBinding& binding,
                                              const std::filesystem::path& path, std::string_view fingerprint,
                                              const CancellationView& abandoned);

// Rebuilds a saved cache into the empty `index`: when the fingerprint and the geometry match, the Host
// tier gets what plan_host_restore chooses (its slab bytes read), and the entries with SSD copies in a
// reopened kept SSD tier come back SSD-resident (each slab checked against its hash at its first
// promotion). The slabs are read unbuffered straight into the Host tier's memory, several reads in
// flight, and each is checked against its saved hash: an entry with a slab whose bytes changed is
// dropped (a block with what resumes through it) and the rest stays; the result is then ok with a
// message that counts them. Damaged tables or a failed read throw nothing: the result says why, and the
// caller clears the index.
[[nodiscard]] PersistResult load_prefix_cache(PrefixCacheIndex& index, const PersistBinding& binding,
                                              const std::filesystem::path& path, std::string_view fingerprint,
                                              const StartupObserver& observer);

// The kept SSD tier files in `ssd_directory` other than `current` (this start's, from kept_ssd_file),
// with their sizes: a prefix cache file that is no longer used leaves its own behind, which nothing
// removes. For the startup log; another running server's file is listed too, so none is deleted here.
struct KeptSsdFile {
    std::filesystem::path path;
    std::uint64_t bytes = 0;
};
[[nodiscard]] std::vector<KeptSsdFile> other_kept_ssd_files(const std::filesystem::path& ssd_directory,
                                                            const std::filesystem::path& current) noexcept;

// The kept SSD tier's file for a prefix cache file: in the SSD tier's directory, named after the
// prefix cache file's absolute path, so each prefix cache file keeps its own.
[[nodiscard]] std::filesystem::path kept_ssd_file(const std::filesystem::path& ssd_directory,
                                                  const std::filesystem::path& prefix_cache_file);

} // namespace infernix::runtime::prefix_cache
