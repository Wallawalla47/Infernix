#include "core/read_only_file.h"

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <exception>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <system_error>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#if defined(__linux__)
#include <linux/aio_abi.h>
#include <sys/syscall.h>
#endif

namespace infernix {

struct ReadOnlyFile::Impl {
    int fd                = -1;
    const std::byte* data = nullptr;
    std::size_t size      = 0;

#if defined(__linux__)
    // Kernel AIO contexts of finished read_direct_blocks calls, reused by later calls: creating and
    // destroying one per call (io_setup, io_destroy) measured ~36 ms on Linux 6.18 (WSL2) against
    // ~0.12 ms for a 4 KiB read, and an n-gram decode round makes a few calls behind the GPU. One
    // context per concurrent caller (a large n-gram call reads on eight threads).
    struct AioContext {
        aio_context_t id = 0;
        unsigned slots   = 0; // reads it can hold in flight; 0: none (io_setup failed)
    };
    static constexpr unsigned kMinAioSlots = 64;
    std::mutex aio_mutex;
    std::vector<AioContext> aio_idle;

    // An idle context for `slots` reads in flight, or a new one; none when io_setup fails.
    AioContext take_aio(std::size_t slots) {
        {
            const std::lock_guard<std::mutex> lock(aio_mutex);
            for (auto it = aio_idle.begin(); it != aio_idle.end(); ++it) {
                if (it->slots >= slots) {
                    const AioContext context = *it;
                    aio_idle.erase(it);
                    return context;
                }
            }
        }
        AioContext context;
        const unsigned want = static_cast<unsigned>(std::max<std::size_t>(slots, kMinAioSlots));
        if (::syscall(SYS_io_setup, want, &context.id) == 0) { context.slots = want; }
        return context;
    }

    // A context with no read in flight goes back to the pool; one that may still have reads in
    // flight is destroyed (io_destroy waits for them).
    void put_aio(AioContext context, bool idle) noexcept {
        if (context.slots == 0) { return; }
        if (idle) {
            try {
                const std::lock_guard<std::mutex> lock(aio_mutex);
                aio_idle.push_back(context);
                return;
            } catch (...) {}
        }
        ::syscall(SYS_io_destroy, context.id);
    }
#endif

    explicit Impl(const std::filesystem::path& path, FileMapping map) {
        fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
        if (fd < 0) {
            throw std::system_error(errno, std::generic_category(), "open " + path.string());
        }

        struct stat status{};
        if (::fstat(fd, &status) != 0) {
            const int error = errno;
            ::close(fd);
            fd = -1;
            throw std::system_error(error, std::generic_category(), "fstat " + path.string());
        }
        if (status.st_size < 0 ||
            static_cast<std::uintmax_t>(status.st_size) > std::numeric_limits<std::size_t>::max()) {
            ::close(fd);
            fd = -1;
            throw std::overflow_error("file size does not fit the process address space");
        }

        size = static_cast<std::size_t>(status.st_size);
        if (map == FileMapping::Whole && size != 0) {
            void* mapping = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
            if (mapping == MAP_FAILED) {
                const int error = errno;
                ::close(fd);
                fd = -1;
                throw std::system_error(error, std::generic_category(), "mmap " + path.string());
            }
            data = static_cast<const std::byte*>(mapping);
        }
    }

    ~Impl() {
#if defined(__linux__)
        for (const AioContext& context : aio_idle) { ::syscall(SYS_io_destroy, context.id); }
#endif
        if (data != nullptr) { ::munmap(const_cast<std::byte*>(data), size); }
        if (fd >= 0) { ::close(fd); }
    }
};

ReadOnlyFile::ReadOnlyFile(const std::filesystem::path& path, FileMapping mapping)
    : impl_(std::make_unique<Impl>(path, mapping)) {}

ReadOnlyFile::~ReadOnlyFile()                                  = default;
ReadOnlyFile::ReadOnlyFile(ReadOnlyFile&&) noexcept            = default;
ReadOnlyFile& ReadOnlyFile::operator=(ReadOnlyFile&&) noexcept = default;

std::span<const std::byte> ReadOnlyFile::mapped_bytes() const noexcept {
    return {impl_->data, impl_->data != nullptr ? impl_->size : 0}; // FileMapping::None: empty
}

std::uint64_t ReadOnlyFile::current_bytes() const noexcept {
    struct stat status{};
    if (::fstat(impl_->fd, &status) != 0 || status.st_size < 0) {
        return impl_->size;
    }
    return static_cast<std::uint64_t>(status.st_size);
}

void ReadOnlyFile::read_direct_blocks(std::span<const std::uint64_t> offsets, std::size_t block_bytes,
                                      std::span<std::byte> ring,
                                      const std::function<void(std::size_t, std::span<const std::byte>)>& consume) const {
    if (block_bytes == 0 || ring.size() < block_bytes) {
        throw std::invalid_argument("direct block reads need a ring of at least one block");
    }
    if (offsets.empty()) { return; }
#if defined(__linux__)
    // Kernel AIO on the O_DIRECT descriptor keeps one read per ring block in flight, as overlapped
    // reads do on Windows (an NVMe needs the queue depth: one thread's serial preads reach ~1/16 of
    // the device's random-read rate).
    const std::size_t slots = std::min(ring.size() / block_bytes, offsets.size());
    // The call's context returns to the file's pool once no read is in flight (destroyed otherwise).
    struct Lease {
        Impl& impl;
        Impl::AioContext context;
        bool idle = false;
        ~Lease() { impl.put_aio(context, idle); }
    } lease{*impl_, impl_->take_aio(slots)};
    if (lease.context.slots != 0) {
        const aio_context_t context = lease.context.id;
        std::vector<iocb> blocks(slots);
        std::vector<io_event> events(slots);
        std::vector<std::size_t> idle;
        idle.reserve(slots);
        for (std::size_t s = slots; s-- > 0;) { idle.push_back(s); }
        std::size_t next = 0, in_flight = 0;
        int failure      = 0;
        bool short_read  = false;
        std::exception_ptr consume_error;
        const auto failed = [&] { return failure != 0 || short_read || consume_error; };
        while (in_flight > 0 || (next < offsets.size() && !failed())) {
            while (next < offsets.size() && !idle.empty() && !failed()) {
                const std::size_t slot = idle.back();
                if (offsets[next] > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max())) {
                    failure = EOVERFLOW;
                    break;
                }
                iocb& block          = blocks[slot];
                block                = iocb{};
                block.aio_fildes     = static_cast<std::uint32_t>(impl_->fd);
                block.aio_lio_opcode = IOCB_CMD_PREAD;
                block.aio_buf        = reinterpret_cast<std::uint64_t>(ring.data() + slot * block_bytes);
                block.aio_nbytes     = block_bytes;
                block.aio_offset     = static_cast<std::int64_t>(offsets[next]);
                block.aio_data       = (static_cast<std::uint64_t>(next) << 16U) | slot;
                iocb* submit         = &block;
                const long submitted = ::syscall(SYS_io_submit, context, 1L, &submit);
                if (submitted != 1) {
                    if (submitted < 0 && errno == EAGAIN && in_flight > 0) { break; } // reap first
                    if (submitted < 0 && errno == EINTR) { continue; }
                    failure = submitted < 0 ? errno : EIO;
                    break;
                }
                idle.pop_back();
                ++next;
                ++in_flight;
            }
            if (in_flight == 0) { continue; }
            const long done = ::syscall(SYS_io_getevents, context, 1L, static_cast<long>(slots), events.data(), nullptr);
            if (done < 0) {
                if (errno == EINTR) { continue; }
                // The kernel still owns the buffers of reads in flight: they must land before the
                // ring is reused, so wait them out serially.
                const int error = errno;
                while (in_flight > 0) {
                    const long one = ::syscall(SYS_io_getevents, context, 1L, 1L, events.data(), nullptr);
                    if (one > 0) { in_flight -= static_cast<std::size_t>(one); }
                    else if (one < 0 && errno != EINTR) { break; }
                }
                lease.idle = in_flight == 0;
                throw std::system_error(error, std::generic_category(), "direct block read: io_getevents");
            }
            for (long e = 0; e < done; ++e) {
                const io_event& event   = events[static_cast<std::size_t>(e)];
                const std::size_t slot  = static_cast<std::size_t>(event.data & 0xFFFFU);
                const std::size_t index = static_cast<std::size_t>(event.data >> 16U);
                if (event.res < 0) {
                    if (failure == 0) { failure = static_cast<int>(-event.res); }
                } else if (static_cast<std::uint64_t>(event.res) != block_bytes) {
                    short_read = true;
                } else if (!failed()) {
                    try {
                        consume(index, std::span<const std::byte>(ring.data() + slot * block_bytes, block_bytes));
                    } catch (...) { consume_error = std::current_exception(); }
                }
                idle.push_back(slot);
                --in_flight;
            }
        }
        lease.idle = true; // every submitted read has been reaped
        if (consume_error) { std::rethrow_exception(consume_error); }
        if (failure != 0) { throw std::system_error(failure, std::generic_category(), "direct block read"); }
        if (short_read) { throw std::runtime_error("direct block read: a read did not complete in full"); }
        return;
    }
#endif
    // Without kernel AIO: serial reads into the first ring block.
    const std::span<std::byte> block = ring.first(block_bytes);
    for (std::size_t i = 0; i < offsets.size(); ++i) {
        if (read_direct(offsets[i], block) != block_bytes) {
            throw std::runtime_error("direct block read: a read did not complete in full");
        }
        consume(i, block);
    }
}

std::size_t ReadOnlyFile::read_direct(std::uint64_t offset,
                                      std::span<std::byte> destination) const {
    if (offset > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max()) ||
        destination.size() > static_cast<std::size_t>(std::numeric_limits<ssize_t>::max())) {
        throw std::overflow_error("direct file read exceeds platform I/O limits");
    }

    // One pread may return less than requested before the end of the file (FUSE and network
    // filesystems cap a read), so read until the range is full or the file ends, as the Windows
    // path does (upstream Neroued/ninfer#381).
    std::size_t total = 0;
    while (total < destination.size()) {
        const ssize_t bytes = ::pread(impl_->fd, destination.data() + total,
                                      destination.size() - total,
                                      static_cast<off_t>(offset + total));
        if (bytes < 0) {
            if (errno == EINTR) { continue; }
            throw std::system_error(errno, std::generic_category(), "direct file read");
        }
        if (bytes == 0) { break; }
        total += static_cast<std::size_t>(bytes);
    }
    return total;
}

} // namespace infernix
