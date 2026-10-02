#include "distribution.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace ninfer::perplexity {
namespace {

constexpr std::array<char, 8> kMagic{'N', 'I', 'N', 'F', 'T', 'O', 'P', 'K'};
constexpr std::uint32_t kVersion   = 1;
constexpr std::uint32_t kEndMarker = std::numeric_limits<std::uint32_t>::max();
// Bucket 0 holds contexts below 1024 tokens; bucket b>0 holds [1024*2^(b-1), 1024*2^b).
constexpr std::uint64_t kFirstBucketTokens = 1024;

template <class T>
void write_pod(std::ofstream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

template <class T>
void write_array(std::ofstream& out, const std::vector<T>& values) {
    out.write(reinterpret_cast<const char*>(values.data()),
              static_cast<std::streamsize>(values.size() * sizeof(T)));
}

void write_string(std::ofstream& out, const std::string& value) {
    write_pod(out, static_cast<std::uint32_t>(value.size()));
    out.write(value.data(), static_cast<std::streamsize>(value.size()));
}

template <class T>
T read_pod(std::ifstream& in, const std::filesystem::path& path) {
    T value{};
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!in) { throw std::runtime_error("truncated top-token reference: " + path.string()); }
    return value;
}

template <class T>
std::vector<T> read_array(std::ifstream& in, std::size_t count, const std::filesystem::path& path) {
    std::vector<T> values(count);
    in.read(reinterpret_cast<char*>(values.data()),
            static_cast<std::streamsize>(count * sizeof(T)));
    if (!in) { throw std::runtime_error("truncated top-token reference: " + path.string()); }
    return values;
}

std::string read_string(std::ifstream& in, const std::filesystem::path& path) {
    const auto size = read_pod<std::uint32_t>(in, path);
    if (size > 4096) { throw std::runtime_error("invalid top-token reference: " + path.string()); }
    std::string value(size, '\0');
    in.read(value.data(), size);
    if (!in) { throw std::runtime_error("truncated top-token reference: " + path.string()); }
    return value;
}

std::size_t bucket_of(std::uint64_t context_tokens) {
    std::size_t bucket = 0;
    for (std::uint64_t limit = kFirstBucketTokens; context_tokens >= limit; limit *= 2) {
        ++bucket;
    }
    return bucket;
}

std::string token_label(std::uint64_t tokens) {
    return tokens == 0 ? "0" : std::to_string(tokens / 1024) + "K";
}

std::pair<std::uint64_t, std::uint64_t> bucket_range(std::size_t bucket) {
    if (bucket == 0) { return {0, kFirstBucketTokens}; }
    return {kFirstBucketTokens << (bucket - 1), kFirstBucketTokens << bucket};
}

} // namespace

std::uint64_t token_fingerprint(std::span<const TokenId> tokens) noexcept {
    std::uint64_t hash = 14695981039346656037ULL;
    for (const TokenId token : tokens) {
        auto value = static_cast<std::uint32_t>(token);
        for (int byte = 0; byte < 4; ++byte) {
            hash ^= value & 0xffU;
            hash *= 1099511628211ULL;
            value >>= 8;
        }
    }
    return hash;
}

ReferenceWriter::ReferenceWriter(const std::filesystem::path& path, const ReferenceHeader& header)
    : path_(path), out_(path, std::ios::binary | std::ios::trunc), top_k_(header.top_k) {
    if (!out_) { throw std::runtime_error("cannot create top-token reference: " + path.string()); }
    out_.write(kMagic.data(), kMagic.size());
    write_pod(out_, kVersion);
    write_pod(out_, header.top_k);
    write_pod(out_, header.context);
    write_pod(out_, header.stride);
    write_string(out_, header.corpus_id);
    write_string(out_, header.mode);
    if (!out_) { throw std::runtime_error("cannot write top-token reference: " + path.string()); }
}

void ReferenceWriter::write(const ReferenceWindow& window) {
    const std::size_t positions = window.target_end - window.target_begin;
    if (window.target_logprobs.size() != positions || window.top_ids.size() != positions * top_k_ ||
        window.top_logprobs.size() != positions * top_k_) {
        throw std::logic_error("top-token reference window has inconsistent sizes");
    }
    write_pod(out_, window.stream_index);
    write_pod(out_, window.target_begin);
    write_pod(out_, window.target_end);
    write_pod(out_, window.input_hash);
    write_array(out_, window.target_logprobs);
    write_array(out_, window.top_ids);
    write_array(out_, window.top_logprobs);
    if (!out_) { throw std::runtime_error("cannot write top-token reference: " + path_.string()); }
}

void ReferenceWriter::finish() {
    write_pod(out_, kEndMarker);
    out_.flush();
    if (!out_) { throw std::runtime_error("cannot write top-token reference: " + path_.string()); }
    out_.close();
}

ReferenceReader::ReferenceReader(const std::filesystem::path& path)
    : path_(path), in_(path, std::ios::binary) {
    if (!in_) { throw std::runtime_error("cannot open top-token reference: " + path.string()); }
    std::array<char, 8> magic{};
    in_.read(magic.data(), magic.size());
    if (!in_ || magic != kMagic || read_pod<std::uint32_t>(in_, path) != kVersion) {
        throw std::runtime_error("not a version-1 top-token reference: " + path.string());
    }
    header_.top_k     = read_pod<std::uint32_t>(in_, path);
    header_.context   = read_pod<std::uint32_t>(in_, path);
    header_.stride    = read_pod<std::uint32_t>(in_, path);
    header_.corpus_id = read_string(in_, path);
    header_.mode      = read_string(in_, path);
    if (header_.top_k == 0 || header_.top_k > kMaximumScoreTopTokens) {
        throw std::runtime_error("invalid top-token reference: " + path.string());
    }
}

ReferenceWindow ReferenceReader::next() {
    ReferenceWindow window;
    window.stream_index = read_pod<std::uint32_t>(in_, path_);
    if (window.stream_index == kEndMarker) {
        throw std::runtime_error("top-token reference has fewer windows than this run: " +
                                 path_.string());
    }
    window.target_begin = read_pod<std::uint64_t>(in_, path_);
    window.target_end   = read_pod<std::uint64_t>(in_, path_);
    window.input_hash   = read_pod<std::uint64_t>(in_, path_);
    if (window.target_end <= window.target_begin ||
        window.target_end - window.target_begin > (std::uint64_t{1} << 24)) {
        throw std::runtime_error("invalid top-token reference window: " + path_.string());
    }
    const std::size_t positions = window.target_end - window.target_begin;
    window.target_logprobs      = read_array<float>(in_, positions, path_);
    window.top_ids              = read_array<TokenId>(in_, positions * header_.top_k, path_);
    window.top_logprobs         = read_array<float>(in_, positions * header_.top_k, path_);
    return window;
}

void ReferenceReader::finish() {
    if (read_pod<std::uint32_t>(in_, path_) != kEndMarker) {
        throw std::runtime_error("top-token reference has more windows than this run: " +
                                 path_.string());
    }
}

PositionDivergence position_divergence(std::span<const float> reference_logprobs,
                                       std::span<const float> evaluated_logprobs, bool top1_match,
                                       float reference_target, float evaluated_target) {
    if (reference_logprobs.empty() || reference_logprobs.size() != evaluated_logprobs.size()) {
        throw std::logic_error("divergence needs one evaluated log-probability per top token");
    }
    double kl             = 0.0;
    double reference_mass = 0.0;
    double evaluated_mass = 0.0;
    for (std::size_t index = 0; index < reference_logprobs.size(); ++index) {
        const double reference = reference_logprobs[index];
        const double evaluated = evaluated_logprobs[index];
        if (!std::isfinite(reference) || !std::isfinite(evaluated)) {
            throw std::runtime_error("divergence received a non-finite log-probability");
        }
        const double probability = std::exp(reference);
        kl += probability * (reference - evaluated);
        reference_mass += probability;
        evaluated_mass += std::exp(evaluated);
    }
    // The remaining tokens form one bucket. FP32 log-probabilities can put either top mass a
    // rounding step above one; the reference's remainder then carries no weight, and the
    // evaluated one is kept positive.
    const double reference_rest = 1.0 - reference_mass;
    if (reference_rest > 0.0) {
        const double evaluated_rest = std::max(1.0 - evaluated_mass, 1.0e-12);
        kl += reference_rest * (std::log(reference_rest) - std::log(evaluated_rest));
    }
    return {.kl                 = kl,
            .reference_top_mass = std::min(reference_mass, 1.0),
            .top1_match         = top1_match,
            .delta_nll = static_cast<double>(reference_target) - static_cast<double>(evaluated_target)};
}

void DivergenceAggregate::Sums::add(const PositionDivergence& value) {
    ++count;
    top1_matches += value.top1_match ? 1U : 0U;
    kl += value.kl;
    kl_squares += value.kl * value.kl;
    delta_nll += value.delta_nll;
    reference_top_mass += value.reference_top_mass;
}

void DivergenceAggregate::Sums::add(const Sums& other) {
    count += other.count;
    top1_matches += other.top1_matches;
    kl += other.kl;
    kl_squares += other.kl_squares;
    delta_nll += other.delta_nll;
    reference_top_mass += other.reference_top_mass;
}

nlohmann::json DivergenceAggregate::Sums::to_json() const {
    const double n = static_cast<double>(count);
    return nlohmann::json{
        {"positions", count},
        {"mean_kl", count ? kl / n : 0.0},
        {"kl_standard_deviation",
         count ? std::sqrt(std::max(kl_squares / n - (kl / n) * (kl / n), 0.0)) : 0.0},
        {"top1_agreement", count ? static_cast<double>(top1_matches) / n : 0.0},
        {"mean_delta_nll", count ? delta_nll / n : 0.0},
        {"mean_reference_top_mass", count ? reference_top_mass / n : 0.0}};
}

void DivergenceAggregate::add(std::uint64_t context_tokens, const PositionDivergence& value) {
    const std::size_t bucket = bucket_of(context_tokens);
    if (buckets_.size() <= bucket) { buckets_.resize(bucket + 1); }
    buckets_[bucket].add(value);
    overall_.add(value);
    values_.push_back(value.kl);
}

void DivergenceAggregate::add(const DivergenceAggregate& other) {
    if (buckets_.size() < other.buckets_.size()) { buckets_.resize(other.buckets_.size()); }
    for (std::size_t bucket = 0; bucket < other.buckets_.size(); ++bucket) {
        buckets_[bucket].add(other.buckets_[bucket]);
    }
    overall_.add(other.overall_);
    values_.insert(values_.end(), other.values_.begin(), other.values_.end());
}

double DivergenceAggregate::mean_kl() const {
    if (overall_.count == 0) { throw std::logic_error("divergence aggregate is empty"); }
    return overall_.kl / static_cast<double>(overall_.count);
}

double DivergenceAggregate::top1_agreement() const {
    if (overall_.count == 0) { throw std::logic_error("divergence aggregate is empty"); }
    return static_cast<double>(overall_.top1_matches) / static_cast<double>(overall_.count);
}

nlohmann::json DivergenceAggregate::to_json() const {
    nlohmann::json out = overall_.to_json();
    if (!values_.empty()) {
        std::vector<double> sorted = values_;
        std::sort(sorted.begin(), sorted.end());
        const auto quantile = [&](double q) {
            const auto index = static_cast<std::size_t>(q * static_cast<double>(sorted.size() - 1));
            return sorted[index];
        };
        out["median_kl"] = quantile(0.5);
        out["p90_kl"]    = quantile(0.9);
        out["p99_kl"]    = quantile(0.99);
        out["p999_kl"]   = quantile(0.999);
        out["max_kl"]    = sorted.back();
    }
    nlohmann::json buckets = nlohmann::json::array();
    for (std::size_t bucket = 0; bucket < buckets_.size(); ++bucket) {
        if (buckets_[bucket].count == 0) { continue; }
        const auto [begin, end] = bucket_range(bucket);
        nlohmann::json item     = buckets_[bucket].to_json();
        item["context_begin"]   = begin;
        item["context_end"]     = end;
        buckets.push_back(std::move(item));
    }
    out["by_context"] = std::move(buckets);
    return out;
}

std::string DivergenceAggregate::bucket_summary() const {
    std::ostringstream out;
    out.setf(std::ios::scientific);
    out.precision(2);
    bool first = true;
    for (std::size_t bucket = 0; bucket < buckets_.size(); ++bucket) {
        const Sums& sums = buckets_[bucket];
        if (sums.count == 0) { continue; }
        const auto [begin, end] = bucket_range(bucket);
        out << (first ? "" : " ") << token_label(begin) << '-' << token_label(end) << ' '
            << sums.kl / static_cast<double>(sums.count);
        first = false;
    }
    return out.str();
}

} // namespace ninfer::perplexity
