#pragma once

#include "string.h"
#include "utils.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>

namespace jinja {

struct value_t;
using value = std::shared_ptr<value_t>;

// Helper to check the type of a value
template <typename T>
struct extract_pointee {
    using type = T;
};

template <typename U>
struct extract_pointee<std::shared_ptr<U>> {
    using type = U;
};

// Every value type has one kind bit; a value's kinds() holds its type's bit and its base types'
// bits, so is_val<T> answers what dynamic_cast to T would, without the RTTI walk.
enum value_kind : std::uint32_t {
    kind_int       = 1U << 0U,
    kind_float     = 1U << 1U,
    kind_string    = 1U << 2U,
    kind_bool      = 1U << 3U,
    kind_array     = 1U << 4U,
    kind_tuple     = 1U << 5U,
    kind_object    = 1U << 6U,
    kind_none      = 1U << 7U,
    kind_undefined = 1U << 8U,
    kind_func      = 1U << 9U,
    kind_kwarg     = 1U << 10U,
};

std::uint32_t kinds_of(const value_t* ptr) noexcept; // defined after value_t

template <typename T>
bool is_val(const value_t* ptr) {
    using PointeeType = typename extract_pointee<T>::type;
    return ptr != nullptr && (kinds_of(ptr) & PointeeType::kind) != 0;
}

template <typename T>
bool is_val(const value& ptr) {
    return is_val<T>(ptr.get());
}

template <typename T, typename... Args>
std::shared_ptr<typename extract_pointee<T>::type> mk_val(Args&&... args) {
    using PointeeType = typename extract_pointee<T>::type;
    return std::make_shared<PointeeType>(std::forward<Args>(args)...);
}

template <typename T>
const typename extract_pointee<T>::type* cast_val(const value& ptr) {
    using PointeeType = typename extract_pointee<T>::type;
    return is_val<T>(ptr) ? static_cast<const PointeeType*>(ptr.get()) : nullptr;
}

template <typename T>
typename extract_pointee<T>::type* cast_val(value& ptr) {
    using PointeeType = typename extract_pointee<T>::type;
    return is_val<T>(ptr) ? static_cast<PointeeType*>(ptr.get()) : nullptr;
}
// End Helper


struct context; // forward declaration


//
// base value type
//

struct func_args; // function argument values

using func_hptr     = value(const func_args&);
using func_handler  = std::function<func_hptr>;
using func_builtins = std::map<std::string, func_handler>;

enum value_compare_op { eq, ge, gt, lt, ne };

bool value_compare(const value& a, const value& b, value_compare_op op);

struct value_t {
    int64_t val_int = 0;
    double val_flt  = 0;
    string val_str;

    std::vector<value> val_arr;
    std::vector<std::pair<value, value>> val_obj;

    func_handler val_func;

    uint32_t origin = 0; // optional whole-value output origin

    value_t()               = default;
    value_t(const value_t&) = default;
    virtual ~value_t()      = default;

    // The value_kind bits of the value's type and of its base types.
    virtual std::uint32_t kinds() const noexcept { return 0; }

    // Note: only for debugging and error reporting purposes
    virtual std::string type() const { return ""; }

    virtual int64_t as_int() const { throw_type_error("is not an int value"); }

    virtual double as_float() const { throw_type_error("is not a float value"); }

    virtual string as_string() const { throw_type_error("is not a string value"); }

    virtual bool as_bool() const { throw_type_error("is not a bool value"); }

    virtual const std::vector<value>& as_array() const {
        throw_type_error("is not an array value");
    }

    virtual const std::vector<std::pair<value, value>>& as_ordered_object() const {
        throw_type_error("is not an object value");
    }

    virtual value invoke(const func_args&) const { throw_type_error("is not a function value"); }

    virtual bool is_none() const { return false; }

    virtual bool is_undefined() const { return false; }

    virtual const func_builtins& get_builtins() const { throw_type_error("has no builtins"); }

    virtual bool has_key(const value&) { throw_type_error("is not an object value"); }

    virtual void insert(const value& /* key */, const value& /* val */) {
        throw_type_error("is not an object value");
    }

    virtual value& at(const value& /* key */, value& /* default_val */) {
        throw_type_error("is not an object value");
    }

    virtual value& at(const value& /* key */) { throw_type_error("is not an object value"); }

    virtual value& at(const std::string& /* key */, value& /* default_val */) {
        throw_type_error("is not an object value");
    }

    virtual value& at(const std::string& /* key */) { throw_type_error("is not an object value"); }

    virtual value& at(int64_t /* idx */, value& /* default_val */) {
        throw_type_error("is not an array value");
    }

    virtual value& at(int64_t /* idx */) { throw_type_error("is not an array value"); }

    virtual bool is_numeric() const { return false; }

    virtual bool is_hashable() const { return false; }

    virtual bool is_immutable() const { return true; }

    virtual hasher unique_hash() const noexcept = 0;

    // TODO: C++20 <=> operator
    // NOTE: We are treating == as equivalent (for normal comparisons) and != as strict nonequal
    // (for strict (is) comparisons)
    virtual bool operator==(const value_t& other) const { return equivalent(other); }

    virtual bool operator!=(const value_t& other) const { return nonequal(other); }

    // Note: only for debugging purposes
    virtual std::string as_repr() const { return as_string().str(); }

private:
    [[noreturn]] void throw_type_error(const char* expected) const {
        throw std::runtime_error(type() + " " + expected);
    }

protected:
    virtual bool equivalent(const value_t&) const = 0;

    virtual bool nonequal(const value_t& other) const { return !equivalent(other); }
};

//
// utils
//

inline std::uint32_t kinds_of(const value_t* ptr) noexcept { return ptr->kinds(); }

const func_builtins& global_builtins();

std::string value_to_json(const value& val, int indent = -1, const std::string_view item_sep = ", ",
                          const std::string_view key_sep = ": ", bool ensure_ascii = false,
                          bool sort_keys = false);

// Note: only used for debugging purposes
std::string value_to_string_repr(const value& val);

struct not_implemented_exception : public std::runtime_error {
    not_implemented_exception(const std::string& msg)
        : std::runtime_error("NotImplemented: " + msg) {}
};

struct value_hasher {
    size_t operator()(const value& val) const noexcept { return val->unique_hash().digest(); }
};

struct value_equivalence {
    bool operator()(const value& lhs, const value& rhs) const { return *lhs == *rhs; }

    bool operator()(const std::pair<value, value>& lhs, const std::pair<value, value>& rhs) const {
        return *(lhs.first) == *(rhs.first) && *(lhs.second) == *(rhs.second);
    }
};

struct value_equality {
    bool operator()(const value& lhs, const value& rhs) const { return !(*lhs != *rhs); }
};

//
// primitive value types
//

struct value_int_t : public value_t {
    static constexpr std::uint32_t kind = kind_int;

    value_int_t(int64_t v) {
        val_int = v;
        val_flt = float_of(v);
    }

    static double float_of(int64_t v) noexcept {
        const double result = static_cast<double>(v);
        return static_cast<int64_t>(result) != v ? (v < 0 ? -INFINITY : INFINITY) : result;
    }

    // The hash of the integer v. A float or boolean equal to an integer hashes as that integer.
    static hasher hash_of(int64_t v) noexcept {
        static const std::size_t type_hash = typeid(value_int_t).hash_code();
        const double flt                   = float_of(v);
        hasher hash;
        hash.update(&type_hash, sizeof(type_hash)).update(&v, sizeof(v)).update(&flt, sizeof(flt));
        return hash;
    }

    std::uint32_t kinds() const noexcept override { return kind; }

    virtual std::string type() const override { return "Integer"; }

    virtual int64_t as_int() const override { return val_int; }

    virtual double as_float() const override { return val_flt; }

    virtual string as_string() const override { return std::to_string(val_int); }

    virtual bool as_bool() const override { return val_int != 0; }

    virtual const func_builtins& get_builtins() const override;

    virtual bool is_numeric() const override { return true; }

    virtual bool is_hashable() const override { return true; }

    virtual hasher unique_hash() const noexcept override {
        static const std::size_t type_hash = typeid(value_int_t).hash_code();
        hasher hash;
        hash.update(&type_hash, sizeof(type_hash))
            .update(&val_int, sizeof(val_int))
            .update(&val_flt, sizeof(val_flt));
        return hash;
    }
protected:
    virtual bool equivalent(const value_t& other) const override {
        return other.is_numeric() && val_int == other.val_int && val_flt == other.val_flt;
    }

    virtual bool nonequal(const value_t& other) const override {
        return !(typeid(*this) == typeid(other) && val_int == other.val_int);
    }
};

using value_int = std::shared_ptr<value_int_t>;

struct value_float_t : public value_t {
    static constexpr std::uint32_t kind = kind_float;

    value_float_t(double v) {
        val_flt = v;
        val_int = std::isfinite(v) ? static_cast<int64_t>(v) : 0;
    }

    std::uint32_t kinds() const noexcept override { return kind; }

    virtual std::string type() const override { return "Float"; }

    virtual double as_float() const override { return val_flt; }

    virtual int64_t as_int() const override { return val_int; }

    string as_string() const override;

    virtual bool as_bool() const override { return val_flt != 0.0; }

    virtual const func_builtins& get_builtins() const override;

    virtual bool is_numeric() const override { return true; }

    virtual bool is_hashable() const override { return true; }

    virtual hasher unique_hash() const noexcept override {
        if (static_cast<double>(val_int) == val_flt) {
            return value_int_t::hash_of(val_int);
        } else {
            static const std::size_t type_hash = typeid(value_float_t).hash_code();
            hasher hash;
            hash.update(&type_hash, sizeof(type_hash))
                .update(&val_int, sizeof(val_int))
                .update(&val_flt, sizeof(val_flt));
            return hash;
        }
    }
protected:
    virtual bool equivalent(const value_t& other) const override {
        return other.is_numeric() && val_int == other.val_int && val_flt == other.val_flt;
    }

    virtual bool nonequal(const value_t& other) const override {
        return !(typeid(*this) == typeid(other) && val_flt == other.val_flt);
    }
};

using value_float = std::shared_ptr<value_float_t>;

struct value_string_t : public value_t {
    static constexpr std::uint32_t kind = kind_string;

    std::uint32_t kinds() const noexcept override { return kind; }

    value_string_t() { val_str = string(); }

    value_string_t(const std::string& v) { val_str = string(v); }

    value_string_t(const string& v) { val_str = v; }

    value_string_t(string&& v) { val_str = std::move(v); }

    virtual std::string type() const override { return "String"; }

    virtual string as_string() const override { return val_str; }

    virtual bool as_bool() const override { return val_str.length() > 0; }

    virtual const func_builtins& get_builtins() const override;

    virtual bool is_hashable() const override { return true; }

    virtual hasher unique_hash() const noexcept override {
        static const std::size_t type_hash = typeid(value_string_t).hash_code();
        auto hash                          = hasher();
        hash.update(&type_hash, sizeof(type_hash));
        val_str.hash_update(hash);
        return hash;
    }

protected:
    virtual bool equivalent(const value_t& other) const override {
        return typeid(*this) == typeid(other) && val_str.same_text(other.val_str);
    }
};

using value_string = std::shared_ptr<value_string_t>;

struct value_bool_t : public value_t {
    static constexpr std::uint32_t kind = kind_bool;

    value_bool_t(bool v) {
        val_int = static_cast<int64_t>(v);
        val_flt = static_cast<double>(v);
    }

    std::uint32_t kinds() const noexcept override { return kind; }

    virtual std::string type() const override { return "Boolean"; }

    virtual int64_t as_int() const override { return val_int; }

    double as_float() const override { return val_flt; }

    virtual bool as_bool() const override { return val_int; }

    virtual string as_string() const override { return std::string(val_int ? "True" : "False"); }

    virtual const func_builtins& get_builtins() const override;

    virtual bool is_numeric() const override { return true; }

    virtual bool is_hashable() const override { return true; }

    virtual hasher unique_hash() const noexcept override { return value_int_t::hash_of(val_int); }
protected:
    virtual bool equivalent(const value_t& other) const override {
        return other.is_numeric() && val_int == other.val_int && val_flt == other.val_flt;
    }

    virtual bool nonequal(const value_t& other) const override {
        return !(typeid(*this) == typeid(other) && val_int == other.val_int);
    }
};

using value_bool = std::shared_ptr<value_bool_t>;

struct value_array_t : public value_t {
    static constexpr std::uint32_t kind = kind_array;

    std::uint32_t kinds() const noexcept override { return kind; }

    value_array_t() = default;

    value_array_t(value& v) { val_arr = v->val_arr; }

    value_array_t(std::vector<value>&& arr) { val_arr = arr; }

    value_array_t(const std::vector<value>& arr) { val_arr = arr; }

    void reverse() {
        if (is_immutable()) { throw std::runtime_error("Attempting to modify immutable type"); }
        std::reverse(val_arr.begin(), val_arr.end());
    }

    void push_back(const value& val) {
        if (is_immutable()) { throw std::runtime_error("Attempting to modify immutable type"); }
        val_arr.push_back(val);
    }

    void push_back(value&& val) {
        if (is_immutable()) { throw std::runtime_error("Attempting to modify immutable type"); }
        val_arr.push_back(std::move(val));
    }

    value pop_at(int64_t index) {
        if (is_immutable()) { throw std::runtime_error("Attempting to modify immutable type"); }
        if (index < 0) { index = static_cast<int64_t>(val_arr.size()) + index; }
        if (index < 0 || index >= static_cast<int64_t>(val_arr.size())) {
            throw std::runtime_error("Index " + std::to_string(index) +
                                     " out of bounds for array of size " +
                                     std::to_string(val_arr.size()));
        }
        value val = val_arr.at(static_cast<size_t>(index));
        val_arr.erase(val_arr.begin() + index);
        return val;
    }

    virtual std::string type() const override { return "Array"; }

    virtual bool is_immutable() const override { return false; }

    virtual const std::vector<value>& as_array() const override { return val_arr; }

    virtual string as_string() const override {
        const bool immutable = is_immutable();
        std::ostringstream ss;
        ss << (immutable ? "(" : "[");
        for (size_t i = 0; i < val_arr.size(); i++) {
            if (i > 0) ss << ", ";
            value val = val_arr.at(i);
            ss << value_to_string_repr(val);
        }
        if (immutable && val_arr.size() == 1) { ss << ","; }
        ss << (immutable ? ")" : "]");
        return string(ss.str(), 0, true);
    }

    virtual bool as_bool() const override { return !val_arr.empty(); }

    virtual value& at(int64_t index, value& default_val) override {
        if (index < 0) { index += val_arr.size(); }
        if (index < 0 || static_cast<size_t>(index) >= val_arr.size()) { return default_val; }
        return val_arr[index];
    }

    virtual value& at(int64_t index) override {
        if (index < 0) { index += val_arr.size(); }
        if (index < 0 || static_cast<size_t>(index) >= val_arr.size()) {
            throw std::runtime_error("Index " + std::to_string(index) +
                                     " out of bounds for array of size " +
                                     std::to_string(val_arr.size()));
        }
        return val_arr[index];
    }

    virtual const func_builtins& get_builtins() const override;

    virtual bool is_hashable() const override {
        if (std::all_of(val_arr.begin(), val_arr.end(), [&](auto& val) -> bool {
                return val->is_immutable() && val->is_hashable();
            })) {
            return true;
        }
        return false;
    }

    virtual hasher unique_hash() const noexcept override {
        auto hash = hasher(typeid(*this));
        for (const auto& val : val_arr) {
            // must use digest to prevent problems from "concatenation" property of hasher
            // for ex. hash of [ "ab", "c" ] should be different from [ "a", "bc" ]
            const size_t val_hash = val->unique_hash().digest();
            hash.update(&val_hash, sizeof(size_t));
        }
        return hash;
    }
protected:
    virtual bool equivalent(const value_t& other) const override {
        return typeid(*this) == typeid(other) && is_hashable() && other.is_hashable() &&
               std::equal(val_arr.begin(), val_arr.end(), other.val_arr.begin(),
                          other.val_arr.end(), value_equivalence());
    }
};

using value_array = std::shared_ptr<value_array_t>;

struct value_tuple_t : public value_array_t {
    static constexpr std::uint32_t kind = kind_tuple;

    std::uint32_t kinds() const noexcept override { return kind | value_array_t::kind; }

    value_tuple_t(value& v) { val_arr = v->val_arr; }

    value_tuple_t(std::vector<value>&& arr) { val_arr = std::move(arr); }

    value_tuple_t(const std::vector<value>& arr) { val_arr = arr; }

    value_tuple_t(const std::pair<value, value>& pair) {
        val_arr.push_back(pair.first);
        val_arr.push_back(pair.second);
    }

    virtual std::string type() const override { return "Tuple"; }

    virtual bool is_immutable() const override { return true; }
};

using value_tuple = std::shared_ptr<value_tuple_t>;

struct value_object_t : public value_t {
    using key_index = std::unordered_map<value, value, value_hasher, value_equivalence>;
    // A small object finds a key by scanning val_obj; past kLinearKeys keys it keeps this index.
    // Equal keys hash alike, so both find the same entry.
    static constexpr std::size_t kLinearKeys = 8;
    std::unique_ptr<key_index> index;
    bool has_builtins = true; // context and loop objects do not have builtins
    value_object_t()  = default;

    static constexpr std::uint32_t kind = kind_object;

    std::uint32_t kinds() const noexcept override { return kind; }

    value_object_t(value& v) {
        val_obj = v->val_obj;
        if (val_obj.size() > kLinearKeys) build_index();
    }

    void build_index() {
        index = std::make_unique<key_index>();
        for (const auto& pair : val_obj) { (*index)[pair.first] = pair.second; }
    }

    // The value stored under an (already checked) key, or null.
    value* find_value(const value& key) {
        if (index) {
            const auto found = index->find(key);
            return found == index->end() ? nullptr : &found->second;
        }
        for (auto& pair : val_obj) {
            if (*pair.first == *key) return &pair.second;
        }
        return nullptr;
    }

    value_object_t(const std::map<value, value>& obj) {
        for (const auto& pair : obj) { insert(pair.first, pair.second); }
    }

    value_object_t(const std::vector<std::pair<value, value>>& obj) {
        for (const auto& pair : obj) { insert(pair.first, pair.second); }
    }

    void insert(const std::string& key, const value& val) {
        insert(mk_val<value_string>(key), val);
    }

    virtual std::string type() const override { return "Object"; }

    virtual bool is_immutable() const override { return false; }

    virtual const std::vector<std::pair<value, value>>& as_ordered_object() const override {
        return val_obj;
    }

    virtual string as_string() const override {
        std::ostringstream ss;
        ss << "{";
        for (size_t i = 0; i < val_obj.size(); i++) {
            if (i > 0) ss << ", ";
            auto& [key, val] = val_obj.at(i);
            ss << value_to_string_repr(key) << ": " << value_to_string_repr(val);
        }
        ss << "}";
        return string(ss.str(), 0, true);
    }

    virtual bool as_bool() const override { return !val_obj.empty(); }

    virtual bool has_key(const value& key) override {
        ensure_hashable_key(key);
        return find_value(key) != nullptr;
    }

    static void ensure_hashable_key(const value& key) {
        if (!key->is_immutable() || !key->is_hashable()) {
            throw std::runtime_error("Object key of unhashable type: " + key->type());
        }
    }

    virtual void insert(const value& key, const value& val) override {
        if (is_immutable()) { throw std::runtime_error("Attempting to modify immutable type"); }
        ensure_hashable_key(key);
        if (index) {
            const auto [slot, inserted] = index->try_emplace(key, val);
            if (inserted) {
                val_obj.push_back({key, val});
                return;
            }
            slot->second = val;
        }
        // the key exists: replace its value in the ordered list instead of appending
        for (auto& pair : val_obj) {
            if (*(pair.first) == *key) {
                pair.second = val;
                return;
            }
        }
        val_obj.push_back({key, val});
        if (val_obj.size() > kLinearKeys) build_index();
    }

    virtual value& at(const value& key, value& default_val) override {
        ensure_hashable_key(key);
        value* found = find_value(key);
        return found ? *found : default_val;
    }

    virtual value& at(const value& key) override {
        ensure_hashable_key(key);
        value* found = find_value(key);
        if (!found) {
            throw std::runtime_error("Key '" + key->as_string().str() +
                                     "' not found in value of type " + type());
        }
        return *found;
    }

    virtual value& at(const std::string& key, value& default_val) override {
        value key_val = mk_val<value_string>(key);
        return at(key_val, default_val);
    }

    virtual value& at(const std::string& key) override {
        value key_val = mk_val<value_string>(key);
        return at(key_val);
    }

    virtual const func_builtins& get_builtins() const override;

    virtual bool is_hashable() const override {
        if (std::all_of(val_obj.begin(), val_obj.end(), [&](auto& pair) -> bool {
                const auto& val = pair.second;
                return val->is_immutable() && val->is_hashable();
            })) {
            return true;
        }
        return false;
    }

    virtual hasher unique_hash() const noexcept override {
        auto hash = hasher(typeid(*this));
        for (const auto& [key, val] : val_obj) {
            // must use digest to prevent problems from "concatenation" property of hasher
            // for ex. hash of key="ab", value="c" should be different from key="a", value="bc"
            const size_t key_hash = key->unique_hash().digest();
            const size_t val_hash = val->unique_hash().digest();
            hash.update(&key_hash, sizeof(key_hash));
            hash.update(&val_hash, sizeof(val_hash));
        }
        return hash;
    }
protected:
    virtual bool equivalent(const value_t& other) const override {
        return typeid(*this) == typeid(other) && is_hashable() && other.is_hashable() &&
               std::equal(val_obj.begin(), val_obj.end(), other.val_obj.begin(),
                          other.val_obj.end(), value_equivalence());
    }
};

using value_object = std::shared_ptr<value_object_t>;

//
// none and undefined types
//

struct value_none_t : public value_t {
    static constexpr std::uint32_t kind = kind_none;

    std::uint32_t kinds() const noexcept override { return kind; }

    virtual std::string type() const override { return "None"; }

    virtual bool is_none() const override { return true; }

    virtual bool as_bool() const override { return false; }

    virtual string as_string() const override { return string(type()); }

    virtual std::string as_repr() const override { return type(); }

    virtual const func_builtins& get_builtins() const override;

    virtual bool is_hashable() const override { return true; }

    virtual hasher unique_hash() const noexcept override { return hasher(typeid(*this)); }
protected:
    virtual bool equivalent(const value_t& other) const override {
        return typeid(*this) == typeid(other);
    }
};

using value_none = std::shared_ptr<value_none_t>;

struct value_undefined_t : public value_t {
    std::string hint; // for debugging, to indicate where undefined came from

    value_undefined_t(const std::string& h = "") : hint(h) {}

    static constexpr std::uint32_t kind = kind_undefined;

    std::uint32_t kinds() const noexcept override { return kind; }

    virtual std::string type() const override {
        return hint.empty() ? "Undefined" : "Undefined (hint: '" + hint + "')";
    }

    virtual bool is_undefined() const override { return true; }

    string as_string() const override { return {}; }

    virtual bool as_bool() const override { return false; }

    virtual std::string as_repr() const override { return type(); }

    virtual const func_builtins& get_builtins() const override;

    virtual hasher unique_hash() const noexcept override { return hasher(typeid(*this)); }
protected:
    virtual bool equivalent(const value_t& other) const override {
        return is_undefined() == other.is_undefined();
    }
};

using value_undefined = std::shared_ptr<value_undefined_t>;

//
// function type
//

struct func_args {
public:
    std::string func_name; // for error messages
    context& ctx;

    func_args(context& ctx) : ctx(ctx) {}

    value get_kwarg(const std::string& key, value default_val) const;
    value get_kwarg_or_pos(const std::string& key, size_t pos) const;
    value get_pos(size_t pos) const;
    value get_pos(size_t pos, value default_val) const;
    const std::vector<value>& get_args() const;

    size_t count() const { return args.size(); }

    void push_back(const value& val);
    void push_front(const value& val);

    void ensure_count(size_t min, size_t max = 999) const {
        size_t n = args.size();
        if (n < min || n > max) {
            throw std::runtime_error("Function '" + func_name + "' expected between " +
                                     std::to_string(min) + " and " + std::to_string(max) +
                                     " arguments, got " + std::to_string(n));
        }
    }

    template <typename T>
    void ensure_val(const value& ptr) const {
        if (!is_val<T>(ptr)) {
            throw std::runtime_error("Function '" + func_name + "' expected value of type " +
                                     std::string(typeid(T).name()) + ", got " + ptr->type());
        }
    }

    void ensure_count(bool require0, bool require1, bool require2, bool require3) const {
        static auto bool_to_int = [](bool b) { return b ? 1 : 0; };
        size_t required = bool_to_int(require0) + bool_to_int(require1) + bool_to_int(require2) +
                          bool_to_int(require3);
        ensure_count(required);
    }

    template <typename T0>
    void ensure_vals(bool required0 = true) const {
        ensure_count(required0, false, false, false);
        if (required0 && args.size() > 0) ensure_val<T0>(args[0]);
    }

    template <typename T0, typename T1>
    void ensure_vals(bool required0 = true, bool required1 = true) const {
        ensure_count(required0, required1, false, false);
        if (required0 && args.size() > 0) ensure_val<T0>(args[0]);
        if (required1 && args.size() > 1) ensure_val<T1>(args[1]);
    }

    template <typename T0, typename T1, typename T2>
    void ensure_vals(bool required0 = true, bool required1 = true, bool required2 = true) const {
        ensure_count(required0, required1, required2, false);
        if (required0 && args.size() > 0) ensure_val<T0>(args[0]);
        if (required1 && args.size() > 1) ensure_val<T1>(args[1]);
        if (required2 && args.size() > 2) ensure_val<T2>(args[2]);
    }

    template <typename T0, typename T1, typename T2, typename T3>
    void ensure_vals(bool required0 = true, bool required1 = true, bool required2 = true,
                     bool required3 = true) const {
        ensure_count(required0, required1, required2, required3);
        if (required0 && args.size() > 0) ensure_val<T0>(args[0]);
        if (required1 && args.size() > 1) ensure_val<T1>(args[1]);
        if (required2 && args.size() > 2) ensure_val<T2>(args[2]);
        if (required3 && args.size() > 3) ensure_val<T3>(args[3]);
    }
private:
    std::vector<value> args;
};

struct value_func_t : public value_t {
    std::string name;
    value arg0; // bound "this" argument, if any

    static constexpr std::uint32_t kind = kind_func;

    std::uint32_t kinds() const noexcept override { return kind; }

    value_func_t(const std::string& name, const func_handler& func) : name(name) {
        val_func = func;
    }

    value_func_t(const std::string& name, const func_handler& func, const value& arg_this)
        : name(name), arg0(arg_this) {
        val_func = func;
    }

    virtual value invoke(const func_args& args) const override {
        func_args new_args(args); // copy
        new_args.func_name = name;
        if (arg0) { new_args.push_front(arg0); }
        return val_func(new_args);
    }

    virtual std::string type() const override { return "Function"; }

    virtual std::string as_repr() const override {
        return type() + "<" + name + ">(" + (arg0 ? arg0->as_repr() : "") + ")";
    }

    virtual bool is_hashable() const override { return false; }

    virtual hasher unique_hash() const noexcept override {
        // Note: this is unused for now, we don't support function as object keys
        // use function pointer as unique identifier
        const auto target = val_func.target<func_hptr>();
        return hasher(typeid(*this)).update(&target, sizeof(target));
    }
protected:
    virtual bool equivalent(const value_t& other) const override {
        // Note: this is unused for now, we don't support function as object keys
        // compare function pointers
        // (val_func == other.val_func does not work as std::function::operator== is only used for
        // nullptr check)
        const auto target_this  = this->val_func.target<func_hptr>();
        const auto target_other = other.val_func.target<func_hptr>();
        return typeid(*this) == typeid(other) && target_this == target_other;
    }
};

using value_func = std::shared_ptr<value_func_t>;

// special value for kwarg
struct value_kwarg_t : public value_t {
    std::string key;
    value val;

    value_kwarg_t(const std::string& k, const value& v) : key(k), val(v) {}

    static constexpr std::uint32_t kind = kind_kwarg;

    std::uint32_t kinds() const noexcept override { return kind; }

    virtual std::string type() const override { return "KwArg"; }

    virtual std::string as_repr() const override { return type(); }

    virtual bool is_hashable() const override { return true; }

    virtual hasher unique_hash() const noexcept override {
        const auto type_hash = typeid(*this).hash_code();
        auto hash            = val->unique_hash();
        hash.update(&type_hash, sizeof(type_hash)).update(key.data(), key.size());
        return hash;
    }
protected:
    virtual bool equivalent(const value_t& other) const override {
        const value_kwarg_t& other_val = static_cast<const value_kwarg_t&>(other);
        return typeid(*this) == typeid(other) && key == other_val.key && val == other_val.val;
    }
};

using value_kwarg = std::shared_ptr<value_kwarg_t>;


} // namespace jinja
