#include "lexer.h"
#include "runtime.h"
#include "value.h"
#include "utils.h"
#include "unicode.h"

#include <array>
#include <cmath>
#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace jinja {

// Scopes

value* scope_frame::find_local(std::string_view name) noexcept {
    if (index_) {
        const auto found = index_->find(name);
        return found == index_->end() ? nullptr : &vars_[found->second].second;
    }
    for (auto& [key, val] : vars_) {
        if (key == name) return &val;
    }
    return nullptr;
}

void scope_frame::bind(std::string_view name, const value& val) {
    if (value* slot = find_local(name)) {
        *slot = val;
        return;
    }
    vars_.emplace_back(std::string(name), val);
    if (index_) {
        index_->emplace(vars_.back().first, vars_.size() - 1);
    } else if (vars_.size() > kLinearScan) {
        index_ = std::make_unique<name_index>();
        for (std::size_t i = 0; i < vars_.size(); ++i) index_->emplace(vars_[i].first, i);
    }
}

context::context(std::string src) : context(std::make_shared<const std::string>(std::move(src))) {}

context::context(std::shared_ptr<const std::string> src)
    : owned_state_(std::make_unique<render_state>()), state_(owned_state_.get()),
      frame_(std::make_shared<scope_frame>()) {
    state_->src          = std::move(src);
    state_->current_time = std::time(nullptr);
    frame_->bind("true", mk_val<value_bool>(true));
    frame_->bind("True", mk_val<value_bool>(true));
    frame_->bind("false", mk_val<value_bool>(false));
    frame_->bind("False", mk_val<value_bool>(false));
    frame_->bind("none", mk_val<value_none>());
    frame_->bind("None", mk_val<value_none>());
}

context::context(const context& parent)
    : state_(parent.state_), frame_(std::make_shared<scope_frame>(parent.frame_)) {}

context::context(const context& caller, const std::weak_ptr<scope_frame>& lexical_scope)
    : state_(caller.state_) {
    auto scope = lexical_scope.lock();
    if (!scope) throw std::runtime_error("macro definition scope has expired");
    frame_ = std::make_shared<scope_frame>(std::move(scope));
}

const value* context::find(std::string_view name) const noexcept {
    for (scope_frame* frame = frame_.get(); frame != nullptr; frame = frame->parent().get()) {
        if (const value* found = frame->find_local(name)) return found;
    }
    return nullptr;
}

// Output

// A statement's value as output: strings keep their parts, numbers and booleans are written as
// text, arrays contribute their items, and every other value writes nothing.
static void append_output(string& out, const value& val) {
    if (is_val<value_string>(val)) {
        out.append(cast_val<value_string>(val)->val_str);
    } else if (is_val<value_int>(val) || is_val<value_float>(val) || is_val<value_bool>(val)) {
        out.append_plain(val->as_string().str());
    } else if (is_val<value_array>(val)) {
        for (const auto& item : val->as_array()) { append_output(out, item); }
    }
}

// A value's text: a string's own parts, or the value's string form.
static void append_text(string& out, const value& val) {
    if (is_val<value_string>(val)) {
        out.append(val->val_str);
    } else {
        out.append(val->as_string());
    }
}

// The text a concatenation extends: a temporary string's own parts are taken over (nothing else
// refers to it), any other value's text is copied.
static string extendable_text(value& left) {
    if (is_val<value_string>(left) && left.use_count() == 1) return std::move(left->val_str);
    return left->as_string();
}

static flow render_list(const statements& stmts, context& ctx, string& out) {
    for (const auto& stmt : stmts) {
        const flow result = stmt->render(ctx, out);
        if (result != flow::next) return result;
    }
    return flow::next;
}

// A macro or caller body's output. The parser admits loop control only inside a loop of the same
// body, as Jinja2 does.
static value_string render_body(const statements& body, context& ctx) {
    auto result = mk_val<value_string>();
    if (render_list(body, ctx, result->val_str) != flow::next) {
        throw std::runtime_error("loop control outside a loop");
    }
    return result;
}

static std::string get_line_col(const std::string& source, size_t pos) {
    size_t line = 1;
    size_t col  = 1;
    for (size_t i = 0; i < pos && i < source.size(); i++) {
        if (source[i] == '\n') {
            line++;
            col = 1;
        } else {
            col++;
        }
    }
    return "line " + std::to_string(line) + ", column " + std::to_string(col);
}

static void ensure_key_type_allowed(const value& val) {
    if (!val->is_hashable()) {
        throw std::runtime_error("Type: " + val->type() + " is not allowed as object key");
    }
}

[[noreturn]] static void rethrow_located(const statement& stmt, const context& ctx,
                                         const std::exception& e) {
    static const std::string no_source;
    const std::string& source = ctx.state().src ? *ctx.state().src : no_source;
    std::ostringstream oss;
    if (source.empty()) {
        oss << "\nError executing " << stmt.type() << " at position " << stmt.pos << ": "
            << e.what();
    } else {
        oss << "\n------------\n";
        oss << "While executing " << stmt.type() << " at " << get_line_col(source, stmt.pos)
            << " in source:\n";
        oss << peak_source(source, stmt.pos) << "\n";
        oss << "Error: " << e.what();
    }
    // throw as another exception to avoid repeated formatting
    throw rethrown_exception(oss.str());
}

// execute with error handling
value statement::execute(context& ctx) {
    ctx.checkpoint();
    try {
        return execute_impl(ctx);
    } catch (const rethrown_exception& /* ex */) {
        throw;
    } catch (const not_implemented_exception& /* ex */) { throw; } catch (const std::exception& e) {
        rethrow_located(*this, ctx, e);
    }
}

flow statement::render_impl(context& ctx, string& out) {
    append_output(out, execute_impl(ctx));
    return flow::next;
}

// render with error handling
flow statement::render(context& ctx, string& out) {
    ctx.checkpoint();
    try {
        return render_impl(ctx, out);
    } catch (const rethrown_exception& /* ex */) {
        throw;
    } catch (const not_implemented_exception& /* ex */) { throw; } catch (const std::exception& e) {
        rethrow_located(*this, ctx, e);
    }
}

flow output_statement::render_impl(context& ctx, string& out) {
    append_text(out, operand->execute(ctx));
    return flow::next;
}

value identifier::execute_impl(context& ctx) {
    if (const value* found = ctx.find(val); found && *found && !(*found)->is_undefined()) {
        return *found;
    }
    const auto& builtins = global_builtins();
    if (const auto it = builtins.find(val); it != builtins.end()) {
        return mk_val<value_func>(val, it->second);
    }
    return mk_val<value_undefined>(val);
}

value object_literal::execute_impl(context& ctx) {
    auto obj = mk_val<value_object>();
    for (const auto& pair : val) {
        value key = pair.first->execute(ctx);
        value val = pair.second->execute(ctx);
        obj->insert(key, val);
    }
    return obj;
}

value binary_expression::execute_impl(context& ctx) {
    value left_val = left->execute(ctx);

    // Logical operators
    if (kind == operation::logical_and) {
        return left_val->as_bool() ? right->execute(ctx) : std::move(left_val);
    } else if (kind == operation::logical_or) {
        return left_val->as_bool() ? std::move(left_val) : right->execute(ctx);
    }

    // Equality operators
    value right_val = right->execute(ctx);
    if (kind == operation::equal) {
        return mk_val<value_bool>(*left_val == *right_val);
    } else if (kind == operation::not_equal) {
        return mk_val<value_bool>(!(*left_val == *right_val));
    }

    if (kind == operation::concat) {
        string output = extendable_text(left_val);
        append_text(output, right_val);
        return mk_val<value_string>(std::move(output));
    }

    auto test_is_in = [&]() -> bool {
        static const func_handler& membership = global_builtins().at("test_is_in");
        func_args args(ctx);
        args.push_back(left_val);
        args.push_back(right_val);
        return membership(args)->as_bool();
    };

    // Handle undefined and null values
    if (is_val<value_undefined>(left_val) || is_val<value_undefined>(right_val)) {
        if (is_val<value_undefined>(right_val) && (kind == operation::in || kind == operation::not_in)) {
            // Special case: `anything in undefined` is `false` and `anything not in undefined` is
            // `true`
            return mk_val<value_bool>(kind == operation::not_in);
        }
        throw std::runtime_error("Cannot perform operation " + op.value + " on undefined values");
    } else if (is_val<value_none>(left_val) || is_val<value_none>(right_val)) {
        if (!is_val<value_none>(right_val) && (kind == operation::in || kind == operation::not_in)) {
            // case: none in {'low': 1}
            // A null left operand is looked up like any other value.
            bool member = test_is_in();
            return mk_val<value_bool>(kind == operation::in ? member : !member);
        }
        throw std::runtime_error("Cannot perform operation on null values");
    }

    // Float operations
    if ((is_val<value_int>(left_val) || is_val<value_float>(left_val)) &&
        (is_val<value_int>(right_val) || is_val<value_float>(right_val))) {
        double a = left_val->as_float();
        double b = right_val->as_float();
        if (kind == operation::add || kind == operation::subtract || kind == operation::multiply) {
            double res    = (kind == operation::add) ? a + b : (kind == operation::subtract) ? a - b : a * b;
            bool is_float = is_val<value_float>(left_val) || is_val<value_float>(right_val);
            if (is_float) {
                return mk_val<value_float>(res);
            } else {
                return mk_val<value_int>(static_cast<int64_t>(res));
            }
        } else if (kind == operation::divide) {
            return mk_val<value_float>(a / b);
        } else if (kind == operation::modulo) {
            double rem    = std::fmod(a, b);
            bool is_float = is_val<value_float>(left_val) || is_val<value_float>(right_val);
            if (is_float) {
                return mk_val<value_float>(rem);
            } else {
                return mk_val<value_int>(static_cast<int64_t>(rem));
            }
        } else if (kind == operation::less) {
            return mk_val<value_bool>(a < b);
        } else if (kind == operation::greater) {
            return mk_val<value_bool>(a > b);
        } else if (kind == operation::greater_equal) {
            return mk_val<value_bool>(a >= b);
        } else if (kind == operation::less_equal) {
            return mk_val<value_bool>(a <= b);
        }
    }

    // Array operations
    if (is_val<value_array>(left_val) && is_val<value_array>(right_val)) {
        if (kind == operation::add) {
            auto& left_arr  = left_val->as_array();
            auto& right_arr = right_val->as_array();
            auto result     = mk_val<value_array>();
            for (const auto& item : left_arr) { result->push_back(item); }
            for (const auto& item : right_arr) { result->push_back(item); }
            return result;
        }
    } else if (is_val<value_array>(right_val)) {
        // case: 1 in [0, 1, 2]
        bool member = test_is_in();
        if (kind == operation::in) {
            return mk_val<value_bool>(member);
        } else if (kind == operation::not_in) {
            return mk_val<value_bool>(!member);
        }
    }

    // String concatenation with ~ and +
    if (is_val<value_string>(left_val) && is_val<value_string>(right_val) && kind == operation::add) {
        string output = extendable_text(left_val);
        output.append(right_val->val_str);
        return mk_val<value_string>(std::move(output));
    }

    // Python-style string repetition
    // TODO: support array/tuple repetition (e.g., [1, 2] * 3 → [1, 2, 1, 2, 1, 2])
    if (kind == operation::multiply && ((is_val<value_string>(left_val) && is_val<value_int>(right_val)) ||
                            (is_val<value_int>(left_val) && is_val<value_string>(right_val)))) {
        const auto& str =
            is_val<value_string>(left_val) ? left_val->as_string() : right_val->as_string();
        const int64_t repeat =
            is_val<value_int>(right_val) ? right_val->as_int() : left_val->as_int();
        auto res = mk_val<value_string>();
        if (repeat <= 0) { return res; }
        for (int64_t i = 0; i < repeat; ++i) {
            ctx.checkpoint();
            res->val_str.append(str);
        }
        return res;
    }

    // String membership
    if (is_val<value_string>(left_val) && is_val<value_string>(right_val)) {
        // case: "a" in "abc"
        bool member = test_is_in();
        if (kind == operation::in) {
            return mk_val<value_bool>(member);
        } else if (kind == operation::not_in) {
            return mk_val<value_bool>(!member);
        }
    }

    // Value key in object
    if (is_val<value_object>(right_val)) {
        // case: key in {key: value}
        bool member = test_is_in();
        if (kind == operation::in) {
            return mk_val<value_bool>(member);
        } else if (kind == operation::not_in) {
            return mk_val<value_bool>(!member);
        }
    }

    throw std::runtime_error("Unknown operator \"" + op.value + "\" between " + left_val->type() +
                             " and " + right_val->type());
}

static value try_builtin_func(context& ctx, const std::string& name, value& input,
                              bool undef_on_missing = false) {
    (void)ctx;
    const auto& builtins = input->get_builtins();
    const auto it        = builtins.find(name);
    if (it != builtins.end()) { return mk_val<value_func>(name, it->second, input); }
    if (undef_on_missing) { return mk_val<value_undefined>(name); }
    throw std::runtime_error("Unknown (built-in) filter '" + name + "' for type " + input->type());
}

// The input's builtin a filter names. Applying a filter calls it with the input first, exactly as
// invoking the builtin bound to the input does.
static const func_handler& filter_builtin(const std::string& name, const value& input) {
    const auto& builtins = input->get_builtins();
    const auto it        = builtins.find(name);
    if (it == builtins.end()) {
        throw std::runtime_error("Unknown (built-in) filter '" + name + "' for type " +
                                 input->type());
    }
    return it->second;
}

static value apply_filter(value input, const statement_ptr& filter, context& ctx) {
    const auto filter_alias = [](const std::string& filter_id) -> const std::string& {
        static const std::string length = "length", default_value = "default",
                                 escape = "escape", strip = "strip";
        if (filter_id == "count") return length;
        if (filter_id == "d") return default_value;
        if (filter_id == "e") return escape;
        if (filter_id == "trim") return strip;
        return filter_id;
    };

    if (is_stmt<identifier>(filter)) {
        const std::string& filter_id = filter_alias(cast_stmt<identifier>(filter)->val);

        // TODO: Refactor filters so this coercion can be done automatically
        if (!input->is_undefined() && !is_val<value_string>(input) &&
            (filter_id == "capitalize" || filter_id == "lower" || filter_id == "replace" ||
             filter_id == "strip" || filter_id == "title" || filter_id == "upper" ||
             filter_id == "wordcount")) {
            input = mk_val<value_string>(input->as_string());
        }
        const func_handler& handler = filter_builtin(filter_id, input);
        func_args args(ctx);
        args.func_name = filter_id;
        args.push_back(input);
        return handler(args);

    } else if (is_stmt<call_expression>(filter)) {
        auto call = cast_stmt<call_expression>(filter);
        if (!is_stmt<identifier>(call->callee)) {
            throw std::runtime_error("Filter callee must be an identifier");
        }
        const std::string& filter_id = filter_alias(cast_stmt<identifier>(call->callee)->val);

        func_args args(ctx);
        args.func_name = filter_id;
        args.push_back(input);
        for (const auto& arg_expr : call->args) { args.push_back(arg_expr->execute(ctx)); }

        return filter_builtin(filter_id, input)(args);

    } else {
        throw std::runtime_error("Invalid filter expression");
    }
}

value filter_expression::execute_impl(context& ctx) {
    return apply_filter(operand ? operand->execute(ctx) : val, filter, ctx);
}

flow filter_statement::render_impl(context& ctx, string& out) {
    auto block = mk_val<value_string>();
    // As in Jinja2, loop control inside the block leaves the loop without the filtered output.
    if (const flow result = render_list(body, ctx, block->val_str); result != flow::next) {
        return result;
    }
    append_output(out, apply_filter(std::move(block), filter, ctx));
    return flow::next;
}

test_expression::test_expression(statement_ptr&& operand, bool negate, statement_ptr&& test)
    : operand(std::move(operand)), negate(negate), test(std::move(test)) {
    chk_type<expression>(this->operand);
    chk_type<identifier, call_expression>(this->test);
    test_name = cast_stmt<identifier>(this->test);
    test_call = cast_stmt<call_expression>(this->test);
    const identifier* name = test_call ? cast_stmt<identifier>(test_call->callee) : test_name;
    if (name) {
        const auto& builtins = global_builtins();
        if (const auto it = builtins.find("test_is_" + name->val); it != builtins.end()) {
            handler = &it->second;
        }
    }
}

value test_expression::execute_impl(context& ctx) {
    // NOTE: "value is something" translates to function call "test_is_something(value)"
    const std::string* test_id = nullptr;
    value input                = operand->execute(ctx);

    func_args args(ctx);
    args.push_back(input);

    if (test_name) {
        test_id = &test_name->val;
    } else if (test_call) {
        const auto* callee = cast_stmt<identifier>(test_call->callee);
        if (!callee) { throw std::runtime_error("Test callee must be an identifier"); }
        test_id = &callee->val;

        for (const auto& arg_expr : test_call->args) { args.push_back(arg_expr->execute(ctx)); }

    } else {
        throw std::runtime_error("Invalid test expression");
    }

    if (!handler) { throw std::runtime_error("Unknown test '" + *test_id + "'"); }

    auto res = (*handler)(args);

    if (negate) {
        return mk_val<value_bool>(!res->as_bool());
    } else {
        return res;
    }
}

value unary_expression::execute_impl(context& ctx) {
    value operand_val = argument->execute(ctx);

    if (op.value == "not") {
        return mk_val<value_bool>(!operand_val->as_bool());
    } else if (op.value == "-") {
        if (is_val<value_int>(operand_val)) {
            return mk_val<value_int>(-operand_val->as_int());
        } else if (is_val<value_float>(operand_val)) {
            return mk_val<value_float>(-operand_val->as_float());
        } else {
            throw std::runtime_error("Unary - operator requires numeric operand");
        }
    }

    throw std::runtime_error("Unknown unary operator '" + op.value + "'");
}

flow if_statement::render_impl(context& ctx, string& out) {
    const value test_val = test->execute(ctx);
    return render_list(test_val->as_bool() ? body : alternate, ctx, out);
}

namespace {

// Binds one element to a loop's target: a name, or a tuple of names unpacked from the element.
void bind_target(context& scope, const statement_ptr& target, const value& item) {
    if (const auto* name = cast_stmt<identifier>(target)) {
        scope.set_val(name->val, item);
        return;
    }
    const auto* tuple = cast_stmt<tuple_literal>(target);
    if (!tuple) throw std::runtime_error("Invalid loop variable(s): " + target->type());
    if (!is_val<value_array>(item)) {
        throw std::runtime_error("Cannot unpack non-iterable type: " + item->type());
    }
    const auto& items = item->as_array();
    if (tuple->val.size() != items.size()) {
        throw std::runtime_error(std::string("Too ") +
                                 (tuple->val.size() > items.size() ? "few" : "many") +
                                 " items to unpack");
    }
    for (size_t j = 0; j < tuple->val.size(); ++j) {
        const auto* element = cast_stmt<identifier>(tuple->val[j]);
        if (!element) {
            throw std::runtime_error("Cannot unpack non-identifier type: " +
                                     tuple->val[j]->type());
        }
        scope.set_val(element->val, items[j]);
    }
}

// Jinja2's LoopContext: one `loop` object per loop, advanced by the loop. The iterable is read
// live, element by element, as Jinja2 reads a list, and a member is computed when the template
// reads it: only `last`, `nextitem`, `length` and `revindex` look ahead, and a filtered loop
// evaluates its condition for an element when the loop or a look-ahead reaches it.
class value_loop_t final : public value_object_t {
public:
    value_loop_t(value iterable, const statement_ptr& target, statement* condition,
                 context* condition_scope)
        : iterable_(std::move(iterable)), target_(target), condition_(condition),
          condition_scope_(condition_scope) {
        has_builtins = false; // loop object has no builtins
        if (is_val<value_object>(iterable_)) {
            kind_ = source::mapping;
            keys_ = is_stmt<identifier>(target);
        } else if (is_val<value_string>(iterable_)) {
            kind_  = source::characters;
            text_  = iterable_->as_string();
            chars_ = unicode::characters(text_.str());
        } else {
            kind_ = source::sequence;
        }
    }

    // Makes the next element current; false when the loop is over.
    bool advance() {
        value next;
        if (!ahead_.empty()) {
            next = std::move(ahead_.front());
            ahead_.pop_front();
        } else if (!fetch(next)) {
            return false;
        }
        previous_ = std::move(current_);
        current_  = std::move(next);
        ++index0_;
        return true;
    }

    const value& current() const noexcept { return current_; }

    // The loop has ended: its condition scope is gone, so a later look-ahead finds no element.
    void finish() noexcept { condition_scope_ = nullptr; }

    value& at(const value& key, value& default_val) override {
        ensure_hashable_key(key);
        const auto which = field_of(key);
        if (!which) return default_val;
        return slots_[static_cast<std::size_t>(*which)] = read(*which);
    }

    value& at(const value& key) override {
        ensure_hashable_key(key);
        const auto which = field_of(key);
        if (!which) {
            throw std::runtime_error("Key '" + key->as_string().str() +
                                     "' not found in value of type " + type());
        }
        return slots_[static_cast<std::size_t>(*which)] = read(*which);
    }

    bool has_key(const value& key) override {
        ensure_hashable_key(key);
        return field_of(key).has_value();
    }

    void insert(const value&, const value&) override {
        throw std::runtime_error("cannot assign attribute on non-namespace object");
    }

    const std::vector<std::pair<value, value>>& as_ordered_object() const override {
        materialize();
        return val_obj;
    }

    string as_string() const override {
        materialize();
        return value_object_t::as_string();
    }

    bool as_bool() const override { return true; }

    bool is_hashable() const override {
        materialize();
        return value_object_t::is_hashable();
    }

    hasher unique_hash() const noexcept override {
        try {
            materialize();
        } catch (...) { /* hash the members read so far */
        }
        return value_object_t::unique_hash();
    }

protected:
    bool equivalent(const value_t& other) const override {
        materialize();
        if (const auto* loop = dynamic_cast<const value_loop_t*>(&other)) loop->materialize();
        return value_object_t::equivalent(other);
    }

private:
    enum class source { sequence, mapping, characters };
    // The members, in the order the object form lists them.
    enum class field : std::size_t {
        index,
        index0,
        revindex,
        revindex0,
        first,
        last,
        length,
        previtem,
        nextitem,
    };
    static constexpr std::size_t kFields = 9;
    static constexpr std::array<std::string_view, kFields> kNames{
        "index", "index0", "revindex", "revindex0", "first",
        "last",  "length", "previtem", "nextitem"};

    static std::optional<field> field_of(const value& key) {
        if (!is_val<value_string>(key)) return std::nullopt;
        const auto& parts = key->val_str.parts;
        const std::string joined = parts.size() == 1 ? std::string() : key->val_str.str();
        const std::string_view name = parts.size() == 1 ? std::string_view(parts[0].val) : joined;
        for (std::size_t i = 0; i < kFields; ++i) {
            if (kNames[i] == name) return static_cast<field>(i);
        }
        return std::nullopt;
    }

    static const value& field_key(std::size_t i) {
        static const std::array<value, kFields> keys = [] {
            std::array<value, kFields> result;
            for (std::size_t j = 0; j < kFields; ++j) {
                result[j] = mk_val<value_string>(std::string(kNames[j]));
            }
            return result;
        }();
        return keys[i];
    }

    std::size_t size() const {
        switch (kind_) {
        case source::mapping: return iterable_->as_ordered_object().size();
        case source::characters: return chars_.size();
        default: return iterable_->as_array().size();
        }
    }

    // The iterable's next element, read live; once the end is reached the loop stays exhausted.
    bool next_element(value& item) {
        if (exhausted_) return false;
        if (cursor_ >= size()) {
            exhausted_ = true;
            return false;
        }
        const std::size_t i = cursor_++;
        switch (kind_) {
        case source::mapping: {
            const auto& pair = iterable_->as_ordered_object()[i];
            // `for key in mapping` iterates keys; a tuple target unpacks (key, value) pairs.
            if (keys_) {
                item = pair.first;
            } else {
                item = mk_val<value_tuple>(pair);
            }
            break;
        }
        case source::characters:
            item = mk_val<value_string>(text_.cut_bytes(chars_[i].begin, chars_[i].end));
            break;
        default: item = iterable_->as_array()[i]; break;
        }
        return true;
    }

    // The next element the loop iterates: the next one passing the condition, if any.
    bool fetch(value& item) {
        while (next_element(item)) {
            if (!condition_) return true;
            if (!condition_scope_) {
                exhausted_ = true;
                return false;
            }
            bind_target(*condition_scope_, target_, item);
            if (condition_->execute(*condition_scope_)->as_bool()) return true;
        }
        return false;
    }

    const value* peek() {
        if (ahead_.empty()) {
            value item;
            if (!fetch(item)) return nullptr;
            ahead_.push_back(std::move(item));
        }
        return &ahead_.front();
    }

    // As Jinja2: the iterable's length when it is first asked for; a filtered loop counts the
    // elements passing its condition, evaluating it for every remaining element.
    std::int64_t count() {
        if (!length_) {
            if (!condition_) {
                length_ = static_cast<std::int64_t>(size());
            } else {
                for (value item; fetch(item);) ahead_.push_back(std::move(item));
                length_ = index0_ + 1 + static_cast<std::int64_t>(ahead_.size());
            }
        }
        return *length_;
    }

    value read(field which) {
        switch (which) {
        case field::index: return mk_val<value_int>(index0_ + 1);
        case field::index0: return mk_val<value_int>(index0_);
        case field::revindex: return mk_val<value_int>(count() - index0_);
        case field::revindex0: return mk_val<value_int>(count() - index0_ - 1);
        case field::first: return mk_val<value_bool>(index0_ == 0);
        case field::last: return mk_val<value_bool>(peek() == nullptr);
        case field::length: return mk_val<value_int>(count());
        case field::previtem:
            return index0_ > 0 ? previous_ : mk_val<value_undefined>("previtem");
        case field::nextitem: {
            const value* next = peek();
            return next ? *next : mk_val<value_undefined>("nextitem");
        }
        }
        throw std::logic_error("invalid loop member");
    }

    // The object form (listing, printing, comparing): every member as currently read.
    void materialize() const {
        auto& self = const_cast<value_loop_t&>(*this);
        self.val_obj.clear();
        for (std::size_t i = 0; i < kFields; ++i) {
            self.val_obj.push_back({field_key(i), self.read(static_cast<field>(i))});
        }
    }

    value iterable_;
    const statement_ptr& target_;
    statement* condition_;
    context* condition_scope_;
    source kind_ = source::sequence;
    bool keys_   = false;
    string text_;
    std::vector<unicode::Character> chars_;
    std::size_t cursor_ = 0;
    bool exhausted_     = false;
    std::deque<value> ahead_;
    value current_;
    value previous_;
    std::int64_t index0_ = -1;
    std::optional<std::int64_t> length_;
    std::array<value, kFields> slots_;
};

} // namespace

flow for_statement::render_impl(context& ctx, string& out) {
    context scope(ctx); // new scope for loop variables

    const auto* select           = cast_stmt<select_expression>(iterable);
    const statement_ptr& subject = select ? select->lhs : iterable;
    value iterable_val           = subject->execute(scope);

    if (iterable_val->is_undefined()) { iterable_val = mk_val<value_array>(); }

    if (!is_val<value_array>(iterable_val) && !is_val<value_object>(iterable_val) &&
        !is_val<value_string>(iterable_val)) {
        throw std::runtime_error("Expected iterable or object type in for loop: got " +
                                 iterable_val->type());
    }

    // A filtered loop's condition sees the enclosing scope and the loop target, not the body's.
    std::optional<context> condition_scope;
    if (select && select->test) condition_scope.emplace(ctx);
    const auto loop = std::make_shared<value_loop_t>(
        std::move(iterable_val), loopvar, select ? select->test.get() : nullptr,
        condition_scope ? &*condition_scope : nullptr);
    struct finish_guard {
        value_loop_t& loop;
        ~finish_guard() { loop.finish(); }
    } finished{*loop};

    // As in Jinja2, the else block runs when no iteration completed its body.
    bool iterated = false;
    while (loop->advance()) {
        ctx.checkpoint();
        scope.set_val("loop", loop);
        bind_target(scope, loopvar, loop->current());
        const flow result = render_list(body, scope, out);
        if (result == flow::break_loop) break;
        if (result == flow::continue_loop) continue;
        iterated = true;
    }
    // The else block belongs to the enclosing scope, and loop control there is the enclosing
    // loop's.
    if (!iterated) return render_list(default_block, ctx, out);
    return flow::next;
}

set_statement::set_statement(statement_ptr&& assignee, statement_ptr&& value, statements&& body)
    : assignee(std::move(assignee)), val(std::move(value)), body(std::move(body)),
      assignee_name(cast_stmt<identifier>(this->assignee)) {
    chk_type<expression>(this->assignee);
    chk_type<expression>(this->val);
}

flow set_statement::render_impl(context& ctx, string&) {
    value rhs;
    if (val) {
        rhs = val->execute(ctx);
    } else {
        auto block = mk_val<value_string>();
        // As in Jinja2, loop control inside the block leaves the variable unassigned.
        if (const flow result = render_list(body, ctx, block->val_str); result != flow::next) {
            return result;
        }
        rhs = std::move(block);
    }

    if (assignee_name) {
        // case: {% set my_var = value %}
        ctx.set_val(assignee_name->val, rhs);

    } else if (is_stmt<tuple_literal>(assignee)) {
        // case: {% set a, b = value %}
        auto tuple = cast_stmt<tuple_literal>(assignee);
        if (!is_val<value_array>(rhs)) {
            throw std::runtime_error("Cannot unpack non-iterable type in set: " + rhs->type());
        }
        auto& arr = rhs->as_array();
        if (arr.size() != tuple->val.size()) {
            throw std::runtime_error(std::string("Too ") +
                                     (tuple->val.size() > arr.size() ? "few" : "many") +
                                     " items to unpack in set");
        }
        for (size_t i = 0; i < tuple->val.size(); ++i) {
            auto& elem = tuple->val[i];
            if (!is_stmt<identifier>(elem)) {
                throw std::runtime_error("Cannot unpack to non-identifier in set: " + elem->type());
            }
            auto var_name = cast_stmt<identifier>(elem)->val;
            ctx.set_val(var_name, arr[i]);
        }

    } else if (is_stmt<member_expression>(assignee)) {
        // case: {% set ns.my_var = value %}
        auto member = cast_stmt<member_expression>(assignee);
        if (member->computed) { throw std::runtime_error("Cannot assign to computed member"); }
        if (!is_stmt<identifier>(member->property)) {
            throw std::runtime_error("Cannot assign to member with non-identifier property");
        }
        auto prop_name = cast_stmt<identifier>(member->property)->val;

        value object = member->object->execute(ctx);
        if (!is_val<value_object>(object)) {
            throw std::runtime_error("Cannot assign to member of non-object");
        }
        auto obj_ptr = cast_val<value_object>(object);
        obj_ptr->insert(prop_name, rhs);

    } else {
        throw std::runtime_error("Invalid LHS inside assignment expression: " + assignee->type());
    }
    return flow::next;
}

static void bind_parameters(const std::string& name, const statements& parameters,
                            const func_args& args, context& ctx) {
    std::vector<value> positional;
    std::map<std::string, value> named;
    for (const auto& input : args.get_args()) {
        if (is_val<value_kwarg>(input)) {
            const auto* keyword = cast_val<value_kwarg>(input);
            if (!named.emplace(keyword->key, keyword->val).second)
                throw std::runtime_error("duplicate macro argument: " + keyword->key);
        } else
            positional.push_back(input);
    }
    if (positional.size() > parameters.size())
        throw std::runtime_error("too many arguments to macro " + name);
    for (size_t i = 0; i < parameters.size(); ++i) {
        const auto* plain        = cast_stmt<identifier>(parameters[i]);
        const auto* with_default = cast_stmt<keyword_argument_expression>(parameters[i]);
        if (!plain && (!with_default || !is_stmt<identifier>(with_default->key))) {
            throw std::runtime_error("invalid parameter in macro " + name);
        }
        const auto key = plain ? plain->val : cast_stmt<identifier>(with_default->key)->val;
        value input;
        if (const auto it = named.find(key); it != named.end()) {
            if (i < positional.size()) throw std::runtime_error("duplicate macro argument: " + key);
            input = it->second;
            named.erase(it);
        } else if (i < positional.size())
            input = positional[i];
        else if (with_default)
            input = with_default->val->execute(ctx);
        else
            input = mk_val<value_undefined>(key);
        ctx.set_val(key, input);
    }
    if (!named.empty()) throw std::runtime_error("unknown macro argument: " + named.begin()->first);
}

value macro_statement::execute_impl(context& ctx) {
    if (!is_stmt<identifier>(this->name)) {
        throw std::runtime_error("Macro name must be an identifier");
    }
    std::string name = cast_stmt<identifier>(this->name)->val;

    const func_handler func = [this, name,
                               scope = ctx.capture_scope()](const func_args& args) -> value {
        context macro_ctx(args.ctx, scope);
        if (const value* caller = args.ctx.find("caller");
            caller && *caller && !(*caller)->is_undefined()) {
            macro_ctx.set_val("caller", *caller);
        }

        bind_parameters(name, this->args, args, macro_ctx);

        // execute macro body
        return render_body(this->body, macro_ctx);
    };

    ctx.set_val(name, mk_val<value_func>(name, func));
    return mk_val<value_undefined>();
}

value call_statement::execute_impl(context& ctx) {
    auto call_expr = cast_stmt<call_expression>(this->call);
    if (!call_expr) { throw std::runtime_error("Call statement requires a valid call expression"); }

    value callee_val = call_expr->callee->execute(ctx);
    if (!is_val<value_func>(callee_val)) {
        throw std::runtime_error("Callee is not a function: got " + callee_val->type());
    }
    auto* callee_func = cast_val<value_func>(callee_val);

    const func_handler func = [this, scope = ctx.current_scope()](const func_args& args) -> value {
        context block_ctx(args.ctx, scope);

        bind_parameters("caller", this->caller_args, args, block_ctx);

        return render_body(this->body, block_ctx);
    };

    context call_ctx(ctx);
    call_ctx.set_val("caller", mk_val<value_func>("caller", func));

    func_args args(call_ctx);

    for (const auto& arg_expr : call_expr->args) {
        auto arg_val = arg_expr->execute(ctx);
        args.push_back(arg_val);
    }

    return callee_func->invoke(args);
}

member_expression::member_expression(statement_ptr&& object, statement_ptr&& property,
                                     bool computed)
    : object(std::move(object)), property(std::move(property)), computed(computed) {
    chk_type<expression>(this->object);
    chk_type<expression>(this->property);
    if (computed) {
        slice_property = cast_stmt<slice_expression>(this->property);
    } else if ((static_index = cast_stmt<integer_literal>(this->property)) == nullptr) {
        if ((static_name = cast_stmt<identifier>(this->property)) != nullptr) {
            static_key = mk_val<value_string>(static_name->val);
        }
    }
}

// The builtin `name` of `input`, if it has one.
static const func_handler* find_builtin(const value& input, const std::string& name) {
    const auto& builtins = input->get_builtins();
    const auto it        = builtins.find(name);
    return it == builtins.end() ? nullptr : &it->second;
}

value member_expression::execute_impl(context& ctx) {
    value object = this->object->execute(ctx);

    value computed_property;
    const value* property = &computed_property;
    if (this->computed) {
        // syntax: obj[expr]

        if (slice_property) {
            const auto* s  = slice_property;
            value step_val = s->step_expr ? s->step_expr->execute(ctx) : mk_val<value_int>(1);
            value start_val =
                s->start_expr ? s->start_expr->execute(ctx) : mk_val<value_undefined>();
            value stop_val = s->stop_expr ? s->stop_expr->execute(ctx) : mk_val<value_undefined>();

            // translate to function call: obj.slice(start, stop, step)
            auto slice_func = try_builtin_func(ctx, "slice", object);
            func_args args(ctx);
            args.push_back(start_val);
            args.push_back(stop_val);
            args.push_back(step_val);
            return slice_func->invoke(args);
        } else {
            computed_property = this->property->execute(ctx);
        }
    } else if (static_index) {
        // syntax: obj.index
        computed_property = mk_val<value_int>(static_index->val);
        if (computed_property->as_int() < 0) {
            throw std::runtime_error("Static member property cannot be negative");
        }
    } else {
        // syntax: obj.prop
        if (!static_name) {
            throw std::runtime_error("Static member property must be an identifier");
        }
        const std::string& prop = static_name->val;

        // behavior of jinja2: obj having prop as a built-in function AND 'prop', as an object key,
        // then obj.prop returns the built-in function, not the property value.
        // while obj['prop'] returns the property value.
        // example: {"obj": {"items": 123}} -> obj.items is the built-in function, obj['items'] is
        // 123

        if (const func_handler* builtin = find_builtin(object, prop)) {
            return mk_val<value_func>(prop, *builtin, object);
        }
        // else, fallthrough to normal property access below
        property = &static_key;
    }

    if ((*property)->is_undefined()) { return mk_val<value_undefined>("object_property"); }

    ensure_key_type_allowed(*property);

    if (is_val<value_undefined>(object)) {
        return mk_val<value_undefined>("object_property");

    } else if (is_val<value_object>(object)) {
        value missing;
        const value& found = object->at(*property, missing);
        if (&found != &missing && !is_val<value_undefined>(found)) return found;
        return try_builtin_func(ctx, (*property)->as_string().str(), object, true);

    } else if (is_val<value_array>(object) || is_val<value_string>(object)) {
        if (is_val<value_int>(*property)) {
            int64_t index = (*property)->as_int();
            if (is_val<value_array>(object)) {
                auto& arr = object->as_array();
                if (index < 0) { index += static_cast<int64_t>(arr.size()); }
                if (index >= 0 && index < static_cast<int64_t>(arr.size())) { return arr[index]; }
            } else { // value_string
                const auto str  = object->as_string();
                const auto size = static_cast<int64_t>(str.length());
                if (index < 0) index += size;
                if (index >= 0 && index < size) {
                    return mk_val<value_string>(str.slice(index, index + 1));
                }
            }
            return mk_val<value_undefined>("object_property");

        } else if (is_val<value_string>(*property)) {
            return try_builtin_func(ctx, (*property)->as_string().str(), object, true);

        } else {
            throw std::runtime_error("Cannot access property with non-string/non-number: got " +
                                     (*property)->type());
        }
    } else {
        if (!is_val<value_string>(*property)) {
            throw std::runtime_error("Cannot access property with non-string: got " +
                                     (*property)->type());
        }
        return try_builtin_func(ctx, (*property)->as_string().str(), object, true);
    }
}

value call_expression::execute_impl(context& ctx) {
    // gather arguments
    func_args args(ctx);
    for (auto& arg_stmt : this->args) {
        auto arg_val = arg_stmt->execute(ctx);
        args.push_back(arg_val);
    }
    // execute callee
    value callee_val = callee->execute(ctx);
    if (!is_val<value_func>(callee_val)) {
        throw std::runtime_error("Callee is not a function: got " + callee_val->type());
    }
    auto* callee_func = cast_val<value_func>(callee_val);
    return callee_func->invoke(args);
}

value keyword_argument_expression::execute_impl(context& ctx) {
    if (!is_stmt<identifier>(key)) {
        throw std::runtime_error("Keyword argument key must be identifiers");
    }

    std::string k = cast_stmt<identifier>(key)->val;

    value v = val->execute(ctx);

    return mk_val<value_kwarg>(k, v);
}

string runtime::render(const program& prog) {
    string out;
    for (const auto& stmt : prog.body) {
        // The parser admits loop control only inside a loop.
        if (stmt->render(ctx, out) != flow::next) {
            throw std::runtime_error("loop control outside a loop");
        }
    }
    return out;
}

std::string runtime::debug_dump_program(const program& prog, const std::string& src) {
    std::ostringstream oss;
    size_t lvl = 0;
    context ctx(src);

    auto indent = [](size_t lvl) -> std::string { return std::string(lvl * 2, ' '); };

    ctx.state().visitor = [&](bool is_leaf, statement* node, std::vector<visitor_pair> children) {
        oss << indent(lvl) << node->type() << ":\n";
        lvl++;
        if (is_leaf) {
            const auto& pos = node->pos;
            oss << indent(lvl) << "(leaf) at " << get_line_col(src, pos) << " in source:\n";
            std::string snippet = peak_source(src, pos);
            string_replace_all(snippet, "\n", "\n" + indent(lvl));
            oss << indent(lvl) << snippet << "\n";
        } else {
            for (auto& [label, children_vec] : children) {
                oss << indent(lvl) << label << ":\n";
                lvl++;
                if (children_vec.empty()) {
                    oss << indent(lvl) << "<empty>\n\n";
                } else {
                    for (auto* child : children_vec) {
                        if (!child) { continue; }
                        child->visit(ctx);
                    }
                }
                lvl--;
            }
        }
        lvl--;
    };

    for (const auto& stmt : prog.body) { stmt->visit(ctx); }

    return oss.str();
}

} // namespace jinja
