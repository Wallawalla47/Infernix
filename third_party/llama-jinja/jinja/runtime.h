#pragma once

#include "lexer.h"
#include "value.h"

#include <cassert>
#include <cstdint>
#include <ctime>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace jinja {

struct statement;
using statement_ptr = std::unique_ptr<statement>;
using statements    = std::vector<statement_ptr>;

// Helpers for dynamic casting and type checking
template <typename T>
struct extract_pointee_unique {
    using type = T;
};

template <typename U>
struct extract_pointee_unique<std::unique_ptr<U>> {
    using type = U;
};

template <typename T>
bool is_stmt(const statement_ptr& ptr) {
    return dynamic_cast<const T*>(ptr.get()) != nullptr;
}

template <typename T>
T* cast_stmt(statement_ptr& ptr) {
    return dynamic_cast<T*>(ptr.get());
}

template <typename T>
const T* cast_stmt(const statement_ptr& ptr) {
    return dynamic_cast<const T*>(ptr.get());
}

// End Helpers



// for visiting AST nodes
// function signature: void(bool is_leaf, statement * node, pair of <label, children>)
using visitor_pair = std::pair<std::string, std::vector<statement*>>;
using visitor_fn   = std::function<void(bool, statement*, std::vector<visitor_pair>)>;

// One variable scope. A scope links to the scope it was entered from: a lookup walks outward to the
// innermost binding of a name, and an assignment always binds in the scope itself, so entering a
// scope copies nothing.
class scope_frame {
public:
    explicit scope_frame(std::shared_ptr<scope_frame> parent = nullptr) noexcept
        : parent_(std::move(parent)) {}

    const std::shared_ptr<scope_frame>& parent() const noexcept { return parent_; }

    value* find_local(std::string_view name) noexcept;
    void bind(std::string_view name, const value& val);

private:
    struct name_hash {
        using is_transparent = void;
        std::size_t operator()(std::string_view name) const noexcept {
            return std::hash<std::string_view>{}(name);
        }
    };
    using name_index = std::unordered_map<std::string, std::size_t, name_hash, std::equal_to<>>;
    // Loop and macro scopes hold a few names; a scope that outgrows a short scan is indexed.
    static constexpr std::size_t kLinearScan = 12;

    std::shared_ptr<scope_frame> parent_;
    std::vector<std::pair<std::string, value>> vars_;
    std::unique_ptr<name_index> index_;
};

// What every scope of one render shares.
struct render_state {
    std::shared_ptr<const std::string> src; // for error reporting
    std::time_t current_time = 0;           // for functions that need current time
    std::function<void()> checkpoint;
    visitor_fn visitor;
    // Macro definition scopes stay alive for the render. Callables refer back weakly, so a scope
    // that holds its own macro does not form an ownership cycle.
    std::vector<std::shared_ptr<scope_frame>> captured_scopes;
};

struct context {
    // A render's outermost scope. src is optional, used for error reporting.
    explicit context(std::string src = "");
    explicit context(std::shared_ptr<const std::string> src);
    // A new scope inside `parent`, such as a loop's.
    context(const context& parent);
    // A macro or caller body: a new scope inside the scope it was defined in, run for `caller`.
    context(const context& caller, const std::weak_ptr<scope_frame>& lexical_scope);
    context& operator=(const context&) = delete;
    ~context() = default;

    render_state& state() const noexcept { return *state_; }

    void checkpoint() const {
        if (state_->checkpoint) state_->checkpoint();
    }

    void visitor(bool is_leaf, statement* node, std::vector<visitor_pair> children) const {
        state_->visitor(is_leaf, node, std::move(children));
    }

    // The innermost binding of `name`, or null when no scope binds it.
    const value* find(std::string_view name) const noexcept;

    void set_val(std::string_view name, const value& val) { frame_->bind(name, val); }

    std::weak_ptr<scope_frame> capture_scope() {
        state_->captured_scopes.push_back(frame_);
        return frame_;
    }

    std::weak_ptr<scope_frame> current_scope() const { return frame_; }

private:
    std::unique_ptr<render_state> owned_state_; // the outermost scope only
    render_state* state_;
    std::shared_ptr<scope_frame> frame_;
};

// utils for visiting AST nodes
static std::vector<statement*> stmts_to_ptr(const statements& stmts) {
    std::vector<statement*> children;
    for (const auto& stmt : stmts) { children.push_back(stmt.get()); }
    return children;
}

// How a statement ends: normally, or at a {% break %} or {% continue %} of the enclosing loop.
// Loop control is a result, not an exception, so the output written before it is kept.
enum class flow : std::uint8_t { next, break_loop, continue_loop };

/**
 * Base class for all nodes in the AST.
 */
struct statement {
    size_t pos; // position in source, for debugging
    virtual ~statement() = default;

    virtual std::string type() const { return "Statement"; }

    virtual void visit(context& ctx) { ctx.visitor(true, this, {}); }

    // Expression position: execute_impl must be overridden by expressions and by statements
    // that produce a value.
    virtual value execute_impl(context&) { throw_exec_error(); }

    // execute is the public method to execute a statement with error handling
    value execute(context&);

    // Statement position: appends the statement's output to `out`. By default the output is the
    // statement's value.
    virtual flow render_impl(context& ctx, string& out);

    // render is the public method to render a statement with error handling
    flow render(context& ctx, string& out);

private:
    [[noreturn]] void throw_exec_error() const {
        throw std::runtime_error("cannot exec " + type());
    }
};

// Type Checking Utilities

template <typename T>
static void chk_type(const statement_ptr& ptr) {
    if (!ptr) return; // Allow null for optional fields
    assert(dynamic_cast<T*>(ptr.get()) != nullptr);
}

template <typename T, typename U>
static void chk_type(const statement_ptr& ptr) {
    if (!ptr) return;
    assert(dynamic_cast<T*>(ptr.get()) != nullptr || dynamic_cast<U*>(ptr.get()) != nullptr);
}

// Base Types

/**
 * Expressions will result in a value at runtime (unlike statements).
 */
struct expression : public statement {
    std::string type() const override { return "Expression"; }
};

struct output_statement : public statement {
    statement_ptr operand;

    explicit output_statement(statement_ptr&& operand) : operand(std::move(operand)) {}

    std::string type() const override { return "Output"; }

    flow render_impl(context& ctx, string& out) override;
};

// Statements

struct program : public statement {
    statements body;

    program() = default;

    explicit program(statements&& body) : body(std::move(body)) {}

    std::string type() const override { return "Program"; }

    [[noreturn]] value execute_impl(context&) override {
        throw std::runtime_error("Cannot execute program directly, use jinja::runtime instead");
    }
};

struct if_statement : public statement {
    statement_ptr test;
    statements body;
    statements alternate;

    if_statement(statement_ptr&& test, statements&& body, statements&& alternate)
        : test(std::move(test)), body(std::move(body)), alternate(std::move(alternate)) {
        chk_type<expression>(this->test);
    }

    std::string type() const override { return "If"; }

    flow render_impl(context& ctx, string& out) override;

    void visit(context& ctx) override {
        ctx.visitor(false, this,
                    {{"test", {test.get()}},
                     {"body", stmts_to_ptr(body)},
                     {"alternate", stmts_to_ptr(alternate)}});
    }
};

struct identifier;
struct tuple_literal;
struct slice_expression;
struct integer_literal;

/**
 * Loop over each item in a sequence
 * https://jinja.palletsprojects.com/en/3.0.x/templates/#for
 */
struct for_statement : public statement {
    statement_ptr loopvar; // Identifier | TupleLiteral
    statement_ptr iterable;
    statements body;
    statements default_block; // if no iteration took place

    for_statement(statement_ptr&& loopvar, statement_ptr&& iterable, statements&& body,
                  statements&& default_block)
        : loopvar(std::move(loopvar)), iterable(std::move(iterable)), body(std::move(body)),
          default_block(std::move(default_block)) {
        chk_type<identifier, tuple_literal>(this->loopvar);
        chk_type<expression>(this->iterable);
    }

    std::string type() const override { return "For"; }

    flow render_impl(context& ctx, string& out) override;

    void visit(context& ctx) override {
        ctx.visitor(false, this,
                    {{"loopvar", {loopvar.get()}},
                     {"iterable", {iterable.get()}},
                     {"body", stmts_to_ptr(body)},
                     {"default_block", stmts_to_ptr(default_block)}});
    }
};

struct break_statement : public statement {
    std::string type() const override { return "Break"; }

    flow render_impl(context&, string&) override { return flow::break_loop; }
};

struct continue_statement : public statement {
    std::string type() const override { return "Continue"; }

    flow render_impl(context&, string&) override { return flow::continue_loop; }
};

// do nothing
struct noop_statement : public statement {
    std::string type() const override { return "Noop"; }

    value execute_impl(context&) override { return mk_val<value_undefined>(); }

    flow render_impl(context&, string&) override { return flow::next; }
};

struct set_statement : public statement {
    statement_ptr assignee;
    statement_ptr val;
    statements body;

    // {% set name = ... %}: the name, resolved once with the program.
    const identifier* assignee_name = nullptr;

    set_statement(statement_ptr&& assignee, statement_ptr&& value, statements&& body);

    std::string type() const override { return "Set"; }

    flow render_impl(context& ctx, string& out) override;

    void visit(context& ctx) override {
        ctx.visitor(
            false, this,
            {{"assignee", {assignee.get()}}, {"value", {val.get()}}, {"body", stmts_to_ptr(body)}});
    }
};

struct macro_statement : public statement {
    statement_ptr name;
    statements args;
    statements body;

    macro_statement(statement_ptr&& name, statements&& args, statements&& body)
        : name(std::move(name)), args(std::move(args)), body(std::move(body)) {
        chk_type<identifier>(this->name);
        for (const auto& arg : this->args) chk_type<expression>(arg);
    }

    std::string type() const override { return "Macro"; }

    value execute_impl(context& ctx) override;

    flow render_impl(context& ctx, string&) override {
        (void)execute_impl(ctx);
        return flow::next;
    }

    void visit(context& ctx) override {
        ctx.visitor(
            false, this,
            {{"name", {name.get()}}, {"args", stmts_to_ptr(args)}, {"body", stmts_to_ptr(body)}});
    }
};

struct comment_statement : public statement {
    std::string val;

    explicit comment_statement(const std::string& v) : val(v) {}

    std::string type() const override { return "Comment"; }

    value execute_impl(context&) override { return mk_val<value_undefined>(); }

    flow render_impl(context&, string&) override { return flow::next; }
};

// Expressions

// Represents an omitted expression in a computed member, e.g. `a[]`.
struct blank_expression : public expression {
    std::string type() const override { return "BlankExpression"; }

    value execute_impl(context&) override { return mk_val<value_undefined>(); }
};

struct member_expression : public expression {
    statement_ptr object;
    statement_ptr property;
    bool computed; // true if obj[expr] and false if obj.prop
    // The property's form, resolved once with the program: obj[a:b:c], obj.0, or obj.prop with
    // its key built once and only ever read.
    const slice_expression* slice_property = nullptr;
    const integer_literal* static_index    = nullptr;
    const identifier* static_name          = nullptr;
    value static_key;

    member_expression(statement_ptr&& object, statement_ptr&& property, bool computed);

    std::string type() const override { return "MemberExpression"; }

    value execute_impl(context& ctx) override;

    void visit(context& ctx) override {
        ctx.visitor(false, this, {{"object", {object.get()}}, {"property", {property.get()}}});
    }
};

struct call_expression : public expression {
    statement_ptr callee;
    statements args;

    call_expression(statement_ptr&& callee, statements&& args)
        : callee(std::move(callee)), args(std::move(args)) {
        chk_type<expression>(this->callee);
        for (const auto& arg : this->args) chk_type<expression>(arg);
    }

    std::string type() const override { return "CallExpression"; }

    value execute_impl(context& ctx) override;

    void visit(context& ctx) override {
        ctx.visitor(false, this, {{"callee", {callee.get()}}, {"args", stmts_to_ptr(args)}});
    }
};

/**
 * Represents a user-defined variable or symbol in the template.
 */
struct identifier : public expression {
    std::string val;

    explicit identifier(const std::string& val) : val(val) {}

    std::string type() const override { return "Identifier"; }

    value execute_impl(context& ctx) override;
};

// Literals

struct integer_literal : public expression {
    int64_t val;

    explicit integer_literal(int64_t val) : val(val) {}

    std::string type() const override { return "IntegerLiteral"; }

    value execute_impl(context&) override { return mk_val<value_int>(val); }
};

struct float_literal : public expression {
    double val;

    explicit float_literal(double val) : val(val) {}

    std::string type() const override { return "FloatLiteral"; }

    value execute_impl(context&) override { return mk_val<value_float>(val); }
};

struct string_literal : public expression {
    std::string val;

    explicit string_literal(const std::string& val) : val(val) {}

    std::string type() const override { return "StringLiteral"; }

    value execute_impl(context&) override { return mk_val<value_string>(val); }

    // Template text: ordinary bytes appended without an intermediate value.
    flow render_impl(context&, string& out) override {
        out.append_plain(val);
        return flow::next;
    }
};

struct array_literal : public expression {
    statements val;

    explicit array_literal(statements&& val) : val(std::move(val)) {
        for (const auto& item : this->val) chk_type<expression>(item);
    }

    std::string type() const override { return "ArrayLiteral"; }

    value execute_impl(context& ctx) override {
        auto arr = mk_val<value_array>();
        for (const auto& item_stmt : val) { arr->push_back(item_stmt->execute(ctx)); }
        return arr;
    }
};

struct tuple_literal : public expression {
    statements val;

    explicit tuple_literal(statements&& val) : val(std::move(val)) {
        for (const auto& item : this->val) chk_type<expression>(item);
    }

    std::string type() const override { return "TupleLiteral"; }

    value execute_impl(context& ctx) override {
        std::vector<value> items;
        items.reserve(val.size());
        for (const auto& item_stmt : val) { items.push_back(item_stmt->execute(ctx)); }
        return mk_val<value_tuple>(std::move(items));
    }
};

struct object_literal : public expression {
    std::vector<std::pair<statement_ptr, statement_ptr>> val;

    explicit object_literal(std::vector<std::pair<statement_ptr, statement_ptr>>&& val)
        : val(std::move(val)) {
        for (const auto& pair : this->val) {
            chk_type<expression>(pair.first);
            chk_type<expression>(pair.second);
        }
    }

    std::string type() const override { return "ObjectLiteral"; }

    value execute_impl(context& ctx) override;
};

// Complex Expressions

/**
 * An operation with two sides, separated by an operator.
 * Note: Either side can be a Complex Expression, with order
 * of operations being determined by the operator.
 */
struct binary_expression : public expression {
    enum class operation : std::uint8_t {
        logical_and,
        logical_or,
        equal,
        not_equal,
        concat,
        in,
        not_in,
        add,
        subtract,
        multiply,
        divide,
        modulo,
        less,
        greater,
        greater_equal,
        less_equal,
        other,
    };

    token op;
    statement_ptr left;
    statement_ptr right;
    operation kind; // op, classified once with the program

    binary_expression(token op, statement_ptr&& left, statement_ptr&& right)
        : op(std::move(op)), left(std::move(left)), right(std::move(right)),
          kind(classify(this->op.value)) {
        chk_type<expression>(this->left);
        chk_type<expression>(this->right);
    }

    static operation classify(const std::string& name) {
        static const std::pair<const char*, operation> names[] = {
            {"and", operation::logical_and}, {"or", operation::logical_or},
            {"==", operation::equal},        {"!=", operation::not_equal},
            {"~", operation::concat},        {"in", operation::in},
            {"not in", operation::not_in},   {"+", operation::add},
            {"-", operation::subtract},      {"*", operation::multiply},
            {"/", operation::divide},        {"%", operation::modulo},
            {"<", operation::less},          {">", operation::greater},
            {">=", operation::greater_equal}, {"<=", operation::less_equal}};
        for (const auto& [text, which] : names) {
            if (name == text) return which;
        }
        return operation::other;
    }

    std::string type() const override { return "BinaryExpression"; }

    value execute_impl(context& ctx) override;

    void visit(context& ctx) override {
        ctx.visitor(false, this, {{"left", {left.get()}}, {"right", {right.get()}}});
    }
};

/**
 * An operation with two sides, separated by the | operator.
 * Operator precedence: https://github.com/pallets/jinja/issues/379#issuecomment-168076202
 */
struct filter_expression : public expression {
    // either an expression or a value is allowed
    statement_ptr operand;
    value_string val; // will be set by filter_statement

    statement_ptr filter;

    filter_expression(statement_ptr&& operand, statement_ptr&& filter)
        : operand(std::move(operand)), filter(std::move(filter)) {
        chk_type<expression>(this->operand);
        chk_type<identifier, call_expression>(this->filter);
    }

    filter_expression(value_string&& val, statement_ptr&& filter)
        : val(std::move(val)), filter(std::move(filter)) {
        chk_type<identifier, call_expression>(this->filter);
    }

    std::string type() const override { return "FilterExpression"; }

    value execute_impl(context& ctx) override;

    void visit(context& ctx) override {
        ctx.visitor(false, this, {{"operand", {operand.get()}}, {"filter", {filter.get()}}});
    }
};

struct filter_statement : public statement {
    statement_ptr filter;
    statements body;

    filter_statement(statement_ptr&& filter, statements&& body)
        : filter(std::move(filter)), body(std::move(body)) {
        chk_type<identifier, call_expression>(this->filter);
    }

    std::string type() const override { return "FilterStatement"; }

    flow render_impl(context& ctx, string& out) override;

    void visit(context& ctx) override {
        ctx.visitor(false, this, {{"filter", {filter.get()}}, {"body", stmts_to_ptr(body)}});
    }
};

/**
 * An operation which filters a sequence of objects by applying a test to each object,
 * and only selecting the objects with the test succeeding.
 *
 * It may also be used as a shortcut for a ternary operator.
 */
struct select_expression : public expression {
    statement_ptr lhs;
    statement_ptr test;

    select_expression(statement_ptr&& lhs, statement_ptr&& test)
        : lhs(std::move(lhs)), test(std::move(test)) {
        chk_type<expression>(this->lhs);
        chk_type<expression>(this->test);
    }

    std::string type() const override { return "SelectExpression"; }

    value execute_impl(context& ctx) override {
        auto predicate = test->execute_impl(ctx);
        if (!predicate->as_bool()) { return mk_val<value_undefined>(); }
        return lhs->execute_impl(ctx);
    }

    void visit(context& ctx) override {
        ctx.visitor(false, this, {{"lhs", {lhs.get()}}, {"test", {test.get()}}});
    }
};

/**
 * An operation with two sides, separated by the "is" operator.
 * NOTE: "value is something" translates to function call "test_is_something(value)"
 */
struct test_expression : public expression {
    statement_ptr operand;
    bool negate;
    statement_ptr test;
    // The test's name (`is name` or `is name(args)`) and builtin, resolved once with the program;
    // an unknown or malformed test fails when the expression runs.
    const identifier* test_name     = nullptr;
    const call_expression* test_call = nullptr;
    const func_handler* handler     = nullptr;

    test_expression(statement_ptr&& operand, bool negate, statement_ptr&& test);

    std::string type() const override { return "TestExpression"; }

    value execute_impl(context& ctx) override;

    void visit(context& ctx) override {
        ctx.visitor(false, this, {{"operand", {operand.get()}}, {"test", {test.get()}}});
    }
};

/**
 * An operation with one side (operator on the left).
 */
struct unary_expression : public expression {
    token op;
    statement_ptr argument;

    unary_expression(token op, statement_ptr&& argument)
        : op(std::move(op)), argument(std::move(argument)) {
        chk_type<expression>(this->argument);
    }

    std::string type() const override { return "UnaryExpression"; }

    value execute_impl(context& ctx) override;

    void visit(context& ctx) override {
        ctx.visitor(false, this, {{"argument", {argument.get()}}});
    }
};

struct slice_expression : public expression {
    statement_ptr start_expr;
    statement_ptr stop_expr;
    statement_ptr step_expr;

    slice_expression(statement_ptr&& start_expr, statement_ptr&& stop_expr,
                     statement_ptr&& step_expr)
        : start_expr(std::move(start_expr)), stop_expr(std::move(stop_expr)),
          step_expr(std::move(step_expr)) {
        chk_type<expression>(this->start_expr);
        chk_type<expression>(this->stop_expr);
        chk_type<expression>(this->step_expr);
    }

    std::string type() const override { return "SliceExpression"; }

    [[noreturn]] value execute_impl(context&) override {
        throw std::runtime_error("must be handled by MemberExpression");
    }

    void visit(context& ctx) override {
        ctx.visitor(false, this,
                    {{"start_expr", {start_expr.get()}},
                     {"stop_expr", {stop_expr.get()}},
                     {"step_expr", {step_expr.get()}}});
    }
};

struct keyword_argument_expression : public expression {
    statement_ptr key;
    statement_ptr val;

    keyword_argument_expression(statement_ptr&& key, statement_ptr&& val)
        : key(std::move(key)), val(std::move(val)) {
        chk_type<identifier>(this->key);
        chk_type<expression>(this->val);
    }

    std::string type() const override { return "KeywordArgumentExpression"; }

    value execute_impl(context& ctx) override;

    void visit(context& ctx) override {
        ctx.visitor(false, this, {{"key", {key.get()}}, {"val", {val.get()}}});
    }
};

struct spread_expression : public expression {
    statement_ptr argument;

    explicit spread_expression(statement_ptr&& argument) : argument(std::move(argument)) {
        chk_type<expression>(this->argument);
    }

    std::string type() const override { return "SpreadExpression"; }

    void visit(context& ctx) override {
        ctx.visitor(false, this, {{"argument", {argument.get()}}});
    }
};

struct call_statement : public statement {
    statement_ptr call;
    statements caller_args;
    statements body;

    call_statement(statement_ptr&& call, statements&& caller_args, statements&& body)
        : call(std::move(call)), caller_args(std::move(caller_args)), body(std::move(body)) {
        chk_type<call_expression>(this->call);
        for (const auto& arg : this->caller_args) chk_type<expression>(arg);
    }

    std::string type() const override { return "CallStatement"; }

    value execute_impl(context& ctx) override;

    void visit(context& ctx) override {
        ctx.visitor(false, this,
                    {{"call", {call.get()}},
                     {"caller_args", stmts_to_ptr(caller_args)},
                     {"body", stmts_to_ptr(body)}});
    }
};

struct ternary_expression : public expression {
    statement_ptr condition;
    statement_ptr true_expr;
    statement_ptr false_expr;

    ternary_expression(statement_ptr&& condition, statement_ptr&& true_expr,
                       statement_ptr&& false_expr)
        : condition(std::move(condition)), true_expr(std::move(true_expr)),
          false_expr(std::move(false_expr)) {
        chk_type<expression>(this->condition);
        chk_type<expression>(this->true_expr);
        chk_type<expression>(this->false_expr);
    }

    std::string type() const override { return "Ternary"; }

    value execute_impl(context& ctx) override {
        value cond_val = condition->execute(ctx);
        if (cond_val->as_bool()) {
            return true_expr->execute(ctx);
        } else {
            return false_expr->execute(ctx);
        }
    }

    void visit(context& ctx) override {
        ctx.visitor(false, this,
                    {{"condition", {condition.get()}},
                     {"true_expr", {true_expr.get()}},
                     {"false_expr", {false_expr.get()}}});
    }
};

struct raised_exception : public std::exception {
    std::string message;

    raised_exception(const std::string& msg) : message(msg) {}

    const char* what() const noexcept override { return message.c_str(); }
};

// Used to rethrow exceptions with modified messages
struct rethrown_exception : public std::exception {
    std::string message;

    rethrown_exception(const std::string& msg) : message(msg) {}

    const char* what() const noexcept override { return message.c_str(); }
};

//////////////////////

struct runtime {
    context& ctx;

    explicit runtime(context& ctx) : ctx(ctx) {}

    // The program's output, as one string with its parts' origins.
    string render(const program& prog);

    static std::string debug_dump_program(const program& prog, const std::string& src);
};

} // namespace jinja
