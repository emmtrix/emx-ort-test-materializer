#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct LiteralValue {
  enum class Kind {
    kNull,
    kBool,
    kInt,
    kDouble,
    kString,
    kArray,
  };

  Kind kind = Kind::kNull;
  bool bool_value = false;
  int64_t int_value = 0;
  double double_value = 0.0;
  std::string string_value;
  std::vector<LiteralValue> array_value;

  static LiteralValue MakeBool(bool value) {
    LiteralValue literal;
    literal.kind = Kind::kBool;
    literal.bool_value = value;
    return literal;
  }

  static LiteralValue MakeInt(int64_t value) {
    LiteralValue literal;
    literal.kind = Kind::kInt;
    literal.int_value = value;
    return literal;
  }

  static LiteralValue MakeDouble(double value) {
    LiteralValue literal;
    literal.kind = Kind::kDouble;
    literal.double_value = value;
    return literal;
  }

  static LiteralValue MakeString(std::string value) {
    LiteralValue literal;
    literal.kind = Kind::kString;
    literal.string_value = std::move(value);
    return literal;
  }

  static LiteralValue MakeArray(std::vector<LiteralValue> value) {
    LiteralValue literal;
    literal.kind = Kind::kArray;
    literal.array_value = std::move(value);
    return literal;
  }
};

struct TensorCall {
  std::string name;
  std::string cpp_type;
  std::string shape_expr;
  std::string values_expr;
  std::optional<LiteralValue> shape;
  std::optional<LiteralValue> values;
  bool is_initializer = false;
  std::vector<std::string> warnings;
};

struct AttributeCall {
  std::string name;
  std::string value_expr;
  std::optional<LiteralValue> value;
  std::vector<std::string> warnings;
};

struct TestRecord {
  std::string source_file;
  std::string macro_name;
  std::string test_suite;
  std::string test_name;
  std::string tester_var;
  std::string op_name;
  std::string opset_expr;
  std::optional<int64_t> opset;
  std::string domain_expr;
  bool saw_run = false;
  std::vector<TensorCall> inputs;
  std::vector<TensorCall> outputs;
  std::vector<AttributeCall> attributes;
  std::vector<std::string> warnings;
};

struct ExtractionResult {
  std::vector<TestRecord> records;
  size_t files_scanned = 0;
};

struct ParseScope {
  std::unordered_map<std::string, LiteralValue> variables;
};

std::string NormalizePath(std::string value) {
  std::replace(value.begin(), value.end(), '\\', '/');
  return value;
}

std::string PathRelativeToOnnxruntimeOrg(const fs::path& path) {
  const std::string normalized = NormalizePath(path.string());
  const std::string marker = "onnxruntime-org/";
  const size_t marker_pos = normalized.find(marker);

  if (marker_pos != std::string::npos) {
    return normalized.substr(marker_pos + marker.size());
  }

  if (normalized == "onnxruntime-org") {
    return ".";
  }

  return normalized;
}

std::string Trim(std::string_view value) {
  size_t begin = 0;
  while (begin < value.size() && std::isspace(static_cast<unsigned char>(value[begin])) != 0) {
    ++begin;
  }

  size_t end = value.size();
  while (end > begin && std::isspace(static_cast<unsigned char>(value[end - 1])) != 0) {
    --end;
  }

  return std::string(value.substr(begin, end - begin));
}

bool IsIdentifierChar(char c) {
  return std::isalnum(static_cast<unsigned char>(c)) != 0 || c == '_';
}

bool StartsWithWord(std::string_view text, size_t offset, std::string_view word) {
  if (offset + word.size() > text.size()) {
    return false;
  }

  if (text.substr(offset, word.size()) != word) {
    return false;
  }

  if (offset > 0 && IsIdentifierChar(text[offset - 1])) {
    return false;
  }

  if (offset + word.size() < text.size() && IsIdentifierChar(text[offset + word.size()])) {
    return false;
  }

  return true;
}

std::string StripComments(std::string_view input) {
  std::string output;
  output.reserve(input.size());

  bool in_string = false;
  bool in_char = false;
  bool escaped = false;

  for (size_t i = 0; i < input.size(); ++i) {
    const char c = input[i];
    const char next = (i + 1 < input.size()) ? input[i + 1] : '\0';

    if (in_string) {
      output.push_back(c);
      if (escaped) {
        escaped = false;
      } else if (c == '\\') {
        escaped = true;
      } else if (c == '"') {
        in_string = false;
      }
      continue;
    }

    if (in_char) {
      output.push_back(c);
      if (escaped) {
        escaped = false;
      } else if (c == '\\') {
        escaped = true;
      } else if (c == '\'') {
        in_char = false;
      }
      continue;
    }

    if (c == '"') {
      in_string = true;
      output.push_back(c);
      continue;
    }

    if (c == '\'') {
      in_char = true;
      output.push_back(c);
      continue;
    }

    if (c == '/' && next == '/') {
      while (i < input.size() && input[i] != '\n') {
        ++i;
      }
      if (i < input.size()) {
        output.push_back('\n');
      }
      continue;
    }

    if (c == '/' && next == '*') {
      i += 2;
      while (i < input.size()) {
        if (input[i] == '*' && i + 1 < input.size() && input[i + 1] == '/') {
          ++i;
          break;
        }
        if (input[i] == '\n') {
          output.push_back('\n');
        }
        ++i;
      }
      continue;
    }

    output.push_back(c);
  }

  return output;
}

bool IsEscaped(std::string_view text, size_t pos) {
  size_t slash_count = 0;
  while (pos > 0 && text[pos - 1] == '\\') {
    --pos;
    ++slash_count;
  }
  return (slash_count % 2) == 1;
}

size_t FindMatching(std::string_view text, size_t open_pos, char open_char, char close_char) {
  if (open_pos >= text.size() || text[open_pos] != open_char) {
    return std::string::npos;
  }

  size_t depth = 0;
  bool in_string = false;
  bool in_char = false;

  for (size_t i = open_pos; i < text.size(); ++i) {
    const char c = text[i];
    if (in_string) {
      if (c == '"' && !IsEscaped(text, i)) {
        in_string = false;
      }
      continue;
    }

    if (in_char) {
      if (c == '\'' && !IsEscaped(text, i)) {
        in_char = false;
      }
      continue;
    }

    if (c == '"') {
      in_string = true;
      continue;
    }

    if (c == '\'') {
      in_char = true;
      continue;
    }

    if (c == open_char) {
      ++depth;
    } else if (c == close_char) {
      --depth;
      if (depth == 0) {
        return i;
      }
    }
  }

  return std::string::npos;
}

int UpdateAngleDepth(std::string_view text, size_t index, int current_depth) {
  const char c = text[index];
  if (c == '<') {
    char prev = '\0';
    size_t prev_index = index;
    while (prev_index > 0) {
      --prev_index;
      if (std::isspace(static_cast<unsigned char>(text[prev_index])) == 0) {
        prev = text[prev_index];
        break;
      }
    }

    if (std::isalnum(static_cast<unsigned char>(prev)) != 0 || prev == '_' || prev == ':' || prev == '>') {
      return current_depth + 1;
    }
  } else if (c == '>' && current_depth > 0) {
    return current_depth - 1;
  }

  return current_depth;
}

std::vector<std::string> SplitTopLevel(std::string_view text, char delimiter) {
  std::vector<std::string> parts;
  size_t part_start = 0;
  int paren_depth = 0;
  int brace_depth = 0;
  int bracket_depth = 0;
  int angle_depth = 0;
  bool in_string = false;
  bool in_char = false;

  for (size_t i = 0; i < text.size(); ++i) {
    const char c = text[i];

    if (in_string) {
      if (c == '"' && !IsEscaped(text, i)) {
        in_string = false;
      }
      continue;
    }

    if (in_char) {
      if (c == '\'' && !IsEscaped(text, i)) {
        in_char = false;
      }
      continue;
    }

    if (c == '"') {
      in_string = true;
      continue;
    }

    if (c == '\'') {
      in_char = true;
      continue;
    }

    switch (c) {
      case '(':
        ++paren_depth;
        break;
      case ')':
        --paren_depth;
        break;
      case '{':
        ++brace_depth;
        break;
      case '}':
        --brace_depth;
        break;
      case '[':
        ++bracket_depth;
        break;
      case ']':
        --bracket_depth;
        break;
      default:
        angle_depth = UpdateAngleDepth(text, i, angle_depth);
        break;
    }

    if (c == delimiter && paren_depth == 0 && brace_depth == 0 && bracket_depth == 0 && angle_depth == 0) {
      parts.push_back(Trim(text.substr(part_start, i - part_start)));
      part_start = i + 1;
    }
  }

  parts.push_back(Trim(text.substr(part_start)));
  return parts;
}

std::vector<std::string> SplitStatements(std::string_view body) {
  std::vector<std::string> statements;
  size_t start = 0;
  int paren_depth = 0;
  int brace_depth = 0;
  int bracket_depth = 0;
  int angle_depth = 0;
  bool in_string = false;
  bool in_char = false;

  for (size_t i = 0; i < body.size(); ++i) {
    const char c = body[i];

    if (in_string) {
      if (c == '"' && !IsEscaped(body, i)) {
        in_string = false;
      }
      continue;
    }

    if (in_char) {
      if (c == '\'' && !IsEscaped(body, i)) {
        in_char = false;
      }
      continue;
    }

    if (c == '"') {
      in_string = true;
      continue;
    }

    if (c == '\'') {
      in_char = true;
      continue;
    }

    switch (c) {
      case '(':
        ++paren_depth;
        break;
      case ')':
        --paren_depth;
        break;
      case '{':
        ++brace_depth;
        break;
      case '}':
        --brace_depth;
        if (brace_depth == 0 && paren_depth == 0 && bracket_depth == 0) {
          const std::string statement = Trim(body.substr(start, i - start + 1));
          if (!statement.empty()) {
            statements.push_back(statement);
          }
          start = i + 1;
        }
        break;
      case '[':
        ++bracket_depth;
        break;
      case ']':
        --bracket_depth;
        break;
      default:
        angle_depth = UpdateAngleDepth(body, i, angle_depth);
        break;
    }

    if (c == ';' && paren_depth == 0 && brace_depth == 0 && bracket_depth == 0 && angle_depth == 0) {
      const std::string statement = Trim(body.substr(start, i - start));
      if (!statement.empty()) {
        statements.push_back(statement);
      }
      start = i + 1;
    }
  }

  const std::string tail = Trim(body.substr(start));
  if (!tail.empty()) {
    statements.push_back(tail);
  }

  return statements;
}

std::string Unquote(std::string_view text) {
  if (text.size() < 2) {
    return std::string(text);
  }

  const char quote = text.front();
  if ((quote != '"' && quote != '\'') || text.back() != quote) {
    return std::string(text);
  }

  std::string unquoted;
  unquoted.reserve(text.size() - 2);
  bool escaped = false;

  for (size_t i = 1; i + 1 < text.size(); ++i) {
    const char c = text[i];
    if (escaped) {
      switch (c) {
        case 'n':
          unquoted.push_back('\n');
          break;
        case 't':
          unquoted.push_back('\t');
          break;
        case 'r':
          unquoted.push_back('\r');
          break;
        default:
          unquoted.push_back(c);
          break;
      }
      escaped = false;
      continue;
    }

    if (c == '\\') {
      escaped = true;
      continue;
    }

    unquoted.push_back(c);
  }

  return unquoted;
}

std::optional<int64_t> ParseIntegerLiteral(std::string value) {
  value = Trim(value);
  if (value.empty()) {
    return std::nullopt;
  }

  while (!value.empty()) {
    const char last = value.back();
    if (last == 'u' || last == 'U' || last == 'l' || last == 'L') {
      value.pop_back();
    } else {
      break;
    }
  }

  if (value.empty()) {
    return std::nullopt;
  }

  char* end = nullptr;
  const long long parsed = std::strtoll(value.c_str(), &end, 0);
  if (end == nullptr || *end != '\0') {
    return std::nullopt;
  }

  return static_cast<int64_t>(parsed);
}

std::optional<double> ParseDoubleLiteral(std::string value) {
  value = Trim(value);
  if (value.empty()) {
    return std::nullopt;
  }

  while (!value.empty()) {
    const char last = value.back();
    if (last == 'f' || last == 'F' || last == 'l' || last == 'L') {
      value.pop_back();
    } else {
      break;
    }
  }

  if (value.empty()) {
    return std::nullopt;
  }

  char* end = nullptr;
  const double parsed = std::strtod(value.c_str(), &end);
  if (end == nullptr || *end != '\0') {
    return std::nullopt;
  }

  return parsed;
}

std::optional<LiteralValue> ParseLiteral(std::string expr, const ParseScope& scope);

std::optional<LiteralValue> ParseArrayLiteral(std::string_view expr, const ParseScope& scope) {
  if (expr.size() < 2 || expr.front() != '{' || expr.back() != '}') {
    return std::nullopt;
  }

  const std::string inner = Trim(expr.substr(1, expr.size() - 2));
  if (inner.empty()) {
    return LiteralValue::MakeArray({});
  }

  std::vector<LiteralValue> items;
  for (const std::string& part : SplitTopLevel(inner, ',')) {
    if (part.empty()) {
      continue;
    }
    const auto literal = ParseLiteral(part, scope);
    if (!literal.has_value()) {
      return std::nullopt;
    }
    items.push_back(*literal);
  }

  return LiteralValue::MakeArray(std::move(items));
}

std::optional<LiteralValue> ParseLiteral(std::string expr, const ParseScope& scope) {
  expr = Trim(expr);
  if (expr.empty()) {
    return std::nullopt;
  }

  while (expr.size() >= 2 && expr.front() == '(' && expr.back() == ')') {
    const size_t close = FindMatching(expr, 0, '(', ')');
    if (close != expr.size() - 1) {
      break;
    }
    expr = Trim(expr.substr(1, expr.size() - 2));
  }

  const auto variable_it = scope.variables.find(expr);
  if (variable_it != scope.variables.end()) {
    return variable_it->second;
  }

  if (expr == "true") {
    return LiteralValue::MakeBool(true);
  }

  if (expr == "false") {
    return LiteralValue::MakeBool(false);
  }

  if (expr.size() >= 2 && ((expr.front() == '"' && expr.back() == '"') || (expr.front() == '\'' && expr.back() == '\''))) {
    return LiteralValue::MakeString(Unquote(expr));
  }

  if (expr.front() == '{' && expr.back() == '}') {
    return ParseArrayLiteral(expr, scope);
  }

  if (const auto integer = ParseIntegerLiteral(expr); integer.has_value()) {
    return LiteralValue::MakeInt(*integer);
  }

  const bool looks_like_double =
      expr.find('.') != std::string::npos ||
      expr.find('e') != std::string::npos ||
      expr.find('E') != std::string::npos ||
      (!expr.empty() && (expr.back() == 'f' || expr.back() == 'F'));

  if (looks_like_double) {
    if (const auto floating = ParseDoubleLiteral(expr); floating.has_value()) {
      return LiteralValue::MakeDouble(*floating);
    }
  }

  return std::nullopt;
}

std::optional<std::string> ExtractLastIdentifier(std::string_view text) {
  if (text.empty()) {
    return std::nullopt;
  }

  size_t end = text.size();
  while (end > 0 && std::isspace(static_cast<unsigned char>(text[end - 1])) != 0) {
    --end;
  }

  if (end == 0) {
    return std::nullopt;
  }

  size_t pos = end;
  while (pos > 0 && IsIdentifierChar(text[pos - 1])) {
    --pos;
  }

  if (pos == end) {
    return std::nullopt;
  }

  return std::string(text.substr(pos, end - pos));
}

bool ParseVariableDefinition(const std::string& statement, ParseScope& scope) {
  const size_t equal_pos = statement.find('=');
  if (equal_pos == std::string::npos) {
    return false;
  }

  const std::string lhs = Trim(statement.substr(0, equal_pos));
  const std::string rhs = Trim(statement.substr(equal_pos + 1));

  if (lhs.find("OpTester") != std::string::npos) {
    return false;
  }

  const auto name = ExtractLastIdentifier(lhs);
  if (!name.has_value()) {
    return false;
  }

  const auto literal = ParseLiteral(rhs, scope);
  if (!literal.has_value()) {
    return false;
  }

  scope.variables[*name] = *literal;
  return true;
}

std::optional<size_t> FindMethodOpenParen(std::string_view statement, size_t start_pos) {
  bool in_string = false;
  bool in_char = false;
  int angle_depth = 0;

  for (size_t i = start_pos; i < statement.size(); ++i) {
    const char c = statement[i];

    if (in_string) {
      if (c == '"' && !IsEscaped(statement, i)) {
        in_string = false;
      }
      continue;
    }

    if (in_char) {
      if (c == '\'' && !IsEscaped(statement, i)) {
        in_char = false;
      }
      continue;
    }

    if (c == '"') {
      in_string = true;
      continue;
    }

    if (c == '\'') {
      in_char = true;
      continue;
    }

    angle_depth = UpdateAngleDepth(statement, i, angle_depth);
    if (c == '(' && angle_depth == 0) {
      return i;
    }
  }

  return std::nullopt;
}

bool ParseTesterConstructor(const std::string& statement, TestRecord& record) {
  const std::string trimmed = Trim(statement);
  if (trimmed.find("OpTester") == std::string::npos && trimmed.find("CompareOpTester") == std::string::npos) {
    return false;
  }

  size_t type_pos = trimmed.find("CompareOpTester");
  std::string type_name = "CompareOpTester";
  if (type_pos == std::string::npos) {
    type_pos = trimmed.find("OpTester");
    type_name = "OpTester";
  }

  if (type_pos == std::string::npos) {
    return false;
  }

  const size_t name_start = type_pos + type_name.size();
  std::string rest = Trim(trimmed.substr(name_start));
  if (rest.empty()) {
    return false;
  }

  size_t var_end = 0;
  while (var_end < rest.size() && IsIdentifierChar(rest[var_end])) {
    ++var_end;
  }

  if (var_end == 0) {
    return false;
  }

  record.tester_var = rest.substr(0, var_end);
  rest = Trim(rest.substr(var_end));
  if (rest.empty()) {
    return false;
  }

  const char open_char = rest.front();
  const char close_char = (open_char == '{') ? '}' : ')';
  if (open_char != '(' && open_char != '{') {
    return false;
  }

  const size_t close_pos = FindMatching(rest, 0, open_char, close_char);
  if (close_pos == std::string::npos) {
    return false;
  }

  const std::vector<std::string> args = SplitTopLevel(rest.substr(1, close_pos - 1), ',');
  if (args.empty()) {
    return false;
  }

  record.op_name = Unquote(args[0]);
  if (args.size() >= 2) {
    record.opset_expr = args[1];
    if (const auto opset_literal = ParseIntegerLiteral(args[1]); opset_literal.has_value()) {
      record.opset = *opset_literal;
    }
  }

  if (args.size() >= 3) {
    record.domain_expr = args[2];
  }

  return true;
}

bool ParseTensorCall(
    const std::string& method_name,
    const std::string& template_type,
    const std::vector<std::string>& args,
    const ParseScope& scope,
    TensorCall& out_call) {
  if (args.size() < 3) {
    return false;
  }

  out_call.cpp_type = template_type;
  out_call.name = Unquote(args[0]);
  out_call.shape_expr = args[1];
  out_call.values_expr = args[2];

  out_call.shape = ParseLiteral(args[1], scope);
  if (!out_call.shape.has_value()) {
    out_call.warnings.push_back("Unable to parse tensor shape literal: " + args[1]);
  }

  out_call.values = ParseLiteral(args[2], scope);
  if (!out_call.values.has_value()) {
    out_call.warnings.push_back("Unable to parse tensor values literal: " + args[2]);
  }

  if (method_name == "AddInput" && args.size() >= 4) {
    const auto initializer_literal = ParseLiteral(args[3], scope);
    if (initializer_literal.has_value() && initializer_literal->kind == LiteralValue::Kind::kBool) {
      out_call.is_initializer = initializer_literal->bool_value;
    }
  }

  return true;
}

bool ParseAttributeCall(
    const std::vector<std::string>& args,
    const ParseScope& scope,
    AttributeCall& out_attribute) {
  if (args.size() < 2) {
    return false;
  }

  out_attribute.name = Unquote(args[0]);
  out_attribute.value_expr = args[1];
  out_attribute.value = ParseLiteral(args[1], scope);
  if (!out_attribute.value.has_value()) {
    out_attribute.warnings.push_back("Unable to parse attribute literal: " + args[1]);
  }

  return true;
}

bool ParseMethodCall(const std::string& statement, TestRecord& record, ParseScope& scope) {
  if (record.tester_var.empty()) {
    return false;
  }

  const std::string method_prefix = record.tester_var + ".";
  const size_t prefix_pos = statement.find(method_prefix);
  if (prefix_pos == std::string::npos) {
    return false;
  }

  size_t method_start = prefix_pos + method_prefix.size();
  size_t method_end = method_start;
  while (method_end < statement.size() && IsIdentifierChar(statement[method_end])) {
    ++method_end;
  }

  if (method_end == method_start) {
    return false;
  }

  const std::string method_name = statement.substr(method_start, method_end - method_start);
  const auto open_paren = FindMethodOpenParen(statement, method_end);
  if (!open_paren.has_value()) {
    return false;
  }

  const size_t close_paren = FindMatching(statement, *open_paren, '(', ')');
  if (close_paren == std::string::npos) {
    return false;
  }

  std::string template_type;
  if (method_end < *open_paren) {
    const std::string between = Trim(statement.substr(method_end, *open_paren - method_end));
    if (!between.empty() && between.front() == '<' && between.back() == '>') {
      template_type = Trim(between.substr(1, between.size() - 2));
    }
  }

  const std::vector<std::string> args = SplitTopLevel(statement.substr(*open_paren + 1, close_paren - *open_paren - 1), ',');

  if (method_name == "AddInput") {
    TensorCall call;
    if (!ParseTensorCall(method_name, template_type, args, scope, call)) {
      record.warnings.push_back("Unable to parse AddInput call: " + statement);
      return true;
    }
    record.inputs.push_back(std::move(call));
    return true;
  }

  if (method_name == "AddOutput") {
    TensorCall call;
    if (!ParseTensorCall(method_name, template_type, args, scope, call)) {
      record.warnings.push_back("Unable to parse AddOutput call: " + statement);
      return true;
    }
    record.outputs.push_back(std::move(call));
    return true;
  }

  if (method_name == "AddAttribute") {
    AttributeCall call;
    if (!ParseAttributeCall(args, scope, call)) {
      record.warnings.push_back("Unable to parse AddAttribute call: " + statement);
      return true;
    }
    record.attributes.push_back(std::move(call));
    return true;
  }

  if (method_name == "Run") {
    record.saw_run = true;
    return true;
  }

  return false;
}

std::vector<TestRecord> ParseTestBody(
    const fs::path& source_path,
    std::string_view macro_name,
    std::string_view suite_name,
    std::string_view test_name,
    std::string_view body) {
  ParseScope scope;
  std::vector<TestRecord> records;

  for (const std::string& statement : SplitStatements(body)) {
    if (statement.empty()) {
      continue;
    }

    if (ParseVariableDefinition(statement, scope)) {
      continue;
    }

    TestRecord fresh_record;
    fresh_record.source_file = PathRelativeToOnnxruntimeOrg(source_path);
    fresh_record.macro_name = std::string(macro_name);
    fresh_record.test_suite = Unquote(suite_name);
    fresh_record.test_name = Unquote(test_name);

    if (ParseTesterConstructor(statement, fresh_record)) {
      records.push_back(std::move(fresh_record));
      continue;
    }

    if (!records.empty() && ParseMethodCall(statement, records.back(), scope)) {
      continue;
    }
  }

  return records;
}

std::vector<TestRecord> ParseSourceFile(const fs::path& source_path) {
  std::ifstream input(source_path);
  if (!input) {
    throw std::runtime_error("Unable to open source file: " + source_path.string());
  }

  std::ostringstream buffer;
  buffer << input.rdbuf();

  const std::string content = StripComments(buffer.str());
  std::vector<TestRecord> records;

  const std::vector<std::string> macros = {"TEST", "TEST_F", "TEST_P"};
  size_t cursor = 0;
  while (cursor < content.size()) {
    size_t best_pos = std::string::npos;
    std::string matched_macro;

    for (const std::string& macro : macros) {
      const size_t pos = content.find(macro, cursor);
      if (pos != std::string::npos && StartsWithWord(content, pos, macro) &&
          (best_pos == std::string::npos || pos < best_pos)) {
        best_pos = pos;
        matched_macro = macro;
      }
    }

    if (best_pos == std::string::npos) {
      break;
    }

    size_t open_paren = best_pos + matched_macro.size();
    while (open_paren < content.size() && std::isspace(static_cast<unsigned char>(content[open_paren])) != 0) {
      ++open_paren;
    }

    if (open_paren >= content.size() || content[open_paren] != '(') {
      cursor = best_pos + matched_macro.size();
      continue;
    }

    const size_t close_paren = FindMatching(content, open_paren, '(', ')');
    if (close_paren == std::string::npos) {
      break;
    }

    const std::vector<std::string> test_args = SplitTopLevel(content.substr(open_paren + 1, close_paren - open_paren - 1), ',');
    if (test_args.size() < 2) {
      cursor = close_paren + 1;
      continue;
    }

    size_t body_open = close_paren + 1;
    while (body_open < content.size() && std::isspace(static_cast<unsigned char>(content[body_open])) != 0) {
      ++body_open;
    }

    if (body_open >= content.size() || content[body_open] != '{') {
      cursor = close_paren + 1;
      continue;
    }

    const size_t body_close = FindMatching(content, body_open, '{', '}');
    if (body_close == std::string::npos) {
      break;
    }

    const std::string_view body(content.data() + body_open + 1, body_close - body_open - 1);
    std::vector<TestRecord> parsed = ParseTestBody(source_path, matched_macro, test_args[0], test_args[1], body);
    for (TestRecord& record : parsed) {
      records.push_back(std::move(record));
    }

    cursor = body_close + 1;
  }

  return records;
}

std::string JsonEscape(std::string_view value) {
  std::string escaped;
  escaped.reserve(value.size() + 8);
  for (char c : value) {
    switch (c) {
      case '\\':
        escaped += "\\\\";
        break;
      case '"':
        escaped += "\\\"";
        break;
      case '\n':
        escaped += "\\n";
        break;
      case '\r':
        escaped += "\\r";
        break;
      case '\t':
        escaped += "\\t";
        break;
      default:
        escaped.push_back(c);
        break;
    }
  }
  return escaped;
}

void WriteIndent(std::ostream& out, int indent) {
  for (int i = 0; i < indent; ++i) {
    out.put(' ');
  }
}

void WriteLiteralJson(std::ostream& out, const LiteralValue& literal, int indent);

void WriteArrayJson(std::ostream& out, const std::vector<LiteralValue>& array, int indent) {
  out << "[";
  if (!array.empty()) {
    out << "\n";
  }

  for (size_t i = 0; i < array.size(); ++i) {
    WriteIndent(out, indent + 2);
    WriteLiteralJson(out, array[i], indent + 2);
    if (i + 1 < array.size()) {
      out << ",";
    }
    out << "\n";
  }

  if (!array.empty()) {
    WriteIndent(out, indent);
  }
  out << "]";
}

void WriteLiteralJson(std::ostream& out, const LiteralValue& literal, int indent) {
  switch (literal.kind) {
    case LiteralValue::Kind::kNull:
      out << "null";
      break;
    case LiteralValue::Kind::kBool:
      out << (literal.bool_value ? "true" : "false");
      break;
    case LiteralValue::Kind::kInt:
      out << literal.int_value;
      break;
    case LiteralValue::Kind::kDouble:
      out << literal.double_value;
      break;
    case LiteralValue::Kind::kString:
      out << "\"" << JsonEscape(literal.string_value) << "\"";
      break;
    case LiteralValue::Kind::kArray:
      WriteArrayJson(out, literal.array_value, indent);
      break;
  }
}

void WriteStringArray(std::ostream& out, const std::vector<std::string>& values, int indent) {
  out << "[";
  if (!values.empty()) {
    out << "\n";
  }

  for (size_t i = 0; i < values.size(); ++i) {
    WriteIndent(out, indent + 2);
    out << "\"" << JsonEscape(values[i]) << "\"";
    if (i + 1 < values.size()) {
      out << ",";
    }
    out << "\n";
  }

  if (!values.empty()) {
    WriteIndent(out, indent);
  }
  out << "]";
}

void WriteTensorCall(std::ostream& out, const TensorCall& tensor, int indent) {
  WriteIndent(out, indent);
  out << "{\n";
  WriteIndent(out, indent + 2);
  out << "\"name\": \"" << JsonEscape(tensor.name) << "\",\n";
  WriteIndent(out, indent + 2);
  out << "\"cpp_type\": \"" << JsonEscape(tensor.cpp_type) << "\",\n";
  WriteIndent(out, indent + 2);
  out << "\"shape_expr\": \"" << JsonEscape(tensor.shape_expr) << "\",\n";
  WriteIndent(out, indent + 2);
  out << "\"values_expr\": \"" << JsonEscape(tensor.values_expr) << "\",\n";
  WriteIndent(out, indent + 2);
  out << "\"is_initializer\": " << (tensor.is_initializer ? "true" : "false") << ",\n";
  WriteIndent(out, indent + 2);
  out << "\"shape\": ";
  if (tensor.shape.has_value()) {
    WriteLiteralJson(out, *tensor.shape, indent + 2);
  } else {
    out << "null";
  }
  out << ",\n";
  WriteIndent(out, indent + 2);
  out << "\"values\": ";
  if (tensor.values.has_value()) {
    WriteLiteralJson(out, *tensor.values, indent + 2);
  } else {
    out << "null";
  }
  out << ",\n";
  WriteIndent(out, indent + 2);
  out << "\"warnings\": ";
  WriteStringArray(out, tensor.warnings, indent + 2);
  out << "\n";
  WriteIndent(out, indent);
  out << "}";
}

void WriteAttributeCall(std::ostream& out, const AttributeCall& attribute, int indent) {
  WriteIndent(out, indent);
  out << "{\n";
  WriteIndent(out, indent + 2);
  out << "\"name\": \"" << JsonEscape(attribute.name) << "\",\n";
  WriteIndent(out, indent + 2);
  out << "\"value_expr\": \"" << JsonEscape(attribute.value_expr) << "\",\n";
  WriteIndent(out, indent + 2);
  out << "\"value\": ";
  if (attribute.value.has_value()) {
    WriteLiteralJson(out, *attribute.value, indent + 2);
  } else {
    out << "null";
  }
  out << ",\n";
  WriteIndent(out, indent + 2);
  out << "\"warnings\": ";
  WriteStringArray(out, attribute.warnings, indent + 2);
  out << "\n";
  WriteIndent(out, indent);
  out << "}";
}

void WriteTensorList(std::ostream& out, const std::vector<TensorCall>& tensors, int indent) {
  out << "[";
  if (!tensors.empty()) {
    out << "\n";
  }
  for (size_t i = 0; i < tensors.size(); ++i) {
    WriteTensorCall(out, tensors[i], indent + 2);
    if (i + 1 < tensors.size()) {
      out << ",";
    }
    out << "\n";
  }
  if (!tensors.empty()) {
    WriteIndent(out, indent);
  }
  out << "]";
}

void WriteAttributeList(std::ostream& out, const std::vector<AttributeCall>& attributes, int indent) {
  out << "[";
  if (!attributes.empty()) {
    out << "\n";
  }
  for (size_t i = 0; i < attributes.size(); ++i) {
    WriteAttributeCall(out, attributes[i], indent + 2);
    if (i + 1 < attributes.size()) {
      out << ",";
    }
    out << "\n";
  }
  if (!attributes.empty()) {
    WriteIndent(out, indent);
  }
  out << "]";
}

std::string CompletenessOf(const TestRecord& record) {
  if (record.inputs.empty()) {
    return "partial";
  }

  size_t resolved_inputs = 0;
  for (const TensorCall& input : record.inputs) {
    if (input.shape.has_value() && input.values.has_value()) {
      ++resolved_inputs;
    }
  }

  if (resolved_inputs == record.inputs.size()) {
    return "complete";
  }

  if (resolved_inputs > 0) {
    return "partial";
  }

  return "metadata_only";
}

void WriteRecord(std::ostream& out, const TestRecord& record, int indent) {
  WriteIndent(out, indent);
  out << "{\n";
  WriteIndent(out, indent + 2);
  out << "\"source_file\": \"" << JsonEscape(record.source_file) << "\",\n";
  WriteIndent(out, indent + 2);
  out << "\"macro\": \"" << JsonEscape(record.macro_name) << "\",\n";
  WriteIndent(out, indent + 2);
  out << "\"test_suite\": \"" << JsonEscape(record.test_suite) << "\",\n";
  WriteIndent(out, indent + 2);
  out << "\"test_name\": \"" << JsonEscape(record.test_name) << "\",\n";
  WriteIndent(out, indent + 2);
  out << "\"tester_variable\": \"" << JsonEscape(record.tester_var) << "\",\n";
  WriteIndent(out, indent + 2);
  out << "\"op_name\": \"" << JsonEscape(record.op_name) << "\",\n";
  WriteIndent(out, indent + 2);
  out << "\"opset_expr\": \"" << JsonEscape(record.opset_expr) << "\",\n";
  WriteIndent(out, indent + 2);
  out << "\"opset\": ";
  if (record.opset.has_value()) {
    out << *record.opset;
  } else {
    out << "null";
  }
  out << ",\n";
  WriteIndent(out, indent + 2);
  out << "\"domain_expr\": \"" << JsonEscape(record.domain_expr) << "\",\n";
  WriteIndent(out, indent + 2);
  out << "\"saw_run_call\": " << (record.saw_run ? "true" : "false") << ",\n";
  WriteIndent(out, indent + 2);
  out << "\"completeness\": \"" << CompletenessOf(record) << "\",\n";
  WriteIndent(out, indent + 2);
  out << "\"inputs\": ";
  WriteTensorList(out, record.inputs, indent + 2);
  out << ",\n";
  WriteIndent(out, indent + 2);
  out << "\"outputs\": ";
  WriteTensorList(out, record.outputs, indent + 2);
  out << ",\n";
  WriteIndent(out, indent + 2);
  out << "\"attributes\": ";
  WriteAttributeList(out, record.attributes, indent + 2);
  out << ",\n";
  WriteIndent(out, indent + 2);
  out << "\"warnings\": ";
  WriteStringArray(out, record.warnings, indent + 2);
  out << "\n";
  WriteIndent(out, indent);
  out << "}";
}

bool IsMsDomainRecord(const TestRecord& record) {
  return record.domain_expr.find("kMSDomain") != std::string::npos;
}

ExtractionResult ExtractFromSourceTree(const fs::path& root, bool ms_domain_only) {
  ExtractionResult result;

  std::vector<fs::path> files;
  if (fs::is_regular_file(root)) {
    files.push_back(root);
  } else {
    for (const fs::directory_entry& entry : fs::recursive_directory_iterator(root)) {
      if (!entry.is_regular_file()) {
        continue;
      }

      if (entry.path().extension() != ".cc" && entry.path().extension() != ".cpp") {
        continue;
      }

      files.push_back(entry.path());
    }
  }

  std::sort(files.begin(), files.end());
  result.files_scanned = files.size();

  for (const fs::path& file : files) {
    std::vector<TestRecord> parsed = ParseSourceFile(file);
    for (TestRecord& record : parsed) {
      if (ms_domain_only && !IsMsDomainRecord(record)) {
        continue;
      }
      result.records.push_back(std::move(record));
    }
  }

  return result;
}

void WriteJsonFile(const fs::path& output_path, const ExtractionResult& result, const fs::path& source_root, bool ms_domain_only) {
  if (output_path.has_parent_path()) {
    fs::create_directories(output_path.parent_path());
  }

  std::ofstream output(output_path);
  if (!output) {
    throw std::runtime_error("Unable to open output file: " + output_path.string());
  }

  output << "{\n";
  WriteIndent(output, 2);
  output << "\"source_root\": \"" << JsonEscape(PathRelativeToOnnxruntimeOrg(source_root)) << "\",\n";
  WriteIndent(output, 2);
  output << "\"ms_domain_only\": " << (ms_domain_only ? "true" : "false") << ",\n";
  WriteIndent(output, 2);
  output << "\"files_scanned\": " << result.files_scanned << ",\n";
  WriteIndent(output, 2);
  output << "\"record_count\": " << result.records.size() << ",\n";
  WriteIndent(output, 2);
  output << "\"records\": [\n";

  for (size_t i = 0; i < result.records.size(); ++i) {
    WriteRecord(output, result.records[i], 4);
    if (i + 1 < result.records.size()) {
      output << ",";
    }
    output << "\n";
  }

  WriteIndent(output, 2);
  output << "]\n";
  output << "}\n";
}

void PrintUsage() {
  std::cerr
      << "Usage: ort_cpp_test_extractor --source <path> --output <json> [--all-domains]\n"
      << "  --source      C++ test file or directory to scan.\n"
      << "  --output      JSON file to write.\n"
      << "  --all-domains Include non-kMSDomain tests as well.\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    fs::path source_path;
    fs::path output_path;
    bool ms_domain_only = true;

    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--source" && i + 1 < argc) {
        source_path = argv[++i];
      } else if (arg == "--output" && i + 1 < argc) {
        output_path = argv[++i];
      } else if (arg == "--all-domains") {
        ms_domain_only = false;
      } else {
        PrintUsage();
        return 1;
      }
    }

    if (source_path.empty() || output_path.empty()) {
      PrintUsage();
      return 1;
    }

    if (!fs::exists(source_path)) {
      throw std::runtime_error("Source path does not exist: " + source_path.string());
    }

    const ExtractionResult result = ExtractFromSourceTree(source_path, ms_domain_only);
    WriteJsonFile(output_path, result, source_path, ms_domain_only);

    size_t complete = 0;
    size_t partial = 0;
    size_t metadata_only = 0;
    for (const TestRecord& record : result.records) {
      const std::string completeness = CompletenessOf(record);
      if (completeness == "complete") {
        ++complete;
      } else if (completeness == "partial") {
        ++partial;
      } else {
        ++metadata_only;
      }
    }

    std::cout
        << "Wrote " << result.records.size() << " extracted test records to " << fs::absolute(output_path).string() << "\n"
        << "Files scanned: " << result.files_scanned << "\n"
        << "Complete: " << complete << ", partial: " << partial << ", metadata_only: " << metadata_only << "\n";
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "ort_cpp_test_extractor failed: " << ex.what() << "\n";
    return 1;
  }
}
