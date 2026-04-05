// fmt_sycl.hpp — std::format-like API for SYCL device kernels (Approach A)
//
// Compile-time converts "{}" / "{:spec}" format strings to printf format
// specifiers, then forwards to sycl::ext::oneapi::experimental::printf.
//
// Usage:
//   fmt::sycl::print<"{} + {} = {}">(a, b, c);
//   FMT_PRINT("{} + {} = {}", a, b, c);   // macro for nicer syntax

#pragma once

#include <sycl/ext/oneapi/experimental/builtins.hpp>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility> // std::index_sequence

namespace fmt {
namespace sycl {
namespace detail {

// ============================================================
// fixed_string — compile-time string usable as NTTP
// ============================================================

template <size_t N>
struct fixed_string {
  char data[N]{};

  constexpr fixed_string() = default;
  constexpr fixed_string(const char (&s)[N]) {
    for (size_t i = 0; i < N; ++i)
      data[i] = s[i];
  }

  constexpr char operator[](size_t i) const { return data[i]; }
};

template <size_t N>
fixed_string(const char (&)[N]) -> fixed_string<N>;

// Length of a fixed_string (excluding null terminator)
template <size_t N>
consteval size_t flen(const fixed_string<N> &) {
  return N > 0 ? N - 1 : 0;
}

// ============================================================
// placeholder_info — describes the first placeholder found
// ============================================================

struct placeholder_info {
  size_t open;     // index of '{'
  size_t close;    // index of '}'
  size_t spec_beg; // index after ':' (or close if no spec)
  bool has_spec;
  bool found;
};

// Find the first {} or {:spec} placeholder starting at position From
template <fixed_string Fmt, size_t From = 0>
consteval placeholder_info find_placeholder() {
  constexpr size_t len = flen(Fmt);
  size_t i = From;
  while (i < len) {
    if (Fmt[i] == '{') {
      if (i + 1 < len && Fmt[i + 1] == '{') {
        i += 2;
        continue;
      }
      // Found start of placeholder — scan to closing '}'
      size_t j = i + 1;
      bool has_spec = false;
      size_t spec_beg = j; // default: no spec
      while (j < len && Fmt[j] != '}') {
        if (Fmt[j] == ':' && !has_spec) {
          has_spec = true;
          spec_beg = j + 1;
        }
        j++;
      }
      return {i, j, spec_beg, has_spec, true};
    } else if (Fmt[i] == '}') {
      if (i + 1 < len && Fmt[i + 1] == '}') {
        i += 2;
        continue;
      }
      i++; // stray } — skip
    } else {
      i++;
    }
  }
  return {0, 0, 0, false, false};
}

// ============================================================
// Literal segment: unescape {{ → {, }} → }, and % → %%
// ============================================================

template <fixed_string Fmt, size_t Begin, size_t End>
consteval size_t literal_out_size() {
  size_t n = 0;
  size_t i = Begin;
  while (i < End) {
    if (i + 1 < End && Fmt[i] == '{' && Fmt[i + 1] == '{') {
      n++;
      i += 2;
    } else if (i + 1 < End && Fmt[i] == '}' && Fmt[i + 1] == '}') {
      n++;
      i += 2;
    } else if (Fmt[i] == '%') {
      n += 2;
      i++;
    } else {
      n++;
      i++;
    }
  }
  return n;
}

template <fixed_string Fmt, size_t Begin, size_t End>
consteval auto make_literal() {
  constexpr size_t N = literal_out_size<Fmt, Begin, End>() + 1;
  fixed_string<N> result{};
  size_t out = 0;
  size_t i = Begin;
  while (i < End) {
    if (i + 1 < End && Fmt[i] == '{' && Fmt[i + 1] == '{') {
      result.data[out++] = '{';
      i += 2;
    } else if (i + 1 < End && Fmt[i] == '}' && Fmt[i + 1] == '}') {
      result.data[out++] = '}';
      i += 2;
    } else if (Fmt[i] == '%') {
      result.data[out++] = '%';
      result.data[out++] = '%';
      i++;
    } else {
      result.data[out++] = Fmt[i++];
    }
  }
  result.data[out] = '\0';
  return result;
}

// ============================================================
// emit_literal — printf a fixed_string ensuring constant addr space
// ============================================================

// Expand the fixed_string into a static constexpr char[] via index_sequence.
// This guarantees the SYCL compiler places it in the constant address space.
template <fixed_string Lit, size_t... Is>
inline void emit_literal_impl(std::index_sequence<Is...>) {
  static constexpr char s[] = {Lit.data[Is]..., '\0'};
  ::sycl::ext::oneapi::experimental::printf(s);
}

template <fixed_string Lit>
inline void emit_literal() {
  constexpr size_t len = flen(Lit);
  if constexpr (len > 0) {
    emit_literal_impl<Lit>(std::make_index_sequence<len>{});
  }
}

// ============================================================
// print_arg_default — print one arg with its default specifier
// ============================================================

template <typename T>
inline void print_arg_default(T arg) {
  using U = std::remove_cv_t<std::decay_t<T>>;

  if constexpr (std::is_same_v<U, bool>) {
    if (arg)
      ::sycl::ext::oneapi::experimental::printf("true");
    else
      ::sycl::ext::oneapi::experimental::printf("false");
  } else if constexpr (std::is_same_v<U, char>) {
    ::sycl::ext::oneapi::experimental::printf("%c", arg);
  } else if constexpr (std::is_integral_v<U> && std::is_signed_v<U>) {
    if constexpr (sizeof(U) <= 4)
      ::sycl::ext::oneapi::experimental::printf("%d",
                                                static_cast<int>(arg));
    else
      ::sycl::ext::oneapi::experimental::printf(
          "%lld", static_cast<long long>(arg));
  } else if constexpr (std::is_integral_v<U> && std::is_unsigned_v<U>) {
    if constexpr (sizeof(U) <= 4)
      ::sycl::ext::oneapi::experimental::printf("%u",
                                                static_cast<unsigned>(arg));
    else
      ::sycl::ext::oneapi::experimental::printf(
          "%llu", static_cast<unsigned long long>(arg));
  } else if constexpr (std::is_floating_point_v<U>) {
    ::sycl::ext::oneapi::experimental::printf("%g",
                                              static_cast<double>(arg));
  } else if constexpr (std::is_pointer_v<U>) {
    using Pointee = std::remove_cv_t<std::remove_pointer_t<U>>;
    if constexpr (std::is_same_v<Pointee, char>)
      ::sycl::ext::oneapi::experimental::printf("%s", arg);
    else
      ::sycl::ext::oneapi::experimental::printf("%p", arg);
  }
}

// ============================================================
// Format spec parsing and printf format string generation
// ============================================================

consteval bool is_type_char(char c) {
  return c == 'd' || c == 'x' || c == 'X' || c == 'o' ||
         c == 'b' || c == 'B' ||
         c == 'f' || c == 'e' || c == 'E' ||
         c == 'g' || c == 'G' ||
         c == 'a' || c == 'A' ||
         c == 'c' || c == 's' || c == 'p';
}

consteval bool is_align_char(char c) {
  return c == '<' || c == '>' || c == '^';
}

// Parsed format spec: [[fill]align][sign][#][0][width][.precision][type]
struct format_spec {
  char fill = '\0';
  char align = '\0';
  char sign = '\0';
  bool alt = false;
  bool zero_pad = false;
  int width = 0;
  int precision = -1;
  char type = '\0';
};

template <fixed_string Fmt, size_t Begin, size_t End>
consteval format_spec parse_spec() {
  format_spec s{};
  size_t i = Begin;

  // [[fill]align]
  if (i + 1 < End && is_align_char(Fmt[i + 1])) {
    s.fill = Fmt[i];
    s.align = Fmt[i + 1];
    i += 2;
  } else if (i < End && is_align_char(Fmt[i])) {
    s.align = Fmt[i];
    i++;
  }

  // [sign]
  if (i < End && (Fmt[i] == '+' || Fmt[i] == '-' || Fmt[i] == ' ')) {
    s.sign = Fmt[i];
    i++;
  }

  // [#]
  if (i < End && Fmt[i] == '#') {
    s.alt = true;
    i++;
  }

  // [0]
  if (i < End && Fmt[i] == '0') {
    s.zero_pad = true;
    i++;
  }

  // [width]
  while (i < End && Fmt[i] >= '0' && Fmt[i] <= '9') {
    s.width = s.width * 10 + (Fmt[i] - '0');
    i++;
  }

  // [.precision]
  if (i < End && Fmt[i] == '.') {
    i++;
    s.precision = 0;
    while (i < End && Fmt[i] >= '0' && Fmt[i] <= '9') {
      s.precision = s.precision * 10 + (Fmt[i] - '0');
      i++;
    }
  }

  // [type]
  if (i < End && is_type_char(Fmt[i])) {
    s.type = Fmt[i];
  }

  return s;
}

// Effective printf type char given the spec and the C++ argument type
template <typename U, char SpecType>
consteval char effective_type() {
  if (SpecType != '\0') return SpecType;
  if constexpr (std::is_same_v<U, char>) return 'c';
  else if constexpr (std::is_floating_point_v<U>) return 'g';
  else if constexpr (std::is_integral_v<U> && std::is_signed_v<U>) return 'd';
  else if constexpr (std::is_integral_v<U> && std::is_unsigned_v<U>) return 'u';
  else if constexpr (std::is_pointer_v<U>) {
    using P = std::remove_cv_t<std::remove_pointer_t<U>>;
    if constexpr (std::is_same_v<P, char>) return 's';
    else return 'p';
  } else return 'd';
}

// Buffer for building printf format strings at compile time
struct printf_fmt_buf {
  char data[32]{};
  size_t len = 0;

  constexpr void push(char c) { data[len++] = c; }
  constexpr void push_int(int val) {
    if (val == 0) { push('0'); return; }
    char tmp[10]{};
    int n = 0;
    while (val > 0) { tmp[n++] = '0' + val % 10; val /= 10; }
    for (int j = n - 1; j >= 0; j--) push(tmp[j]);
  }
};

// Build the printf format string: %[flags][width][.precision][length]type
template <format_spec Spec, char EffType, bool Is64>
consteval printf_fmt_buf build_printf_fmt() {
  printf_fmt_buf buf;
  buf.push('%');

  // Flags
  if (Spec.align == '<') buf.push('-');
  if (Spec.sign == '+') buf.push('+');
  else if (Spec.sign == ' ') buf.push(' ');
  if (Spec.alt) buf.push('#');
  if (Spec.zero_pad && Spec.align != '<') buf.push('0');

  // Width
  if (Spec.width > 0) buf.push_int(Spec.width);

  // Precision
  if (Spec.precision >= 0) {
    buf.push('.');
    buf.push_int(Spec.precision);
  }

  // Length modifier for 64-bit integers
  bool is_int = (EffType == 'd' || EffType == 'u' ||
                 EffType == 'x' || EffType == 'X' || EffType == 'o');
  if (is_int && Is64) {
    buf.push('l');
    buf.push('l');
  }

  // Type
  buf.push(EffType);
  buf.data[buf.len] = '\0';
  return buf;
}

// ============================================================
// emit_printf_with_arg — printf a format spec with one argument
// ============================================================

template <printf_fmt_buf PF, typename T, size_t... Is>
inline void emit_printf_with_arg_impl(T arg, std::index_sequence<Is...>) {
  static constexpr char s[] = {PF.data[Is]..., '\0'};
  ::sycl::ext::oneapi::experimental::printf(s, arg);
}

template <printf_fmt_buf PF, typename T>
inline void emit_printf_with_arg(T arg) {
  emit_printf_with_arg_impl<PF>(arg, std::make_index_sequence<PF.len>{});
}

// ============================================================
// print_arg_with_spec — print one arg using a parsed format spec
// ============================================================

template <format_spec Spec, typename T>
inline void print_arg_with_spec(T arg) {
  using U = std::remove_cv_t<std::decay_t<T>>;

  // Bool without explicit type → "true"/"false" (no printf formatting)
  if constexpr (std::is_same_v<U, bool> && Spec.type == '\0') {
    if (arg)
      ::sycl::ext::oneapi::experimental::printf("true");
    else
      ::sycl::ext::oneapi::experimental::printf("false");
    return;
  }

  // Binary (b/B) → Phase 4, degrade to default
  if constexpr (Spec.type == 'b' || Spec.type == 'B') {
    print_arg_default(arg);
    return;
  }

  constexpr char etype = effective_type<U, Spec.type>();
  constexpr bool is_64 = sizeof(U) > 4;
  constexpr auto pfmt = build_printf_fmt<Spec, etype, is_64>();

  // Cast argument to the type printf expects
  if constexpr (etype == 'c') {
    emit_printf_with_arg<pfmt>(static_cast<char>(arg));
  } else if constexpr (etype == 'd') {
    if constexpr (sizeof(U) <= 4)
      emit_printf_with_arg<pfmt>(static_cast<int>(arg));
    else
      emit_printf_with_arg<pfmt>(static_cast<long long>(arg));
  } else if constexpr (etype == 'u' || etype == 'x' || etype == 'X' ||
                        etype == 'o') {
    if constexpr (sizeof(U) <= 4)
      emit_printf_with_arg<pfmt>(static_cast<unsigned>(arg));
    else
      emit_printf_with_arg<pfmt>(static_cast<unsigned long long>(arg));
  } else if constexpr (etype == 'f' || etype == 'e' || etype == 'E' ||
                        etype == 'g' || etype == 'G' || etype == 'a' ||
                        etype == 'A') {
    emit_printf_with_arg<pfmt>(static_cast<double>(arg));
  } else {
    emit_printf_with_arg<pfmt>(arg);
  }
}

// ============================================================
// print_impl — recursive: split on first placeholder, recurse
// ============================================================

// Base case: no more arguments — print remaining literal text
template <fixed_string Fmt, size_t Pos>
inline void print_impl() {
  constexpr size_t end = flen(Fmt);
  constexpr size_t out_sz = literal_out_size<Fmt, Pos, end>();
  if constexpr (out_sz > 0) {
    constexpr auto lit = make_literal<Fmt, Pos, end>();
    emit_literal<lit>();
  }
}

// Recursive case: handle next placeholder, then recurse
template <fixed_string Fmt, size_t Pos, typename T, typename... Rest>
inline void print_impl(T arg, Rest... rest) {
  constexpr auto info = find_placeholder<Fmt, Pos>();
  static_assert(info.found, "fmt-sycl: too many arguments for format string");

  // Print literal prefix before this placeholder
  constexpr size_t prefix_sz = literal_out_size<Fmt, Pos, info.open>();
  if constexpr (prefix_sz > 0) {
    constexpr auto prefix = make_literal<Fmt, Pos, info.open>();
    emit_literal<prefix>();
  }

  // Print the argument
  if constexpr (info.has_spec && info.close > info.spec_beg) {
    constexpr auto spec = parse_spec<Fmt, info.spec_beg, info.close>();
    print_arg_with_spec<spec>(arg);
  } else {
    print_arg_default(arg);
  }

  // Recurse with remaining format string and arguments
  print_impl<Fmt, info.close + 1>(rest...);
}

} // namespace detail

// ============================================================
// Public API
// ============================================================

// No arguments: print literal text only
template <detail::fixed_string Fmt>
inline void print() {
  detail::print_impl<Fmt, 0>();
}

// One or more arguments
template <detail::fixed_string Fmt, typename T, typename... Rest>
inline void print(T arg, Rest... rest) {
  detail::print_impl<Fmt, 0>(arg, rest...);
}

} // namespace sycl
} // namespace fmt

// Convenience macro — nicer syntax without explicit template angle brackets
#define FMT_PRINT(fmtstr, ...) ::fmt::sycl::print<fmtstr>(__VA_ARGS__)
