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

#include "dragonbox.hpp"

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
  int index;       // -1 = auto ({}), >=0 = positional ({N})
};

// Find the first {} or {:spec} or {N} or {N:spec} placeholder
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
      // Found start of placeholder — parse index then spec
      size_t j = i + 1;
      int index = -1; // auto by default

      // Parse optional numeric index
      if (j < len && Fmt[j] >= '0' && Fmt[j] <= '9') {
        index = 0;
        while (j < len && Fmt[j] >= '0' && Fmt[j] <= '9') {
          index = index * 10 + (Fmt[j] - '0');
          j++;
        }
      }

      // Parse optional :spec
      bool has_spec = false;
      size_t spec_beg = j;
      if (j < len && Fmt[j] == ':') {
        has_spec = true;
        spec_beg = j + 1;
        j++;
        while (j < len && Fmt[j] != '}') j++;
      }
      // j now points at '}' (or end)
      size_t close = j;
      if (!has_spec) spec_beg = close;
      return {i, close, spec_beg, has_spec, true, index};
    } else if (Fmt[i] == '}') {
      if (i + 1 < len && Fmt[i + 1] == '}') {
        i += 2;
        continue;
      }
      i++;
    } else {
      i++;
    }
  }
  return {0, 0, 0, false, false, -1};
}

// ============================================================
// get_arg — compile-time indexed access into parameter pack
// ============================================================

template <size_t I, typename T, typename... Rest>
constexpr auto get_arg(T arg, Rest... rest) {
  if constexpr (I == 0) return arg;
  else return get_arg<I - 1>(rest...);
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
    double val = static_cast<double>(arg);
    bool neg = __builtin_bit_cast(uint64_t, val) >> 63;
    if (neg) {
      ::sycl::ext::oneapi::experimental::printf("-");
      val = -val;
    }
    char dbuf[32];
    int dlen = dragonbox::format_shortest(dbuf, val);
    for (int i = 0; i < dlen; i++)
      ::sycl::ext::oneapi::experimental::printf("%c", dbuf[i]);
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
// Device-side buffer for Phase 4 (binary, fill, center)
// ============================================================

struct fmt_buf {
  static constexpr int cap = 255;
  char data[cap + 1]{};
  int len = 0;
  void push(char c) { if (len < cap) data[len++] = c; }
  void push_n(char c, int n) {
    for (int i = 0; i < n && len < cap; i++) data[len++] = c;
  }
};

inline void print_buf(const fmt_buf &buf) {
  for (int i = 0; i < buf.len; i++)
    ::sycl::ext::oneapi::experimental::printf("%c", buf.data[i]);
}

inline void print_fill(char c, int n) {
  for (int i = 0; i < n; i++)
    ::sycl::ext::oneapi::experimental::printf("%c", c);
}

// Format unsigned integer into buffer in given base
template <int Base, bool Upper, typename U>
inline void uint_to_buf(fmt_buf &buf, U val) {
  if (val == 0) { buf.push('0'); return; }
  char tmp[65];
  int n = 0;
  while (val > 0) {
    int d = static_cast<int>(val % Base);
    if constexpr (Upper)
      tmp[n++] = "0123456789ABCDEF"[d];
    else
      tmp[n++] = "0123456789abcdef"[d];
    val /= Base;
  }
  for (int i = n - 1; i >= 0; i--) buf.push(tmp[i]);
}

// Full integer formatting into buffer (handles sign, prefix, zero-pad, fill)
template <format_spec Spec, char EffType, typename T>
inline void format_int_buf(T arg) {
  using U = std::remove_cv_t<std::decay_t<T>>;
  using Uns = std::conditional_t<(sizeof(U) <= 4), unsigned, unsigned long long>;
  constexpr int base = (EffType == 'b' || EffType == 'B') ? 2
                     : (EffType == 'o')                    ? 8
                     : (EffType == 'x' || EffType == 'X')  ? 16
                                                           : 10;
  constexpr bool upper = (EffType == 'X' || EffType == 'B');

  // Absolute value + sign
  bool neg = false;
  Uns uval;
  if constexpr (std::is_same_v<U, bool>) {
    uval = static_cast<Uns>(arg);
  } else if constexpr (std::is_signed_v<U>) {
    if (arg < 0) { neg = true; uval = Uns(0) - static_cast<Uns>(arg); }
    else uval = static_cast<Uns>(arg);
  } else {
    uval = static_cast<Uns>(arg);
  }

  char sc = neg ? '-' : (Spec.sign == '+') ? '+' : (Spec.sign == ' ') ? ' ' : '\0';

  // Alt prefix
  char pfx[3] = {};
  int pfx_n = 0;
  if constexpr (Spec.alt) {
    if constexpr (base == 2) { pfx[0] = '0'; pfx[1] = upper ? 'B' : 'b'; pfx_n = 2; }
    else if constexpr (base == 16) { pfx[0] = '0'; pfx[1] = upper ? 'X' : 'x'; pfx_n = 2; }
    else if constexpr (base == 8) { if (uval != 0) { pfx[0] = '0'; pfx_n = 1; } }
  }

  // Digits
  fmt_buf digits;
  uint_to_buf<base, upper>(digits, uval);

  // Sign-aware zero padding (only when no explicit fill/align)
  int content_w = (sc ? 1 : 0) + pfx_n + digits.len;
  bool zpad = Spec.zero_pad && !Spec.fill && !Spec.align;

  fmt_buf content;
  if (sc) content.push(sc);
  for (int i = 0; i < pfx_n; i++) content.push(pfx[i]);
  if (zpad && Spec.width > content_w)
    content.push_n('0', Spec.width - content_w);
  for (int i = 0; i < digits.len; i++) content.push(digits.data[i]);

  // Apply fill + alignment
  char fill_c = Spec.fill ? Spec.fill : ' ';
  char align_c = Spec.align ? Spec.align : '>';
  int pad = Spec.width > content.len ? Spec.width - content.len : 0;

  fmt_buf out;
  if (pad == 0) {
    for (int i = 0; i < content.len; i++) out.push(content.data[i]);
  } else if (align_c == '<') {
    for (int i = 0; i < content.len; i++) out.push(content.data[i]);
    out.push_n(fill_c, pad);
  } else if (align_c == '^') {
    out.push_n(fill_c, pad / 2);
    for (int i = 0; i < content.len; i++) out.push(content.data[i]);
    out.push_n(fill_c, pad - pad / 2);
  } else {
    out.push_n(fill_c, pad);
    for (int i = 0; i < content.len; i++) out.push(content.data[i]);
  }
  print_buf(out);
}

// Compute width of float formatted as {:f} (for fill padding calculation)
template <format_spec Spec>
inline int compute_float_f_width(double val) {
  int w = 0;
  if (val < 0.0) { w++; val = -val; }
  else if (Spec.sign == '+' || Spec.sign == ' ') w++;
  if (val < 1.0) w++;
  else { double v = val; while (v >= 1.0) { w++; v /= 10.0; } }
  int p = Spec.precision >= 0 ? Spec.precision : 6;
  if (p > 0 || Spec.alt) w++;
  w += p;
  return w;
}

// Apply fill + alignment and print (shared by int/hex-float buffer paths)
inline void apply_padding_and_print(const fmt_buf &content, char fill,
                                    char align, int width) {
  int pad = width > content.len ? width - content.len : 0;
  if (pad == 0) {
    print_buf(content);
  } else if (align == '<') {
    print_buf(content);
    print_fill(fill, pad);
  } else if (align == '^') {
    print_fill(fill, pad / 2);
    print_buf(content);
    print_fill(fill, pad - pad / 2);
  } else {
    print_fill(fill, pad);
    print_buf(content);
  }
}

// Hex digit helper
inline char hex_digit(int d, bool upper) {
  if (d < 10) return static_cast<char>('0' + d);
  return static_cast<char>((upper ? 'A' : 'a') + d - 10);
}

// Format hex float from IEEE 754 bits (fixes {:a} 0x prefix difference)
template <format_spec Spec, char EffType, typename T>
inline void format_hex_float(T arg) {
  double val = static_cast<double>(arg);
  constexpr bool upper = (EffType == 'A');

  fmt_buf content;

  // Extract IEEE 754 bits (must do before sign check for -0.0)
  uint64_t bits = __builtin_bit_cast(uint64_t, val);
  bool negative = (bits >> 63) != 0;

  // Sign (use sign bit, not val < 0, to catch -0.0)
  if (negative) { content.push('-'); val = -val; bits &= 0x7FFFFFFFFFFFFFFFULL; }
  else if (Spec.sign == '+') content.push('+');
  else if (Spec.sign == ' ') content.push(' ');

  // std::format {:a} never adds 0x prefix (unlike printf)
  // # only forces the decimal point (handled below)
  int biased_exp = static_cast<int>((bits >> 52) & 0x7FF); // sign already cleared
  uint64_t mantissa = bits & 0x000FFFFFFFFFFFFFULL;

  // Inf / NaN
  if (biased_exp == 0x7FF) {
    const char *s = (mantissa == 0) ? (upper ? "INF" : "inf")
                                    : (upper ? "NAN" : "nan");
    while (*s) content.push(*s++);
    apply_padding_and_print(content, Spec.fill ? Spec.fill : ' ',
                            Spec.align ? Spec.align : '>', Spec.width);
    return;
  }

  // Leading digit + exponent
  int exponent;
  if (biased_exp == 0) {
    if (mantissa == 0) { // zero
      content.push('0');
      content.push(upper ? 'P' : 'p');
      content.push('+');
      content.push('0');
      apply_padding_and_print(content, Spec.fill ? Spec.fill : ' ',
                              Spec.align ? Spec.align : '>', Spec.width);
      return;
    }
    content.push('0'); // subnormal
    exponent = -1022;
  } else {
    content.push('1'); // normal
    exponent = biased_exp - 1023;
  }

  // Fractional mantissa (52 bits = 13 hex digits, trim trailing zeros)
  if (mantissa != 0 || Spec.alt) {
    content.push('.');
    if (mantissa != 0) {
      int last_nz = -1;
      for (int i = 0; i < 13; i++)
        if (((mantissa >> (48 - i * 4)) & 0xF) != 0) last_nz = i;
      for (int i = 0; i <= last_nz; i++)
        content.push(hex_digit(static_cast<int>((mantissa >> (48 - i * 4)) & 0xF), upper));
    }
  }

  // Exponent
  content.push(upper ? 'P' : 'p');
  if (exponent >= 0) content.push('+');
  else { content.push('-'); exponent = -exponent; }
  if (exponent == 0) {
    content.push('0');
  } else {
    char tmp[10]; int n = 0;
    while (exponent > 0) { tmp[n++] = static_cast<char>('0' + exponent % 10); exponent /= 10; }
    for (int i = n - 1; i >= 0; i--) content.push(tmp[i]);
  }

  apply_padding_and_print(content, Spec.fill ? Spec.fill : ' ',
                          Spec.align ? Spec.align : '>', Spec.width);
}

// Float with custom fill — print fill chars around printf output
template <format_spec Spec, char EffType, typename T>
inline void print_float_with_fill(T arg) {
  double val = static_cast<double>(arg);
  // Inner printf spec: same format but without width/fill/align
  constexpr format_spec inner = {
    '\0', '\0', Spec.sign, Spec.alt, false, 0, Spec.precision, Spec.type
  };
  constexpr auto pfmt = build_printf_fmt<inner, EffType, false>();

  int inner_w;
  if constexpr (EffType == 'f' || EffType == 'F')
    inner_w = compute_float_f_width<Spec>(val);
  else
    inner_w = Spec.width; // can't compute — skip padding

  char fill_c = Spec.fill ? Spec.fill : ' ';
  char align_c = Spec.align ? Spec.align : '>';
  int pad = Spec.width > inner_w ? Spec.width - inner_w : 0;

  if (align_c == '<') {
    emit_printf_with_arg<pfmt>(val);
    print_fill(fill_c, pad);
  } else if (align_c == '^') {
    print_fill(fill_c, pad / 2);
    emit_printf_with_arg<pfmt>(val);
    print_fill(fill_c, pad - pad / 2);
  } else {
    print_fill(fill_c, pad);
    emit_printf_with_arg<pfmt>(val);
  }
}

// ============================================================
// print_arg_with_spec — print one arg using a parsed format spec
// ============================================================

consteval bool is_int_format(char c) {
  return c == 'd' || c == 'u' || c == 'x' || c == 'X' ||
         c == 'o' || c == 'b' || c == 'B';
}

consteval bool is_float_format(char c) {
  return c == 'f' || c == 'e' || c == 'E' ||
         c == 'g' || c == 'G' || c == 'a' || c == 'A';
}

template <format_spec Spec, typename T>
inline void print_arg_with_spec(T arg) {
  using U = std::remove_cv_t<std::decay_t<T>>;

  // Bool without explicit type (or {:s}) → "true"/"false"
  if constexpr (std::is_same_v<U, bool> &&
                (Spec.type == '\0' || Spec.type == 's')) {
    fmt_buf content;
    if (arg) { content.push('t'); content.push('r'); content.push('u'); content.push('e'); }
    else { content.push('f'); content.push('a'); content.push('l'); content.push('s'); content.push('e'); }
    apply_padding_and_print(content, Spec.fill ? Spec.fill : ' ',
                            Spec.align ? Spec.align : '>', Spec.width);
    return;
  }

  constexpr char etype = effective_type<U, Spec.type>();

  // Determine if buffer path is needed
  constexpr bool is_signed_int = std::is_signed_v<U> && !std::is_same_v<U, bool>;
  constexpr bool needs_buf =
      (etype == 'b' || etype == 'B') ||
      (Spec.align == '^') ||
      (Spec.fill != '\0' && Spec.fill != ' ') ||
      // signed hex/oct: std::format shows -sign, printf doesn't
      (is_signed_int && (etype == 'x' || etype == 'X' || etype == 'o')) ||
      // {:#x} on 0: printf skips prefix, std::format doesn't
      (Spec.alt && (etype == 'x' || etype == 'X'));
  // {:a}/{:A} always needs custom hex float (std::format never adds 0x)
  constexpr bool needs_hex_float = (etype == 'a' || etype == 'A');

  if constexpr (needs_hex_float) {
    format_hex_float<Spec, etype>(arg);
  } else if constexpr (needs_buf && is_int_format(etype)) {
    // Buffer path — full integer formatting
    format_int_buf<Spec, etype>(arg);
  } else if constexpr (needs_buf && etype == 'c') {
    // Buffer path — char with fill/center
    fmt_buf content;
    content.push(static_cast<char>(arg));
    apply_padding_and_print(content, Spec.fill ? Spec.fill : ' ',
                            Spec.align ? Spec.align : '>', Spec.width);
  } else if constexpr (needs_buf && etype == 's') {
    // Buffer path — string with fill/center
    fmt_buf content;
    const char *s = arg;
    while (*s) content.push(*s++);
    apply_padding_and_print(content, Spec.fill ? Spec.fill : ' ',
                            Spec.align ? Spec.align : '<', Spec.width);
  } else if constexpr (needs_buf && is_float_format(etype)) {
    // Multi-printf path — fill around printf output
    print_float_with_fill<Spec, etype>(arg);
  } else {
    // Fast printf path
    constexpr bool is_64 = sizeof(U) > 4;
    constexpr auto pfmt = build_printf_fmt<Spec, etype, is_64>();

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
    } else if constexpr (is_float_format(etype)) {
      emit_printf_with_arg<pfmt>(static_cast<double>(arg));
    } else {
      emit_printf_with_arg<pfmt>(arg);
    }
  }
}

// ============================================================
// print_impl — walk format string, access args by index
// ============================================================

// Helper: print one arg (selected by index) with optional spec
template <fixed_string Fmt, placeholder_info Info, typename... Args>
inline void print_one_arg(Args... args) {
  constexpr size_t idx = Info.index >= 0 ? static_cast<size_t>(Info.index) : 0;
  auto arg = get_arg<idx>(args...);

  if constexpr (Info.has_spec && Info.close > Info.spec_beg) {
    constexpr auto spec = parse_spec<Fmt, Info.spec_beg, Info.close>();
    print_arg_with_spec<spec>(arg);
  } else {
    print_arg_default(arg);
  }
}

// Recursive: find next placeholder, print prefix + arg, advance
template <fixed_string Fmt, size_t Pos, size_t AutoIdx, typename... Args>
inline void print_impl(Args... args) {
  constexpr auto info = find_placeholder<Fmt, Pos>();

  if constexpr (!info.found) {
    // No more placeholders — print remaining literal text
    constexpr size_t end = flen(Fmt);
    constexpr size_t out_sz = literal_out_size<Fmt, Pos, end>();
    if constexpr (out_sz > 0) {
      constexpr auto lit = make_literal<Fmt, Pos, end>();
      emit_literal<lit>();
    }
  } else {
    // Print literal prefix before this placeholder
    constexpr size_t prefix_sz = literal_out_size<Fmt, Pos, info.open>();
    if constexpr (prefix_sz > 0) {
      constexpr auto prefix = make_literal<Fmt, Pos, info.open>();
      emit_literal<prefix>();
    }

    // Resolve arg index: positional {N} or auto-incremented {}
    constexpr bool is_auto = (info.index < 0);
    constexpr auto resolved = placeholder_info{
        info.open, info.close, info.spec_beg, info.has_spec, info.found,
        is_auto ? static_cast<int>(AutoIdx) : info.index};

    // Print the selected argument
    print_one_arg<Fmt, resolved>(args...);

    // Recurse — advance AutoIdx only if this was auto-indexed
    constexpr size_t next_auto = is_auto ? AutoIdx + 1 : AutoIdx;
    print_impl<Fmt, info.close + 1, next_auto>(args...);
  }
}

} // namespace detail

// ============================================================
// Public API
// ============================================================

template <detail::fixed_string Fmt, typename... Args>
inline void print(Args... args) {
  detail::print_impl<Fmt, 0, 0>(args...);
}

} // namespace sycl
} // namespace fmt

// Convenience macro — nicer syntax without explicit template angle brackets
#define FMT_PRINT(fmtstr, ...) ::fmt::sycl::print<fmtstr>(__VA_ARGS__)
