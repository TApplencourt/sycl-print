// sycl_khr_print.hpp — std::format-like API for SYCL device kernels
//
// Compile-time converts "{}" / "{:spec}" format strings to printf format
// specifiers, then forwards to sycl::ext::oneapi::experimental::printf.
//
// Usage:
//   sycl::khr::print<"{} + {} = {}">(a, b, c);
//   KHR_PRINT("{} + {} = {}", a, b, c);   // macro for nicer syntax

#pragma once

// Backend detection: DPC++ vs AdaptiveCpp
#if defined(__ADAPTIVECPP__) || defined(__HIPSYCL__) || defined(__ACPP__)
#define FMT_SYCL_ACPP 1
#include <sycl/sycl.hpp>
#else
#define FMT_SYCL_ACPP 0
#include <sycl/ext/oneapi/experimental/builtins.hpp>
#endif

#include <algorithm> // std::copy_n
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <tuple>
#include <type_traits>
#include <utility> // std::index_sequence

namespace sycl {
#if !FMT_SYCL_ACPP
inline namespace _V1 {
#endif
namespace khr {

namespace print_detail {

// ============================================================
// Dragonbox — shortest-decimal float formatting for device code
// ============================================================
// Ported from fmtlib (https://github.com/fmtlib/fmt) v12.0.1
// Original algorithm: https://github.com/jk-jeon/dragonbox
// License: MIT (same as fmt)
//
// Produces the shortest decimal representation of float/double.
// Uses compressed cache tables (216 bytes) — GPU-friendly.

#if FMT_SYCL_ACPP
namespace dragonbox {

struct uint128 {
  uint64_t hi, lo;
  constexpr uint128(uint64_t h, uint64_t l) : hi(h), lo(l) {}
  constexpr uint128(uint64_t v = 0) : hi(0), lo(v) {}
  constexpr uint64_t high() const { return hi; }
  constexpr uint64_t low() const { return lo; }
  constexpr uint128 &operator+=(uint64_t rhs) {
    uint64_t new_lo = lo + rhs;
    hi += (new_lo < lo) ? 1 : 0;
    lo = new_lo;
    return *this;
  }
};

inline auto umul128(uint64_t x, uint64_t y) noexcept -> uint128 {
  const uint64_t mask = 0xFFFFFFFFu;
  uint64_t a = x >> 32, b = x & mask;
  uint64_t c = y >> 32, d = y & mask;
  uint64_t ac = a * c, bc = b * c, ad = a * d, bd = b * d;
  uint64_t mid = (bd >> 32) + (ad & mask) + (bc & mask);
  return {ac + (mid >> 32) + (ad >> 32) + (bc >> 32), (mid << 32) + (bd & mask)};
}

inline auto umul128_upper64(uint64_t x, uint64_t y) noexcept -> uint64_t {
  return umul128(x, y).high();
}

inline auto umul192_upper128(uint64_t x, uint128 y) noexcept -> uint128 {
  uint128 r = umul128(x, y.high());
  r += umul128_upper64(x, y.low());
  return r;
}

inline auto umul192_lower128(uint64_t x, uint128 y) noexcept -> uint128 {
  uint64_t high = x * y.high();
  uint128 high_low = umul128(x, y.low());
  return {high + high_low.high(), high_low.low()};
}

inline auto umul96_upper64(uint32_t x, uint64_t y) noexcept -> uint64_t {
  return umul128_upper64(static_cast<uint64_t>(x) << 32, y);
}

inline auto umul96_lower64(uint32_t x, uint64_t y) noexcept -> uint64_t { return x * y; }

inline auto rotr(uint32_t n, uint32_t r) noexcept -> uint32_t {
  r &= 31;
  return (n >> r) | (n << (32 - r));
}

inline auto rotr(uint64_t n, uint32_t r) noexcept -> uint64_t {
  r &= 63;
  return (n >> r) | (n << (64 - r));
}

inline auto floor_log10_pow2(int e) noexcept -> int { return (e * 315653) >> 20; }

inline auto floor_log2_pow10(int e) noexcept -> int { return (e * 1741647) >> 19; }

inline auto floor_log10_pow2_minus_log10_4_over_3(int e) noexcept -> int {
  return (e * 631305 - 261663) >> 21;
}

template <typename T> struct float_info;

template <> struct float_info<float> {
  using carrier_uint = uint32_t;
  static constexpr int exponent_bits = 8;
  static constexpr int kappa = 1;
  static constexpr int big_divisor = 100;
  static constexpr int small_divisor = 10;
  static constexpr int min_k = -31;
  static constexpr int max_k = 46;
  static constexpr int shorter_interval_tie_lower_threshold = -35;
  static constexpr int shorter_interval_tie_upper_threshold = -35;
};

template <> struct float_info<double> {
  using carrier_uint = uint64_t;
  static constexpr int exponent_bits = 11;
  static constexpr int kappa = 2;
  static constexpr int big_divisor = 1000;
  static constexpr int small_divisor = 100;
  static constexpr int min_k = -292;
  static constexpr int max_k = 341;
  static constexpr int shorter_interval_tie_lower_threshold = -77;
  static constexpr int shorter_interval_tie_upper_threshold = -77;
};

template <typename T> struct decimal_fp {
  using significand_type = typename float_info<T>::carrier_uint;
  significand_type significand;
  int exponent;
};

template <typename Float> constexpr auto num_significand_bits() -> int {
  return std::numeric_limits<Float>::digits - 1;
}

template <typename Float>
constexpr auto exponent_mask() -> typename float_info<Float>::carrier_uint {
  using uint = typename float_info<Float>::carrier_uint;
  return ((uint(1) << float_info<Float>::exponent_bits) - 1) << num_significand_bits<Float>();
}

template <typename Float> constexpr auto exponent_bias() -> int {
  return std::numeric_limits<Float>::max_exponent - 1;
}

struct div_info {
  uint32_t divisor;
  int shift;
};
static constexpr div_info div_infos[] = {{10, 16}, {100, 16}};

template <int N> auto check_divisibility_and_divide_by_pow10(uint32_t &n) noexcept -> bool {
  constexpr auto info = div_infos[N - 1];
  constexpr uint32_t magic = (1u << info.shift) / info.divisor + 1;
  n *= magic;
  const uint32_t mask = (1u << info.shift) - 1;
  bool result = (n & mask) < magic;
  n >>= info.shift;
  return result;
}

inline auto divide_by_10_to_kappa_plus_1(uint32_t n) noexcept -> uint32_t {
  return static_cast<uint32_t>((static_cast<uint64_t>(n) * 1374389535) >> 37);
}

inline auto divide_by_10_to_kappa_plus_1(uint64_t n) noexcept -> uint64_t {
  return umul128_upper64(n, 2361183241434822607ull) >> 7;
}

template <typename T> struct cache_accessor;

template <> struct cache_accessor<float> {
  using carrier_uint = uint32_t;
  using cache_entry_type = uint64_t;

  static auto get_cached_power(int k) noexcept -> uint64_t {
    static constexpr uint64_t table[] = {
        0x81ceb32c4b43fcf5, 0xa2425ff75e14fc32, 0xcad2f7f5359a3b3f, 0xfd87b5f28300ca0e,
        0x9e74d1b791e07e49, 0xc612062576589ddb, 0xf79687aed3eec552, 0x9abe14cd44753b53,
        0xc16d9a0095928a28, 0xf1c90080baf72cb2, 0x971da05074da7bef, 0xbce5086492111aeb,
        0xec1e4a7db69561a6, 0x9392ee8e921d5d08, 0xb877aa3236a4b44a, 0xe69594bec44de15c,
        0x901d7cf73ab0acda, 0xb424dc35095cd810, 0xe12e13424bb40e14, 0x8cbccc096f5088cc,
        0xafebff0bcb24aaff, 0xdbe6fecebdedd5bf, 0x89705f4136b4a598, 0xabcc77118461cefd,
        0xd6bf94d5e57a42bd, 0x8637bd05af6c69b6, 0xa7c5ac471b478424, 0xd1b71758e219652c,
        0x83126e978d4fdf3c, 0xa3d70a3d70a3d70b, 0xcccccccccccccccd, 0x8000000000000000,
        0xa000000000000000, 0xc800000000000000, 0xfa00000000000000, 0x9c40000000000000,
        0xc350000000000000, 0xf424000000000000, 0x9896800000000000, 0xbebc200000000000,
        0xee6b280000000000, 0x9502f90000000000, 0xba43b74000000000, 0xe8d4a51000000000,
        0x9184e72a00000000, 0xb5e620f480000000, 0xe35fa931a0000000, 0x8e1bc9bf04000000,
        0xb1a2bc2ec5000000, 0xde0b6b3a76400000, 0x8ac7230489e80000, 0xad78ebc5ac620000,
        0xd8d726b7177a8000, 0x878678326eac9000, 0xa968163f0a57b400, 0xd3c21bcecceda100,
        0x84595161401484a0, 0xa56fa5b99019a5c8, 0xcecb8f27f4200f3a, 0x813f3978f8940985,
        0xa18f07d736b90be6, 0xc9f2c9cd04674edf, 0xfc6f7c4045812297, 0x9dc5ada82b70b59e,
        0xc5371912364ce306, 0xf684df56c3e01bc7, 0x9a130b963a6c115d, 0xc097ce7bc90715b4,
        0xf0bdc21abb48db21, 0x96769950b50d88f5, 0xbc143fa4e250eb32, 0xeb194f8e1ae525fe,
        0x92efd1b8d0cf37bf, 0xb7abc627050305ae, 0xe596b7b0c643c71a, 0x8f7e32ce7bea5c70,
        0xb35dbf821ae4f38c, 0xe0352f62a19e306f};
    return table[k - float_info<float>::min_k];
  }

  struct compute_mul_result {
    carrier_uint result;
    bool is_integer;
  };
  struct compute_mul_parity_result {
    bool parity;
    bool is_integer;
  };

  static auto compute_mul(carrier_uint u,
                          const cache_entry_type &cache) noexcept -> compute_mul_result {
    auto r = umul96_upper64(u, cache);
    return {static_cast<carrier_uint>(r >> 32), static_cast<carrier_uint>(r) == 0};
  }

  static auto compute_delta(const cache_entry_type &cache, int beta) noexcept -> uint32_t {
    return static_cast<uint32_t>(cache >> (64 - 1 - beta));
  }

  static auto compute_mul_parity(carrier_uint two_f, const cache_entry_type &cache,
                                 int beta) noexcept -> compute_mul_parity_result {
    auto r = umul96_lower64(two_f, cache);
    return {((r >> (64 - beta)) & 1) != 0, static_cast<uint32_t>(r >> (32 - beta)) == 0};
  }

  static auto compute_left_endpoint_for_shorter_interval_case(const cache_entry_type &cache,
                                                              int beta) noexcept -> carrier_uint {
    return static_cast<carrier_uint>((cache - (cache >> (num_significand_bits<float>() + 2))) >>
                                     (64 - num_significand_bits<float>() - 1 - beta));
  }

  static auto compute_right_endpoint_for_shorter_interval_case(const cache_entry_type &cache,
                                                               int beta) noexcept -> carrier_uint {
    return static_cast<carrier_uint>((cache + (cache >> (num_significand_bits<float>() + 1))) >>
                                     (64 - num_significand_bits<float>() - 1 - beta));
  }

  static auto compute_round_up_for_shorter_interval_case(const cache_entry_type &cache,
                                                         int beta) noexcept -> carrier_uint {
    return (static_cast<carrier_uint>(cache >> (64 - num_significand_bits<float>() - 2 - beta)) +
            1) /
           2;
  }
};

template <> struct cache_accessor<double> {
  using carrier_uint = uint64_t;
  using cache_entry_type = uint128;

  static auto get_cached_power(int k) noexcept -> uint128 {
    static constexpr uint128 pow10_significands[] = {
        {0xff77b1fcbebcdc4f, 0x25e8e89c13bb0f7b}, {0xce5d73ff402d98e3, 0xfb0a3d212dc81290},
        {0xa6b34ad8c9dfc06f, 0xf42faa48c0ea481f}, {0x86a8d39ef77164bc, 0xae5dff9c02033198},
        {0xd98ddaee19068c76, 0x3badd624dd9b0958}, {0xafbd2350644eeacf, 0xe5d1929ef90898fb},
        {0x8df5efabc5979c8f, 0xca8d3ffa1ef463c2}, {0xe55990879ddcaabd, 0xcc420a6a101d0516},
        {0xb94470938fa89bce, 0xf808e40e8d5b3e6a}, {0x95a8637627989aad, 0xdde7001379a44aa9},
        {0xf1c90080baf72cb1, 0x5324c68b12dd6339}, {0xc350000000000000, 0x0000000000000000},
        {0x9dc5ada82b70b59d, 0xf020000000000000}, {0xfee50b7025c36a08, 0x02f236d04753d5b5},
        {0xcde6fd5e09abcf26, 0xed4c0226b55e6f87}, {0xa6539930bf6bff45, 0x84db8346b786151d},
        {0x865b86925b9bc5c2, 0x0b8a2392ba45a9b3}, {0xd910f7ff28069da4, 0x1b2ba1518094da05},
        {0xaf58416654a6babb, 0x387ac8d1970027b3}, {0x8da471a9de737e24, 0x5ceaecfed289e5d3},
        {0xe4d5e82392a40515, 0x0fabaf3feaa5334b}, {0xb8da1662e7b00a17, 0x3d6a751f3b936244},
        {0x95527a5202df0ccb, 0x0f37801e0c43ebc9}, {0xf13e34aabb430a15, 0x647726b9e7c68ff0},
    };

    static constexpr uint64_t powers_of_5_64[] = {
        0x0000000000000001, 0x0000000000000005, 0x0000000000000019, 0x000000000000007d,
        0x0000000000000271, 0x0000000000000c35, 0x0000000000003d09, 0x000000000001312d,
        0x000000000005f5e1, 0x00000000001dcd65, 0x00000000009502f9, 0x0000000002e90edd,
        0x000000000e8d4a51, 0x0000000048c27395, 0x000000016bcc41e9, 0x000000071afd498d,
        0x0000002386f26fc1, 0x000000b1a2bc2ec5, 0x000003782dace9d9, 0x00001158e460913d,
        0x000056bc75e2d631, 0x0001b1ae4d6e2ef5, 0x000878678326eac9, 0x002a5a058fc295ed,
        0x00d3c21bcecceda1, 0x0422ca8b0a00a425, 0x14adf4b7320334b9};

    static constexpr int compression_ratio = 27;

    int cache_index = (k - float_info<double>::min_k) / compression_ratio;
    int kb = cache_index * compression_ratio + float_info<double>::min_k;
    int offset = k - kb;

    uint128 base_cache = pow10_significands[cache_index];
    if (offset == 0)
      return base_cache;

    int alpha = floor_log2_pow10(kb + offset) - floor_log2_pow10(kb) - offset;

    uint64_t pow5 = powers_of_5_64[offset];
    uint128 recovered_cache = umul128(base_cache.high(), pow5);
    uint128 middle_low = umul128(base_cache.low(), pow5);

    recovered_cache += middle_low.high();

    uint64_t high_to_middle = recovered_cache.high() << (64 - alpha);
    uint64_t middle_to_low = recovered_cache.low() << (64 - alpha);

    recovered_cache = uint128{(recovered_cache.low() >> alpha) | high_to_middle,
                              ((middle_low.low() >> alpha) | middle_to_low)};
    return {recovered_cache.high(), recovered_cache.low() + 1};
  }

  struct compute_mul_result {
    carrier_uint result;
    bool is_integer;
  };
  struct compute_mul_parity_result {
    bool parity;
    bool is_integer;
  };

  static auto compute_mul(carrier_uint u,
                          const cache_entry_type &cache) noexcept -> compute_mul_result {
    auto r = umul192_upper128(u, cache);
    return {r.high(), r.low() == 0};
  }

  static auto compute_delta(const cache_entry_type &cache, int beta) noexcept -> uint32_t {
    return static_cast<uint32_t>(cache.high() >> (64 - 1 - beta));
  }

  static auto compute_mul_parity(carrier_uint two_f, const cache_entry_type &cache,
                                 int beta) noexcept -> compute_mul_parity_result {
    auto r = umul192_lower128(two_f, cache);
    return {((r.high() >> (64 - beta)) & 1) != 0,
            ((r.high() << beta) | (r.low() >> (64 - beta))) == 0};
  }

  static auto compute_left_endpoint_for_shorter_interval_case(const cache_entry_type &cache,
                                                              int beta) noexcept -> carrier_uint {
    return (cache.high() - (cache.high() >> (num_significand_bits<double>() + 2))) >>
           (64 - num_significand_bits<double>() - 1 - beta);
  }

  static auto compute_right_endpoint_for_shorter_interval_case(const cache_entry_type &cache,
                                                               int beta) noexcept -> carrier_uint {
    return (cache.high() + (cache.high() >> (num_significand_bits<double>() + 1))) >>
           (64 - num_significand_bits<double>() - 1 - beta);
  }

  static auto compute_round_up_for_shorter_interval_case(const cache_entry_type &cache,
                                                         int beta) noexcept -> carrier_uint {
    return ((cache.high() >> (64 - num_significand_bits<double>() - 2 - beta)) + 1) / 2;
  }
};

inline auto remove_trailing_zeros(uint32_t &n, int s = 0) noexcept -> int {
  constexpr uint32_t mod_inv_5 = 0xcccccccd;
  constexpr uint32_t mod_inv_25 = 0xc28f5c29;
  while (true) {
    auto q = rotr(n * mod_inv_25, 2);
    if (q > UINT32_MAX / 100)
      break;
    n = q;
    s += 2;
  }
  auto q = rotr(n * mod_inv_5, 1);
  if (q <= UINT32_MAX / 10) {
    n = q;
    s |= 1;
  }
  return s;
}

inline auto remove_trailing_zeros(uint64_t &n) noexcept -> int {
  constexpr uint32_t ten8 = 100000000u;
  if ((n % ten8) == 0) {
    auto n32 = static_cast<uint32_t>(n / ten8);
    int s = remove_trailing_zeros(n32, 8);
    n = n32;
    return s;
  }
  constexpr uint64_t mod_inv_5 = 0xcccccccccccccccd;
  constexpr uint64_t mod_inv_25 = 0x8f5c28f5c28f5c29;
  int s = 0;
  while (true) {
    auto q = rotr(n * mod_inv_25, 2);
    if (q > UINT64_MAX / 100)
      break;
    n = q;
    s += 2;
  }
  auto q = rotr(n * mod_inv_5, 1);
  if (q <= UINT64_MAX / 10) {
    n = q;
    s |= 1;
  }
  return s;
}

template <typename T>
auto is_left_endpoint_integer_shorter_interval(int exponent) noexcept -> bool {
  return exponent >= 2 && exponent <= 3;
}

template <typename T> inline auto shorter_interval_case(int exponent) noexcept -> decimal_fp<T> {
  decimal_fp<T> ret;
  const int minus_k = floor_log10_pow2_minus_log10_4_over_3(exponent);
  const int beta = exponent + floor_log2_pow10(-minus_k);

  using cache_entry_type = typename cache_accessor<T>::cache_entry_type;
  const cache_entry_type cache = cache_accessor<T>::get_cached_power(-minus_k);

  auto xi = cache_accessor<T>::compute_left_endpoint_for_shorter_interval_case(cache, beta);
  auto zi = cache_accessor<T>::compute_right_endpoint_for_shorter_interval_case(cache, beta);

  if (!is_left_endpoint_integer_shorter_interval<T>(exponent))
    ++xi;

  ret.significand = zi / 10;
  if (ret.significand * 10 >= xi) {
    ret.exponent = minus_k + 1;
    ret.exponent += remove_trailing_zeros(ret.significand);
    return ret;
  }

  ret.significand = cache_accessor<T>::compute_round_up_for_shorter_interval_case(cache, beta);
  ret.exponent = minus_k;

  if (exponent >= float_info<T>::shorter_interval_tie_lower_threshold &&
      exponent <= float_info<T>::shorter_interval_tie_upper_threshold) {
    ret.significand = ret.significand % 2 == 0 ? ret.significand : ret.significand - 1;
  } else if (ret.significand < xi) {
    ++ret.significand;
  }
  return ret;
}

template <typename T> auto to_decimal(T x) noexcept -> decimal_fp<T> {
  using carrier_uint = typename float_info<T>::carrier_uint;
  using cache_entry_type = typename cache_accessor<T>::cache_entry_type;
  auto br = __builtin_bit_cast(carrier_uint, x);

  const carrier_uint significand_mask =
      (static_cast<carrier_uint>(1) << num_significand_bits<T>()) - 1;
  carrier_uint significand = (br & significand_mask);
  int exponent = static_cast<int>((br & exponent_mask<T>()) >> num_significand_bits<T>());

  if (exponent != 0) {
    exponent -= exponent_bias<T>() + num_significand_bits<T>();
    if (significand == 0)
      return shorter_interval_case<T>(exponent);
    significand |= (static_cast<carrier_uint>(1) << num_significand_bits<T>());
  } else {
    if (significand == 0)
      return {0, 0};
    exponent = std::numeric_limits<T>::min_exponent - num_significand_bits<T>() - 1;
  }

  const bool include_left_endpoint = (significand % 2 == 0);
  const bool include_right_endpoint = include_left_endpoint;

  const int minus_k = floor_log10_pow2(exponent) - float_info<T>::kappa;
  const cache_entry_type cache = cache_accessor<T>::get_cached_power(-minus_k);
  const int beta = exponent + floor_log2_pow10(-minus_k);

  const uint32_t deltai = cache_accessor<T>::compute_delta(cache, beta);
  const carrier_uint two_fc = significand << 1;

  const typename cache_accessor<T>::compute_mul_result z_mul =
      cache_accessor<T>::compute_mul((two_fc | 1) << beta, cache);

  decimal_fp<T> ret;
  ret.significand = divide_by_10_to_kappa_plus_1(z_mul.result);
  uint32_t r = static_cast<uint32_t>(z_mul.result - float_info<T>::big_divisor * ret.significand);

  if (r < deltai) {
    if (r == 0 && (z_mul.is_integer & !include_right_endpoint)) {
      --ret.significand;
      r = float_info<T>::big_divisor;
      goto small_divisor;
    }
  } else if (r > deltai) {
    goto small_divisor;
  } else {
    const typename cache_accessor<T>::compute_mul_parity_result x_mul =
        cache_accessor<T>::compute_mul_parity(two_fc - 1, cache, beta);
    if (!(x_mul.parity | (x_mul.is_integer & include_left_endpoint)))
      goto small_divisor;
  }

  ret.exponent = minus_k + float_info<T>::kappa + 1;
  ret.exponent += remove_trailing_zeros(ret.significand);
  return ret;

small_divisor:
  ret.significand *= 10;
  ret.exponent = minus_k + float_info<T>::kappa;

  uint32_t dist = r - (deltai / 2) + (float_info<T>::small_divisor / 2);
  const bool approx_y_parity = ((dist ^ (float_info<T>::small_divisor / 2)) & 1) != 0;

  const bool divisible = check_divisibility_and_divide_by_pow10<float_info<T>::kappa>(dist);
  ret.significand += dist;
  if (!divisible)
    return ret;

  const auto y_mul = cache_accessor<T>::compute_mul_parity(two_fc, cache, beta);
  if (y_mul.parity != approx_y_parity)
    --ret.significand;
  else if (y_mul.is_integer & (ret.significand % 2 != 0))
    --ret.significand;
  return ret;
}

inline auto count_digits(uint64_t n) -> int {
  int count = 1;
  while (n >= 10) {
    n /= 10;
    ++count;
  }
  return count;
}

inline auto write_digits(char *buf, uint64_t n, int num_digits) -> char * {
  char *end = buf + num_digits;
  char *p = end;
  while (n >= 10) {
    *--p = '0' + static_cast<char>(n % 10);
    n /= 10;
  }
  *--p = '0' + static_cast<char>(n);
  return end;
}

constexpr auto use_fixed(int exp, int exp_upper) -> bool { return exp >= -4 && exp < exp_upper; }

template <typename T> inline auto format_shortest(char *buf, T value) -> int {
  if (value == T(0)) {
    buf[0] = '0';
    return 1;
  }

  auto dec = to_decimal(value);
  auto significand = static_cast<uint64_t>(dec.significand);
  int sig_size = count_digits(significand);
  int exponent = dec.exponent + sig_size - 1;

  constexpr int exp_upper =
      std::numeric_limits<T>::digits10 != 0
          ? (16 < std::numeric_limits<T>::digits10 + 1 ? 16 : std::numeric_limits<T>::digits10 + 1)
          : 16;

  char *p = buf;

  if (use_fixed(exponent, exp_upper)) {
    if (exponent >= 0) {
      int int_digits = exponent + 1;
      if (int_digits >= sig_size) {
        write_digits(p, significand, sig_size);
        p += sig_size;
        for (int i = 0; i < int_digits - sig_size; i++)
          *p++ = '0';
      } else {
        char tmp[20];
        write_digits(tmp, significand, sig_size);
        for (int i = 0; i < int_digits; i++)
          p[i] = tmp[i];
        p[int_digits] = '.';
        for (int i = int_digits; i < sig_size; i++)
          p[i + 1] = tmp[i];
        p += sig_size + 1;
      }
    } else {
      *p++ = '0';
      *p++ = '.';
      int leading_zeros = -(exponent + 1);
      for (int i = 0; i < leading_zeros; i++)
        *p++ = '0';
      write_digits(p, significand, sig_size);
      p += sig_size;
    }
  } else {
    char digits[20];
    write_digits(digits, significand, sig_size);
    *p++ = digits[0];
    if (sig_size > 1) {
      *p++ = '.';
      for (int i = 1; i < sig_size; i++)
        *p++ = digits[i];
    }
    *p++ = 'e';
    int abs_exp = exponent < 0 ? -exponent : exponent;
    *p++ = exponent < 0 ? '-' : '+';
    if (abs_exp >= 100) {
      *p++ = '0' + static_cast<char>(abs_exp / 100);
      abs_exp %= 100;
    }
    *p++ = '0' + static_cast<char>(abs_exp / 10);
    *p++ = '0' + static_cast<char>(abs_exp % 10);
  }
  return static_cast<int>(p - buf);
}

} // namespace dragonbox
#endif // FMT_SYCL_ACPP

// ============================================================
// fixed_string — compile-time string usable as NTTP
// ============================================================

template <size_t N> struct fixed_string {
  char data[N]{};

  constexpr fixed_string() = default;
  constexpr fixed_string(const char (&s)[N]) { std::copy_n(s, N, data); }

  constexpr char operator[](size_t i) const { return data[i]; }
};

template <size_t N> fixed_string(const char (&)[N]) -> fixed_string<N>;

// Length of a fixed_string (excluding null terminator)
template <size_t N> consteval size_t flen(const fixed_string<N> &) {
  static_assert(
      N >= 1,
      "Implementation Error: fixed_string must include null terminator. Something did go wrong");
  return N - 1;
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
  int index; // -1 = auto ({}), >=0 = positional ({N})
};

// Find the first {} or {:spec} or {N} or {N:spec} placeholder
template <fixed_string Fmt, size_t From = 0> consteval placeholder_info find_placeholder() {
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

      // Parse optional :spec — skip nested {} for dynamic width/prec
      bool has_spec = false;
      size_t spec_beg = j;
      if (j < len && Fmt[j] == ':') {
        has_spec = true;
        spec_beg = j + 1;
        j++;
        int depth = 0;
        while (j < len) {
          if (Fmt[j] == '{')
            depth++;
          else if (Fmt[j] == '}') {
            if (depth == 0)
              break;
            depth--;
          }
          j++;
        }
      }
      // j now points at '}' (or end)
      size_t close = j;
      if (!has_spec)
        spec_beg = close;
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
// Literal segment: unescape {{ → {, }} → }, and % → %%
// ============================================================

// Unified literal walker: unescape {{ → {, }} → }, and % → %%.
// When out == nullptr, counts output size; otherwise writes to out.
template <fixed_string Fmt, size_t Begin, size_t End>
consteval size_t walk_literal(char *out, size_t pos = 0) {
  size_t i = Begin;
  while (i < End) {
    if (i + 1 < End && Fmt[i] == '{' && Fmt[i + 1] == '{') {
      if (out)
        out[pos] = '{';
      pos++;
      i += 2;
    } else if (i + 1 < End && Fmt[i] == '}' && Fmt[i + 1] == '}') {
      if (out)
        out[pos] = '}';
      pos++;
      i += 2;
    } else if (Fmt[i] == '%') {
      if (out) {
        out[pos] = '%';
        out[pos + 1] = '%';
      }
      pos += 2;
      i++;
    } else {
      if (out)
        out[pos] = Fmt[i];
      pos++;
      i++;
    }
  }
  return pos;
}

template <fixed_string Fmt, size_t Begin, size_t End> consteval size_t literal_out_size() {
  return walk_literal<Fmt, Begin, End>(nullptr);
}

template <fixed_string Fmt, size_t Begin, size_t End> consteval auto make_literal() {
  constexpr size_t N = literal_out_size<Fmt, Begin, End>() + 1;
  fixed_string<N> result{};
  walk_literal<Fmt, Begin, End>(result.data);
  result.data[N - 1] = '\0';
  return result;
}

// Cast an integer to its printf-compatible type (int/long long or unsigned variants)
template <typename U> inline auto signed_int_cast(U arg) {
  if constexpr (sizeof(U) <= 4)
    return static_cast<int>(arg);
  else
    return static_cast<long long>(arg);
}

template <typename U> inline auto unsigned_int_cast(U arg) {
  if constexpr (sizeof(U) <= 4)
    return static_cast<unsigned>(arg);
  else
    return static_cast<unsigned long long>(arg);
}

// Types supported by sycl::khr::print
template <typename T>
concept sycl_printable = std::same_as<T, bool> || std::same_as<T, char> || std::integral<T> ||
                         std::floating_point<T> || std::is_pointer_v<T>;

// ============================================================
// Format spec parsing and printf format string generation
// ============================================================

consteval bool is_type_char(char c) {
  return c == 'd' || c == 'x' || c == 'X' || c == 'o' || c == 'b' || c == 'B' || c == 'f' ||
         c == 'F' || c == 'e' || c == 'E' || c == 'g' || c == 'G' || c == 'a' || c == 'A' ||
         c == 'c' || c == 's' || c == 'p';
}

consteval bool is_align_char(char c) { return c == '<' || c == '>' || c == '^'; }

consteval bool is_int_format(char c) {
  return c == 'd' || c == 'u' || c == 'x' || c == 'X' || c == 'o' || c == 'b' || c == 'B';
}

consteval bool is_float_format(char c) {
  return c == 'f' || c == 'F' || c == 'e' || c == 'E' || c == 'g' || c == 'G' || c == 'a' ||
         c == 'A';
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
  int width_arg = -1; // >=0: dynamic width from arg N, -1: static
  int prec_arg = -1;  // >=0: dynamic precision from arg N, -1: static
  int dyn_count = 0;  // number of auto-indexed dynamic args consumed

  constexpr char fill_or(char def = ' ') const { return fill ? fill : def; }
  constexpr char align_or(char def = '>') const { return align ? align : def; }
};

// Parse a dynamic arg reference {N} or {} inside a spec.
// Returns the arg index (or auto_idx if {}), advances i past '}'.
// Sets auto_idx to -1 after first manual use.
consteval int parse_dynamic_arg(const char *data, size_t len, size_t &i, int &auto_idx) {
  // i points at '{'
  i++; // skip '{'
  int idx;
  if (i < len && data[i] >= '0' && data[i] <= '9') {
    idx = 0;
    while (i < len && data[i] >= '0' && data[i] <= '9') {
      idx = idx * 10 + (data[i] - '0');
      i++;
    }
  } else {
    idx = auto_idx >= 0 ? auto_idx++ : 0;
  }
  if (i < len && data[i] == '}')
    i++; // skip '}'
  return idx;
}

template <fixed_string Fmt, size_t Begin, size_t End, int DynAutoStart = -1>
consteval format_spec parse_spec() {
  format_spec s{};
  size_t i = Begin;
  int dyn_auto = DynAutoStart; // auto-index counter for dynamic args

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

  // [width] — static digits or dynamic {N}/{}
  if (i < End && Fmt[i] == '{') {
    s.width_arg = parse_dynamic_arg(Fmt.data, End, i, dyn_auto);
  } else {
    while (i < End && Fmt[i] >= '0' && Fmt[i] <= '9') {
      s.width = s.width * 10 + (Fmt[i] - '0');
      i++;
    }
  }

  // [.precision] — static digits or dynamic {N}/{}
  if (i < End && Fmt[i] == '.') {
    i++;
    s.precision = 0;
    if (i < End && Fmt[i] == '{') {
      s.prec_arg = parse_dynamic_arg(Fmt.data, End, i, dyn_auto);
      s.precision = -1; // will be resolved at runtime
    } else {
      while (i < End && Fmt[i] >= '0' && Fmt[i] <= '9') {
        s.precision = s.precision * 10 + (Fmt[i] - '0');
        i++;
      }
    }
  }

  // [type]
  if (i < End && is_type_char(Fmt[i])) {
    s.type = Fmt[i];
  }

  // Count auto dynamic args consumed
  s.dyn_count = (dyn_auto >= 0 && DynAutoStart >= 0) ? (dyn_auto - DynAutoStart) : 0;

  return s;
}

// Effective printf type char given the spec and the C++ argument type
template <typename U, char SpecType> consteval char effective_type() {
  if (SpecType != '\0')
    return SpecType;
  if constexpr (std::same_as<U, bool>)
    return 's';
  else if constexpr (std::same_as<U, char>)
    return 'c';
  else if constexpr (std::floating_point<U>)
    return 'g';
  else if constexpr (std::signed_integral<U>)
    return 'd';
  else if constexpr (std::unsigned_integral<U>)
    return 'u';
  else if constexpr (std::is_pointer_v<U>) {
    using P = std::remove_cv_t<std::remove_pointer_t<U>>;
    if constexpr (std::same_as<P, char>)
      return 's';
    else
      return 'p';
  }
}

// ============================================================
// Device-side buffer (used by ACPP accumulator path)
// ============================================================

#ifndef KHR_SYCL_PRINT_BUFFER_SIZE
#define KHR_SYCL_PRINT_BUFFER_SIZE 255
#endif

struct fmt_buf {
  static constexpr int cap = KHR_SYCL_PRINT_BUFFER_SIZE;
  char data[cap + 1]{};
  int len = 0;
  void push(char c) {
    if (len < cap)
      data[len++] = c;
  }
  void push_n(char c, int n) {
    for (int i = 0; i < n && len < cap; i++)
      data[len++] = c;
  }
  void push_str(const char *s) {
    while (*s)
      push(*s++);
  }
};

inline void push_bool(fmt_buf &buf, bool val) {
  if (val)
    buf.push_str("true");
  else
    buf.push_str("false");
}

// Format unsigned integer into buffer in given base
template <int Base, bool Upper, typename U> inline void uint_to_buf(fmt_buf &buf, U val) {
  static_assert(Base == 2 || Base == 8 || Base == 10 || Base == 16,
                "uint_to_buf: Base must be 2, 8, 10, or 16");
  if (val == 0) {
    buf.push('0');
    return;
  }
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
  for (int i = n - 1; i >= 0; i--)
    buf.push(tmp[i]);
}

// Hex digit helper
inline char hex_digit(int d, bool upper) {
  if (d < 10)
    return static_cast<char>('0' + d);
  return static_cast<char>((upper ? 'A' : 'a') + d - 10);
}

// Build hex float content into buf — shared by DPC++ and ACPP paths.
template <format_spec Spec, char EffType, typename T>
inline void hex_float_to_buf(fmt_buf &content, T arg) {
  static_assert(EffType == 'a' || EffType == 'A', "hex_float_to_buf: EffType must be 'a' or 'A'");
  double val = static_cast<double>(arg);
  constexpr bool upper = (EffType == 'A');

  // Extract IEEE 754 bits (must do before sign check for -0.0)
  uint64_t bits = __builtin_bit_cast(uint64_t, val);
  bool negative = (bits >> 63) != 0;

  // Sign (use sign bit, not val < 0, to catch -0.0)
  if (negative) {
    content.push('-');
    val = -val;
    bits &= 0x7FFFFFFFFFFFFFFFULL;
  } else if (Spec.sign == '+')
    content.push('+');
  else if (Spec.sign == ' ')
    content.push(' ');

  // std::format {:a} never adds 0x prefix (unlike printf)
  // # only forces the decimal point (handled below)
  int biased_exp = static_cast<int>((bits >> 52) & 0x7FF); // sign already cleared
  uint64_t mantissa = bits & 0x000FFFFFFFFFFFFFULL;

  // Inf / NaN
  if (biased_exp == 0x7FF) {
    content.push_str((mantissa == 0) ? (upper ? "INF" : "inf") : (upper ? "NAN" : "nan"));
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
        if (((mantissa >> (48 - i * 4)) & 0xF) != 0)
          last_nz = i;
      for (int i = 0; i <= last_nz; i++)
        content.push(hex_digit(static_cast<int>((mantissa >> (48 - i * 4)) & 0xF), upper));
    }
  }

  // Exponent
  content.push(upper ? 'P' : 'p');
  if (exponent >= 0)
    content.push('+');
  else {
    content.push('-');
    exponent = -exponent;
  }
  if (exponent == 0) {
    content.push('0');
  } else {
    char tmp[10];
    int n = 0;
    while (exponent > 0) {
      tmp[n++] = static_cast<char>('0' + exponent % 10);
      exponent /= 10;
    }
    for (int i = n - 1; i >= 0; i--)
      content.push(tmp[i]);
  }
}

// Cast an arg to its printf-compatible type.
template <char EffType, typename T> inline auto printf_cast(T arg) {
  using U = std::decay_t<T>;
  if constexpr (std::same_as<U, bool> && EffType == 's')
    return arg ? "true" : "false";
  else if constexpr (EffType == 'c')
    return static_cast<char>(arg);
  else if constexpr (EffType == 'd') {
    return signed_int_cast(arg);
  } else if constexpr (EffType == 'u' || EffType == 'x' || EffType == 'X' || EffType == 'o') {
    return unsigned_int_cast(arg);
  } else if constexpr (is_float_format(EffType))
    return static_cast<double>(arg);
  else
    return arg;
}

#if !FMT_SYCL_ACPP
namespace specifiers_path {

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

template <fixed_string Lit> inline void emit_literal() {
  constexpr size_t len = flen(Lit);
  if constexpr (len > 0) {
    emit_literal_impl<Lit>(std::make_index_sequence<len>{});
  }
}

// Buffer for building printf format strings at compile time
struct printf_fmt_buf {
  char data[32]{};
  size_t len = 0;

  constexpr void push(char c) { data[len++] = c; }
  constexpr void push_int(int val) {
    if (val == 0) {
      push('0');
      return;
    }
    char tmp[10]{};
    int n = 0;
    while (val > 0) {
      tmp[n++] = '0' + val % 10;
      val /= 10;
    }
    for (int j = n - 1; j >= 0; j--)
      push(tmp[j]);
  }
};

// Build the printf format string: %[flags][width][.precision][length]type
template <format_spec Spec, char EffType, bool Is64> consteval printf_fmt_buf build_printf_fmt() {
  static_assert(is_type_char(EffType) || EffType == 'u',
                "build_printf_fmt: EffType must be a valid format type character");
  printf_fmt_buf buf;
  buf.push('%');

  // Flags
  if (Spec.align == '<')
    buf.push('-');
  if (Spec.sign == '+')
    buf.push('+');
  else if (Spec.sign == ' ')
    buf.push(' ');
  if (Spec.alt)
    buf.push('#');
  if (Spec.zero_pad && Spec.align != '<')
    buf.push('0');

  // Width
  if (Spec.width > 0)
    buf.push_int(Spec.width);

  // Precision
  if (Spec.precision >= 0) {
    buf.push('.');
    buf.push_int(Spec.precision);
  }

  // Length modifier for 64-bit integers
  bool is_int =
      (EffType == 'd' || EffType == 'u' || EffType == 'x' || EffType == 'X' || EffType == 'o');
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
// Atomic print — single ::sycl::ext::oneapi::experimental::printf call for the entire format
// ============================================================

// Check at compile time if a placeholder + arg type can use a standard
// printf specifier (i.e. does NOT need the buffer/dragonbox path).
template <typename U, format_spec Spec> consteval bool is_printf_compatible() {
  constexpr char etype = effective_type<U, Spec.type>();
  // Binary, hex-float, center, custom fill, signed hex/oct, alt hex
  if (etype == 'b' || etype == 'B')
    return false;
  if (etype == 'a' || etype == 'A')
    return false;
  if (Spec.align == '^')
    return false;
  if (Spec.fill != '\0' && Spec.fill != ' ')
    return false;
  constexpr bool is_signed_int = std::signed_integral<U>;
  if (is_signed_int && (etype == 'x' || etype == 'X' || etype == 'o'))
    return false;
  if (Spec.alt && (etype == 'x' || etype == 'X'))
    return false;
  // Dynamic width/precision
  if (Spec.width_arg >= 0 || Spec.prec_arg >= 0)
    return false;
  return true;
}

// Walk the entire format string at compile time. Return true if every
// placeholder can be handled by a single printf specifier.
template <fixed_string Fmt, size_t Pos = 0, size_t AutoIdx = 0, typename... Args>
consteval bool all_printf_compatible() {
#if FMT_SYCL_ACPP
  return false; // Always use print_impl on ACPP (format into buffers)
#endif
  constexpr auto info = find_placeholder<Fmt, Pos>();
  if constexpr (!info.found) {
    return true;
  } else {
    constexpr bool is_auto = (info.index < 0);
    constexpr size_t idx = is_auto ? AutoIdx : static_cast<size_t>(info.index);
    using U = std::decay_t<std::tuple_element_t<idx, std::tuple<Args...>>>;

    constexpr format_spec spec = (info.has_spec && info.close > info.spec_beg)
                                     ? parse_spec<Fmt, info.spec_beg, info.close>()
                                     : format_spec{};

    if constexpr (!is_printf_compatible<U, spec>()) {
      return false;
    } else {
      constexpr size_t next_auto = is_auto ? AutoIdx + 1 : AutoIdx;
      return all_printf_compatible<Fmt, info.close + 1, next_auto, Args...>();
    }
  }
}

// Unified walk: when out==nullptr, counts total size; otherwise writes.
// Handles literal segments and printf specifiers for each placeholder.
template <fixed_string Fmt, size_t Pos, size_t AutoIdx, typename... Args>
consteval size_t walk_combined_fmt(char *out, size_t pos = 0) {
  constexpr auto info = find_placeholder<Fmt, Pos>();
  if constexpr (!info.found) {
    return walk_literal<Fmt, Pos, flen(Fmt)>(out, pos);
  } else {
    pos = walk_literal<Fmt, Pos, info.open>(out, pos);
    constexpr bool is_auto = (info.index < 0);
    constexpr size_t idx = is_auto ? AutoIdx : static_cast<size_t>(info.index);
    using U = std::decay_t<std::tuple_element_t<idx, std::tuple<Args...>>>;
    constexpr format_spec spec = (info.has_spec && info.close > info.spec_beg)
                                     ? parse_spec<Fmt, info.spec_beg, info.close>()
                                     : format_spec{};
    constexpr char etype = effective_type<U, spec.type>();
    constexpr bool is_64 = sizeof(U) > 4;
    constexpr auto pfmt = build_printf_fmt<spec, etype, is_64>();
    for (size_t i = 0; i < pfmt.len; i++) {
      if (out)
        out[pos] = pfmt.data[i];
      pos++;
    }
    constexpr size_t next_auto = is_auto ? AutoIdx + 1 : AutoIdx;
    return walk_combined_fmt<Fmt, info.close + 1, next_auto, Args...>(out, pos);
  }
}

template <fixed_string Fmt, typename... Args> consteval auto build_combined_printf_fmt() {
  constexpr size_t N = walk_combined_fmt<Fmt, 0, 0, Args...>(nullptr) + 1;
  fixed_string<N> result{};
  walk_combined_fmt<Fmt, 0, 0, Args...>(result.data);
  result.data[N - 1] = '\0';
  return result;
}

// Unpack a tuple of printf-cast args and emit a single printf call.
template <fixed_string CombinedFmt, size_t... FmtIs, typename... CastArgs>
inline void emit_printf_impl(std::index_sequence<FmtIs...>, CastArgs... args) {
  static constexpr char s[] = {CombinedFmt.data[FmtIs]..., '\0'};
  ::sycl::ext::oneapi::experimental::printf(s, args...);
}

// Walk placeholders at compile time, accumulate printf-cast args in
// placeholder order (handles positional: "{0} {1} {0}" → a, b, a).
// Returns a tuple of all cast args.
template <fixed_string Fmt, size_t Pos, size_t AutoIdx, typename... Args, typename... CastArgs>
inline auto collect_printf_args(std::tuple<Args...> all_args, CastArgs... cast_args) {
  constexpr auto info = find_placeholder<Fmt, Pos>();
  if constexpr (!info.found) {
    return std::tuple(cast_args...);
  } else {
    constexpr bool is_auto = (info.index < 0);
    constexpr size_t idx = is_auto ? AutoIdx : static_cast<size_t>(info.index);
    using U = std::decay_t<std::tuple_element_t<idx, std::tuple<Args...>>>;
    constexpr format_spec spec = (info.has_spec && info.close > info.spec_beg)
                                     ? parse_spec<Fmt, info.spec_beg, info.close>()
                                     : format_spec{};
    constexpr char etype = effective_type<U, spec.type>();
    constexpr size_t next_auto = is_auto ? AutoIdx + 1 : AutoIdx;
    return collect_printf_args<Fmt, info.close + 1, next_auto, Args...>(
        all_args, cast_args..., printf_cast<etype>(std::get<idx>(all_args)));
  }
}

template <fixed_string CombinedFmt, typename Tuple, size_t... Is>
inline void emit_printf(Tuple &t, std::index_sequence<Is...>) {
  emit_printf_impl<CombinedFmt>(std::make_index_sequence<flen(CombinedFmt)>{}, std::get<Is>(t)...);
}

template <fixed_string Fmt, typename... Args> inline void print_combined_dispatch(Args... args) {
  constexpr auto combined = build_combined_printf_fmt<Fmt, Args...>();
  auto cast_args = collect_printf_args<Fmt, 0, 0, Args...>(std::tuple<Args...>(args...));
  emit_printf<combined>(cast_args,
                        std::make_index_sequence<std::tuple_size_v<decltype(cast_args)>>{});
}
} // namespace specifiers_path
#endif // !FMT_SYCL_ACPP

#if FMT_SYCL_ACPP
// ============================================================
// ACPP accumulator path
//
// Collects the entire formatted output into one fmt_buf, then
// flushes with a single sycl::detail::print call — atomic per
// work-item, no interleaving between format args.
// ============================================================

namespace buffer_path {

inline void write(fmt_buf &out, const char *s) {
  while (*s)
    out.push(*s++);
}

template <typename T> inline void write_decimal(fmt_buf &out, T val) {
  char tmp[22]{};
  int i = 20;
  using U = std::make_unsigned_t<T>;
  bool neg = false;
  U uval;
  if constexpr (std::signed_integral<T>) {
    neg = val < 0;
    uval = neg ? U(0) - static_cast<U>(val) : static_cast<U>(val);
  } else {
    uval = static_cast<U>(val);
  }
  if (uval == 0) {
    tmp[i--] = '0';
  } else {
    while (uval > 0) {
      tmp[i--] = '0' + static_cast<char>(uval % 10);
      uval /= 10;
    }
  }
  if (neg)
    tmp[i--] = '-';
  write(out, tmp + i + 1);
}

template <typename T> inline void write_arg_default(fmt_buf &out, T arg) {
  using U = std::decay_t<T>;
  if constexpr (std::same_as<U, bool>) {
    write(out, arg ? "true" : "false");
  } else if constexpr (std::same_as<U, char>) {
    out.push(arg);
  } else if constexpr (std::signed_integral<U> || std::unsigned_integral<U>) {
    write_decimal(out, arg);
  } else if constexpr (std::floating_point<U>) {
    // Dragonbox always available on ACPP (buffer guarantees atomicity).
    U val = arg;
    using bits_t = std::conditional_t<std::is_same_v<U, float>, uint32_t, uint64_t>;
    bool neg = __builtin_bit_cast(bits_t, val) >> (sizeof(bits_t) * 8 - 1);
    if (neg) {
      out.push('-');
      val = -val;
    }
    char dbuf[24];
    int dlen = dragonbox::format_shortest(dbuf, val);
    for (int k = 0; k < dlen; k++)
      out.push(dbuf[k]);
  } else if constexpr (std::is_pointer_v<U>) {
    using Pointee = std::remove_cv_t<std::remove_pointer_t<U>>;
    if constexpr (std::same_as<Pointee, char>)
      write(out, arg);
    else
      write(out, "<?p>"); // Level 4: pointer
  }
}

template <fixed_string Lit> inline void write_literal(fmt_buf &out) {
  constexpr size_t len = flen(Lit);
  if constexpr (len > 0) {
    for (size_t i = 0; i < len; i++)
      out.push(Lit[i]);
  }
}

// Append fmt_buf src into accumulator out.
inline void append_buf(fmt_buf &out, const fmt_buf &src) {
  for (int i = 0; i < src.len; i++)
    out.push(src.data[i]);
}

// Apply fill+alignment around content and append to out.
inline void apply_padding(fmt_buf &out, const fmt_buf &content, char fill, char align, int width) {
  int pad = width > content.len ? width - content.len : 0;
  if (pad == 0) {
    append_buf(out, content);
  } else if (align == '<') {
    append_buf(out, content);
    for (int i = 0; i < pad; i++)
      out.push(fill);
  } else if (align == '^') {
    for (int i = 0; i < pad / 2; i++)
      out.push(fill);
    append_buf(out, content);
    for (int i = pad / 2; i < pad; i++)
      out.push(fill);
  } else { // '>' (default)
    for (int i = 0; i < pad; i++)
      out.push(fill);
    append_buf(out, content);
  }
}

// Format a single integer argument with full spec support into out.
// Mirrors format_int_buf but targets the accumulator directly.
template <format_spec Spec, char EffType, typename T>
inline void write_int(fmt_buf &out, T arg, int width = Spec.width) {
  static_assert(is_int_format(EffType));
  using U = std::decay_t<T>;
  using Uns = std::conditional_t<(sizeof(U) <= 4), unsigned, unsigned long long>;
  constexpr int base = (EffType == 'b' || EffType == 'B')   ? 2
                       : (EffType == 'o')                   ? 8
                       : (EffType == 'x' || EffType == 'X') ? 16
                                                            : 10;
  constexpr bool upper = (EffType == 'X' || EffType == 'B');

  bool neg = false;
  Uns uval;
  if constexpr (std::same_as<U, bool>) {
    uval = static_cast<Uns>(arg);
  } else if constexpr (std::signed_integral<U>) {
    if (arg < 0) {
      neg = true;
      uval = Uns(0) - static_cast<Uns>(arg);
    } else
      uval = static_cast<Uns>(arg);
  } else {
    uval = static_cast<Uns>(arg);
  }

  char sc = neg ? '-' : (Spec.sign == '+') ? '+' : (Spec.sign == ' ') ? ' ' : '\0';

  char pfx[3] = {};
  int pfx_n = 0;
  if constexpr (Spec.alt) {
    if constexpr (base == 2) {
      pfx[0] = '0';
      pfx[1] = upper ? 'B' : 'b';
      pfx_n = 2;
    } else if constexpr (base == 16) {
      pfx[0] = '0';
      pfx[1] = upper ? 'X' : 'x';
      pfx_n = 2;
    } else if constexpr (base == 8) {
      if (uval != 0) {
        pfx[0] = '0';
        pfx_n = 1;
      }
    }
  }

  fmt_buf digits;
  uint_to_buf<base, upper>(digits, uval);

  int content_w = (sc ? 1 : 0) + pfx_n + digits.len;
  bool zpad = Spec.zero_pad && !Spec.fill && !Spec.align;

  fmt_buf content;
  if (sc)
    content.push(sc);
  for (int i = 0; i < pfx_n; i++)
    content.push(pfx[i]);
  if (zpad && width > content_w)
    content.push_n('0', width - content_w);
  for (int i = 0; i < digits.len; i++)
    content.push(digits.data[i]);

  apply_padding(out, content, Spec.fill_or(), Spec.align_or(), width);
}

// ── Float formatting helpers ──────────────────────────────────────────────────

// Format a non-negative finite double in fixed notation into buf.
// Handles up to prec=15 safely (uint64_t scale limit ~1e15 for val<1e4).
inline void fmt_fixed(fmt_buf &out, double val, int prec, bool alt = false) {
  // Compute scale = 10^prec
  double scale = 1.0;
  for (int i = 0; i < prec; i++)
    scale *= 10.0;

  // Round: add 0.5 then split
  double shifted = val * scale + 0.5;
  auto iscale = static_cast<uint64_t>(scale);
  auto total = static_cast<uint64_t>(shifted);
  uint64_t ipart = total / iscale;
  uint64_t frac = total % iscale;

  write_decimal(out, ipart);

  if (prec > 0 || alt) {
    out.push('.');
    if (prec > 0) {
      // Write fractional digits with leading zeros
      char fbuf[20]{};
      for (int i = prec - 1; i >= 0; i--) {
        fbuf[i] = '0' + static_cast<char>(frac % 10);
        frac /= 10;
      }
      for (int i = 0; i < prec; i++)
        out.push(fbuf[i]);
    }
  }
}

// Format a non-negative finite double in scientific notation into buf.
inline void fmt_sci(fmt_buf &out, double val, int prec, bool upper, bool alt = false) {
  int exp = 0;
  if (val == 0.0) {
    exp = 0;
  } else {
    double tmp = val;
    if (tmp >= 10.0) {
      while (tmp >= 10.0) {
        tmp /= 10.0;
        exp++;
      }
      val = tmp;
    } else if (tmp < 1.0) {
      while (tmp < 1.0) {
        tmp *= 10.0;
        exp--;
      }
      val = tmp;
    }
    // If rounding would make the mantissa overflow to 10 (e.g. 9.999... rounds
    // to 10.00000), detect it by pre-formatting and adjust before the real emit.
    {
      fmt_buf check;
      fmt_fixed(check, val, prec, false);
      if (check.len >= 2 && check.data[0] == '1' && check.data[1] == '0') {
        val /= 10.0;
        exp++;
      }
    }
  }
  fmt_fixed(out, val, prec, alt);
  out.push(upper ? 'E' : 'e');
  out.push(exp >= 0 ? '+' : '-');
  if (exp < 0)
    exp = -exp;
  if (exp < 10)
    out.push('0'); // at least 2 exponent digits
  write_decimal(out, static_cast<unsigned>(exp));
}

// Remove trailing zeros (and decimal point) from the fractional part of buf,
// stopping at 'e'/'E' if present (scientific notation).
inline void trim_trailing_zeros(fmt_buf &buf) {
  int dot_pos = -1;
  int e_pos = buf.len;
  for (int i = 0; i < buf.len; i++) {
    if (buf.data[i] == '.')
      dot_pos = i;
    else if (buf.data[i] == 'e' || buf.data[i] == 'E') {
      e_pos = i;
      break;
    }
  }
  if (dot_pos < 0)
    return;
  int trim = e_pos;
  while (trim > dot_pos + 1 && buf.data[trim - 1] == '0')
    trim--;
  if (trim == dot_pos + 1)
    trim = dot_pos; // remove dot itself
  int new_len = trim;
  for (int i = e_pos; i < buf.len; i++)
    buf.data[new_len++] = buf.data[i];
  buf.len = new_len;
  buf.data[new_len] = '\0';
}

// Format g/G: shortest of fixed/scientific, remove trailing zeros unless alt.
// prec = significant digits (default 6, min 1).
inline void fmt_g(fmt_buf &out, double val, int prec, bool upper, bool alt) {
  if (prec == 0)
    prec = 1;
  int exp = 0;
  if (val != 0.0) {
    double tmp = val;
    if (tmp >= 10.0) {
      while (tmp >= 10.0) {
        tmp /= 10.0;
        exp++;
      }
    } else if (tmp < 1.0) {
      while (tmp < 1.0) {
        tmp *= 10.0;
        exp--;
      }
    }
  }
  fmt_buf tmp_buf;
  if (exp >= -4 && exp < prec) {
    int f_prec = prec - (exp + 1);
    if (f_prec < 0)
      f_prec = 0;
    fmt_fixed(tmp_buf, val, f_prec, alt);
  } else {
    fmt_sci(tmp_buf, val, prec - 1, upper, alt);
  }
  if (!alt)
    trim_trailing_zeros(tmp_buf);
  append_buf(out, tmp_buf);
}

// Format a float/double argument with full spec (width, prec, sign, fill, zero-pad).
template <format_spec Spec, char EffType, typename T>
inline void write_float(fmt_buf &out, T arg, int dyn_w = Spec.width, int dyn_p = Spec.precision) {
  // Hex float: sign and content handled entirely by hex_float_to_buf
  if constexpr (EffType == 'a' || EffType == 'A') {
    fmt_buf content;
    hex_float_to_buf<Spec, EffType>(content, arg);
    apply_padding(out, content, Spec.fill_or(), Spec.align_or(), dyn_w);
    return;
  }

  constexpr bool upper = (EffType == 'F' || EffType == 'E' || EffType == 'G');
  double val = static_cast<double>(arg);
  int prec = dyn_p >= 0 ? dyn_p : (Spec.precision >= 0 ? Spec.precision : 6);

  // Extract sign from bits (handles -0.0)
  uint64_t bits = __builtin_bit_cast(uint64_t, val);
  bool neg = (bits >> 63) != 0;
  if (neg)
    val = -val;
  char sign_ch = neg ? '-' : (Spec.sign == '+') ? '+' : (Spec.sign == ' ') ? ' ' : '\0';

  // Build the numeric part (without sign)
  fmt_buf digits;
  int biased_exp = static_cast<int>((bits >> 52) & 0x7FF);
  if (biased_exp == 0x7FF) { // inf or nan
    uint64_t mant = bits & 0x000FFFFFFFFFFFFFULL;
    digits.push_str(mant == 0 ? (upper ? "INF" : "inf") : (upper ? "NAN" : "nan"));
  } else if (EffType == 'f' || EffType == 'F') {
    fmt_fixed(digits, val, prec, Spec.alt);
  } else if (EffType == 'e' || EffType == 'E') {
    fmt_sci(digits, val, prec, upper, Spec.alt);
  } else if (EffType == 'g' || EffType == 'G') {
    fmt_g(digits, val, prec, upper, Spec.alt);
  }

  // Assemble: sign + optional zero-fill + digits
  int content_w = (sign_ch ? 1 : 0) + digits.len;
  bool zpad = Spec.zero_pad && !Spec.fill && !Spec.align;

  fmt_buf content;
  if (sign_ch)
    content.push(sign_ch);
  if (zpad && dyn_w > content_w)
    content.push_n('0', dyn_w - content_w);
  append_buf(content, digits);

  apply_padding(out, content, Spec.fill_or(), Spec.align_or(), dyn_w);
}

// ── Dispatch with spec ────────────────────────────────────────────────────────

// Format one argument with an explicit format spec into out.
template <format_spec Spec, bool Dynamic = false, typename T>
inline void write_arg_with_spec(fmt_buf &out, T arg, int dyn_w = Spec.width,
                                int dyn_p = Spec.precision) {
  using U = std::decay_t<T>;

  // bool default/s → "true"/"false" with padding (default left-align like strings)
  if constexpr (std::same_as<U, bool> && (Spec.type == '\0' || Spec.type == 's')) {
    fmt_buf content;
    push_bool(content, arg);
    apply_padding(out, content, Spec.fill_or(), Spec.align_or('<'), dyn_w);
    return;
  }

  constexpr char etype = effective_type<U, Spec.type>();

  if constexpr (is_int_format(etype)) {
    write_int<Spec, etype>(out, arg, dyn_w);
  } else if constexpr (etype == 'c') {
    fmt_buf content;
    content.push(static_cast<char>(arg));
    apply_padding(out, content, Spec.fill_or(), Spec.align_or('<'), dyn_w);
  } else if constexpr (etype == 's') {
    fmt_buf content;
    const char *s = printf_cast<'s'>(arg);
    if (dyn_p >= 0) {
      for (int k = 0; k < dyn_p && s[k]; k++)
        content.push(s[k]);
    } else {
      content.push_str(s);
    }
    apply_padding(out, content, Spec.fill_or(), Spec.align_or('<'), dyn_w);
  } else if constexpr (is_float_format(etype)) {
    write_float<Spec, etype>(out, arg, dyn_w, dyn_p);
  } else {
    write(out, "<?>");
  }
}

template <fixed_string Fmt, placeholder_info Info, size_t AutoIdx, typename... Args>
inline void write_one_arg(fmt_buf &out, const std::tuple<Args...> &all_args) {
  constexpr size_t idx = Info.index >= 0 ? static_cast<size_t>(Info.index) : 0;
  auto arg = std::get<idx>(all_args);
  if constexpr (Info.has_spec && Info.close > Info.spec_beg) {
    constexpr int dyn_start = static_cast<int>(AutoIdx) + 1;
    constexpr auto spec = parse_spec<Fmt, Info.spec_beg, Info.close, dyn_start>();
    if constexpr (spec.width_arg >= 0 || spec.prec_arg >= 0) {
      int dyn_w = spec.width;
      int dyn_p = spec.precision;
      if constexpr (spec.width_arg >= 0) {
        constexpr size_t wi = static_cast<size_t>(spec.width_arg);
        dyn_w = static_cast<int>(std::get<wi>(all_args));
      }
      if constexpr (spec.prec_arg >= 0) {
        constexpr size_t pi = static_cast<size_t>(spec.prec_arg);
        dyn_p = static_cast<int>(std::get<pi>(all_args));
      }
      write_arg_with_spec<spec, true>(out, arg, dyn_w, dyn_p);
    } else {
      write_arg_with_spec<spec>(out, arg);
    }
  } else {
    write_arg_default(out, arg);
  }
}

template <fixed_string Fmt, size_t Pos, size_t AutoIdx, typename... Args>
inline void format(fmt_buf &out, const std::tuple<Args...> &all_args) {
  constexpr auto info = find_placeholder<Fmt, Pos>();
  if constexpr (!info.found) {
    constexpr size_t end = flen(Fmt);
    constexpr size_t sz = literal_out_size<Fmt, Pos, end>();
    if constexpr (sz > 0) {
      constexpr auto lit = make_literal<Fmt, Pos, end>();
      write_literal<lit>(out);
    }
  } else {
    constexpr size_t prefix_sz = literal_out_size<Fmt, Pos, info.open>();
    if constexpr (prefix_sz > 0) {
      constexpr auto prefix = make_literal<Fmt, Pos, info.open>();
      write_literal<prefix>(out);
    }
    constexpr bool is_auto = (info.index < 0);
    constexpr auto resolved = placeholder_info{
        info.open,     info.close, info.spec_beg,
        info.has_spec, info.found, is_auto ? static_cast<int>(AutoIdx) : info.index};
    write_one_arg<Fmt, resolved, AutoIdx>(out, all_args);

    constexpr int dyn_start = static_cast<int>(AutoIdx) + 1;
    constexpr int dyn_used = (info.has_spec && info.close > info.spec_beg)
                                 ? parse_spec<Fmt, info.spec_beg, info.close, dyn_start>().dyn_count
                                 : 0;
    constexpr size_t next_auto = is_auto ? AutoIdx + 1 + static_cast<size_t>(dyn_used) : AutoIdx;
    format<Fmt, info.close + 1, next_auto>(out, all_args);
  }
}
} // namespace buffer_path
#endif // FMT_SYCL_ACPP

} // namespace print_detail

// ============================================================
// Public API
// ============================================================

template <print_detail::fixed_string Fmt, print_detail::sycl_printable... Args>
inline void print(Args... args) {
#if FMT_SYCL_ACPP
  // Accumulate everything into one buffer, then one sycl::detail::print call.
  print_detail::fmt_buf out;
  print_detail::buffer_path::format<Fmt, 0, 0>(out, std::tuple<Args...>(args...));
  // sycl::detail::print wraps printf internally, so a literal % in out.data
  // would be misinterpreted as a format specifier.  Escape % → %% first.
  constexpr int esc_cap = KHR_SYCL_PRINT_BUFFER_SIZE * 2 + 1;
  char escaped[esc_cap];
  int elen = 0;
  for (int i = 0; i < out.len && elen + 2 < esc_cap; i++) {
    if (out.data[i] == '%')
      escaped[elen++] = '%';
    escaped[elen++] = out.data[i];
  }
  escaped[elen] = '\0';
  ::sycl::detail::print(escaped);
#else
  if constexpr (sizeof...(Args) == 0) {
    // No args — just emit the literal
    constexpr size_t end = print_detail::flen(Fmt);
    constexpr size_t out_sz = print_detail::literal_out_size<Fmt, 0, end>();
    if constexpr (out_sz > 0) {
      constexpr auto lit = print_detail::make_literal<Fmt, 0, end>();
      print_detail::specifiers_path::emit_literal<lit>();
    }
  } else {
    static_assert(print_detail::specifiers_path::all_printf_compatible<Fmt, 0, 0, Args...>(),
                  "This format string uses features not supported on DPC++ "
                  "({:b}, {:a}, {:^}, custom fill, {:#x} with signed int, "
                  "dynamic width/precision, dragonbox default float). "
                  "These features are only available on ACPP.");
    print_detail::specifiers_path::print_combined_dispatch<Fmt>(args...);
  }
#endif
}

namespace print_detail {
template <fixed_string Fmt> consteval auto append_newline() {
  constexpr size_t len = flen(Fmt);
  fixed_string<len + 2> result; // +1 for '\n', +1 for '\0'
  for (size_t i = 0; i < len; ++i)
    result.data[i] = Fmt[i];
  result.data[len] = '\n';
  result.data[len + 1] = '\0';
  return result;
}
} // namespace print_detail

template <print_detail::fixed_string Fmt, print_detail::sycl_printable... Args>
inline void println(Args... args) {
  print<print_detail::append_newline<Fmt>()>(args...);
}

} // namespace khr
#if !FMT_SYCL_ACPP
} // namespace _V1
#endif
} // namespace sycl

// Convenience macro — nicer syntax without explicit template angle brackets
#define KHR_PRINT(fmtstr, ...) ::sycl::khr::print<fmtstr>(__VA_ARGS__)
#define KHR_PRINTLN(fmtstr, ...) ::sycl::khr::println<fmtstr>(__VA_ARGS__)
