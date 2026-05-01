// sycl_khx_print.hpp — std::format-like API for SYCL device kernels
//
// Compile-time converts "{}" / "{:spec}" format strings to printf format
// specifiers, then forwards to sycl::ext::oneapi::experimental::printf.
//
// Usage:
//   sycl::ext::khx::print<"{} + {} = {}">(a, b, c);
//   KHX_PRINT("{} + {} = {}", a, b, c);   // macro for nicer syntax

#pragma once

// Backend detection: DPC++ vs AdaptiveCpp vs host-only (coverage)
#if defined(FMT_SYCL_HOST_ACPP)
#define FMT_SYCL_ACPP 1
#include <cstdio>
#elif defined(FMT_SYCL_HOST)
#define FMT_SYCL_ACPP 0
#include <cstdio>
#elif defined(__ADAPTIVECPP__) || defined(__HIPSYCL__) || defined(__ACPP__)
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
namespace ext {
namespace khx {

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

inline constexpr uint64_t float_pow10_table[] = {
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

inline constexpr uint128 double_pow10_significands[] = {
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

inline constexpr uint64_t double_powers_of_5_64[] = {
    0x0000000000000001, 0x0000000000000005, 0x0000000000000019, 0x000000000000007d,
    0x0000000000000271, 0x0000000000000c35, 0x0000000000003d09, 0x000000000001312d,
    0x000000000005f5e1, 0x00000000001dcd65, 0x00000000009502f9, 0x0000000002e90edd,
    0x000000000e8d4a51, 0x0000000048c27395, 0x000000016bcc41e9, 0x000000071afd498d,
    0x0000002386f26fc1, 0x000000b1a2bc2ec5, 0x000003782dace9d9, 0x00001158e460913d,
    0x000056bc75e2d631, 0x0001b1ae4d6e2ef5, 0x000878678326eac9, 0x002a5a058fc295ed,
    0x00d3c21bcecceda1, 0x0422ca8b0a00a425, 0x14adf4b7320334b9};

template <> struct cache_accessor<float> {
  using carrier_uint = uint32_t;
  using cache_entry_type = uint64_t;

  static auto get_cached_power(int k) noexcept -> uint64_t {
    return float_pow10_table[k - float_info<float>::min_k];
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
    constexpr int compression_ratio = 27;

    int cache_index = (k - float_info<double>::min_k) / compression_ratio;
    int kb = cache_index * compression_ratio + float_info<double>::min_k;
    int offset = k - kb;

    uint128 base_cache = double_pow10_significands[cache_index];
    if (offset == 0)
      return base_cache;

    int alpha = floor_log2_pow10(kb + offset) - floor_log2_pow10(kb) - offset;

    uint64_t pow5 = double_powers_of_5_64[offset];
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
        write_digits(p, significand, sig_size);
        for (int i = sig_size - 1; i >= int_digits; i--)
          p[i + 1] = p[i];
        p[int_digits] = '.';
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
    write_digits(p, significand, sig_size);
    if (sig_size > 1) {
      for (int i = sig_size; i >= 2; i--)
        p[i] = p[i - 1];
      p[1] = '.';
      p += sig_size + 1;
    } else {
      p += 1;
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

// Forward decl — defined alongside the other runtime helpers below.
constexpr placeholder_info find_placeholder_rt(const char *s, int len, int from);

// NTTP wrapper: same algorithm, just calls the runtime version on Fmt.data.
// Both are evaluated at compile time when invoked from a consteval context.
template <fixed_string Fmt, size_t From = 0> consteval placeholder_info find_placeholder() {
  return find_placeholder_rt(Fmt.data, static_cast<int>(flen(Fmt)), static_cast<int>(From));
}

// ============================================================
// Literal segment: unescape {{ → {, }} → } (and optionally % → %%)
// ============================================================

// Unified literal walker: unescape {{ → {, }} → }.
// When EscapePercent=true (DPC++ path), also escapes % → %% for printf.
// When EscapePercent=false (ACPP buffer path), % is left as-is; the
// runtime loop in print() will escape all % at once after the buffer
// is fully assembled (covering both literals and argument values).
// When out == nullptr, counts output size; otherwise writes to out.
template <fixed_string Fmt, size_t Begin, size_t End, bool EscapePercent = true>
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
    } else if (EscapePercent && Fmt[i] == '%') {
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

template <fixed_string Fmt, size_t Begin, size_t End, bool EscapePercent = true>
consteval size_t literal_out_size() {
  return walk_literal<Fmt, Begin, End, EscapePercent>(nullptr);
}

template <fixed_string Fmt, size_t Begin, size_t End, bool EscapePercent = true>
consteval auto make_literal() {
  constexpr size_t N = literal_out_size<Fmt, Begin, End, EscapePercent>() + 1;
  fixed_string<N> result{};
  walk_literal<Fmt, Begin, End, EscapePercent>(result.data);
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

// Types supported by sycl::ext::khx::print
template <typename T>
concept sycl_printable = std::same_as<T, bool> || std::same_as<T, char> || std::integral<T> ||
                         std::floating_point<T> || std::is_pointer_v<T>;

} // namespace print_detail

// ============================================================
// Customization point: formatter<T>
// Users specialize sycl::ext::khx::formatter<T> to teach the
// library how to print a custom type. The specialization must
// expose a static `format(T)` returning a `formatted<Fmt, ...>`
// where Fmt is a compile-time format string and the values
// reduce, after recursive expansion, to sycl_printable types.
// ============================================================

template <print_detail::fixed_string Fmt, typename... Args>
struct formatted {
  static constexpr auto format_string = Fmt;
  std::tuple<Args...> values;
};

template <typename T> struct formatter; // primary, intentionally undefined

template <typename T>
concept has_formatter = requires(T v) {
  { formatter<std::decay_t<T>>::format(v) };
};

template <typename T>
concept sycl_formattable = print_detail::sycl_printable<T> || has_formatter<T>;

namespace print_detail {

// ============================================================
// Format spec parsing and printf format string generation
// ============================================================

constexpr bool is_type_char(char c) {
  return c == 'd' || c == 'x' || c == 'X' || c == 'o' || c == 'b' || c == 'B' || c == 'f' ||
         c == 'F' || c == 'e' || c == 'E' || c == 'g' || c == 'G' || c == 'a' || c == 'A' ||
         c == 'c' || c == 's' || c == 'p';
}

constexpr bool is_align_char(char c) { return c == '<' || c == '>' || c == '^'; }

constexpr bool is_int_format(char c) {
  return c == 'd' || c == 'u' || c == 'x' || c == 'X' || c == 'o' || c == 'b' || c == 'B';
}

constexpr bool is_float_format(char c) {
  return c == 'f' || c == 'F' || c == 'e' || c == 'E' || c == 'g' || c == 'G' || c == 'a' ||
         c == 'A';
}

// Parsed format spec: [[fill]align][sign][#][0][width][.precision][type]
struct format_spec {
  char fill = '\0';
  char align = '\0';
  char sign = '\0';
  char type = '\0';
  bool alt = false;
  bool zero_pad = false;
  uint8_t width = 0;
  int8_t precision = -1;
  int8_t width_arg = -1; // >=0: dynamic width from arg N, -1: static
  int8_t prec_arg = -1;  // >=0: dynamic precision from arg N, -1: static
  uint8_t dyn_count = 0; // number of auto-indexed dynamic args consumed

  constexpr char fill_or(char def = ' ') const { return fill ? fill : def; }
  constexpr char align_or(char def = '>') const { return align ? align : def; }
};

// Parse a dynamic arg reference {N} or {} inside a spec.
// Returns the arg index (or auto_idx if {}), advances i past '}'.
// Sets auto_idx to -1 after first manual use.
constexpr int parse_dynamic_arg(const char *data, size_t len, size_t &i, int &auto_idx) {
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

// Forward decl — defined alongside the other runtime helpers below.
constexpr format_spec parse_spec_rt(const char *data, int begin, int end,
                                    int dyn_auto_start);

// NTTP wrapper: same algorithm, just calls the runtime version on Fmt.data.
template <fixed_string Fmt, size_t Begin, size_t End, int DynAutoStart = -1>
consteval format_spec parse_spec() {
  return parse_spec_rt(Fmt.data, static_cast<int>(Begin), static_cast<int>(End),
                       DynAutoStart);
}

// Forward decl — defined alongside the other runtime helpers below.
template <typename U> constexpr char effective_type_rt(char spec_type);

// Effective printf type char given the spec and the C++ argument type.
// NTTP wrapper over effective_type_rt — same algorithm, no duplication.
template <typename U, char SpecType> consteval char effective_type() {
  return effective_type_rt<U>(SpecType);
}

// ============================================================
// Runtime placeholder + spec parsing
// ============================================================
// Both backends use these. The consteval NTTP wrappers above (find_placeholder,
// parse_spec) just call into these — same algorithm, no duplication. ACPP also
// uses them at runtime via print_string's consteval ctor and the inner walker.

// Find the first {} or {:spec} or {N} or {N:spec} placeholder in s[from..len).
constexpr placeholder_info find_placeholder_rt(const char *s, int len, int from) {
  int i = from;
  while (i < len) {
    if (s[i] == '{') {
      if (i + 1 < len && s[i + 1] == '{') { i += 2; continue; }
      int j = i + 1;
      int index = -1;
      if (j < len && s[j] >= '0' && s[j] <= '9') {
        index = 0;
        while (j < len && s[j] >= '0' && s[j] <= '9') {
          index = index * 10 + (s[j] - '0');
          j++;
        }
      }
      bool has_spec = false;
      int spec_beg = j;
      if (j < len && s[j] == ':') {
        has_spec = true;
        spec_beg = j + 1;
        j++;
        int depth = 0;
        while (j < len) {
          if (s[j] == '{') depth++;
          else if (s[j] == '}') { if (depth == 0) break; depth--; }
          j++;
        }
      }
      int close = j;
      if (!has_spec) spec_beg = close;
      return {static_cast<size_t>(i), static_cast<size_t>(close),
              static_cast<size_t>(spec_beg), has_spec, true, index};
    } else if (s[i] == '}') {
      if (i + 1 < len && s[i + 1] == '}') { i += 2; continue; }
      i++;
    } else {
      i++;
    }
  }
  return {0, 0, 0, false, false, -1};
}

// Runtime version of parse_spec — works on const char* instead of fixed_string NTTP
constexpr format_spec parse_spec_rt(const char *data, int begin, int end,
                                    int dyn_auto_start = -1) {
  format_spec s{};
  size_t i = static_cast<size_t>(begin);
  size_t e = static_cast<size_t>(end);
  int dyn_auto = dyn_auto_start;

  if (i + 1 < e && is_align_char(data[i + 1])) {
    s.fill = data[i]; s.align = data[i + 1]; i += 2;
  } else if (i < e && is_align_char(data[i])) {
    s.align = data[i]; i++;
  }
  if (i < e && (data[i] == '+' || data[i] == '-' || data[i] == ' ')) {
    s.sign = data[i]; i++;
  }
  if (i < e && data[i] == '#') { s.alt = true; i++; }
  if (i < e && data[i] == '0') { s.zero_pad = true; i++; }
  if (i < e && data[i] == '{') {
    s.width_arg = parse_dynamic_arg(data, e, i, dyn_auto);
  } else {
    while (i < e && data[i] >= '0' && data[i] <= '9') {
      s.width = s.width * 10 + (data[i] - '0'); i++;
    }
  }
  if (i < e && data[i] == '.') {
    i++; s.precision = 0;
    if (i < e && data[i] == '{') {
      s.prec_arg = parse_dynamic_arg(data, e, i, dyn_auto);
      s.precision = -1;
    } else {
      while (i < e && data[i] >= '0' && data[i] <= '9') {
        s.precision = s.precision * 10 + (data[i] - '0'); i++;
      }
    }
  }
  if (i < e && is_type_char(data[i])) { s.type = data[i]; }
  s.dyn_count = (dyn_auto >= 0 && dyn_auto_start >= 0) ? (dyn_auto - dyn_auto_start) : 0;
  return s;
}

// Runtime version of effective_type — spec_type is a runtime parameter
template <typename U> constexpr char effective_type_rt(char spec_type) {
  if (spec_type != '\0') return spec_type;
  if constexpr (std::same_as<U, bool>) return 's';
  else if constexpr (std::same_as<U, char>) return 'c';
  else if constexpr (std::floating_point<U>) return 'g';
  else if constexpr (std::signed_integral<U>) return 'd';
  else if constexpr (std::unsigned_integral<U>) return 'u';
  else if constexpr (std::is_pointer_v<U>) {
    using P = std::remove_cv_t<std::remove_pointer_t<U>>;
    if constexpr (std::same_as<P, char>) return 's';
    else return 'p';
  }
}

template <typename T>
consteval bool type_can_produce_pct() {
  using U = std::decay_t<T>;
  if constexpr (std::same_as<U, char>) return true;
  else if constexpr (std::is_pointer_v<U>)
    return std::same_as<std::remove_cv_t<std::remove_pointer_t<U>>, char>;
  else if constexpr (sycl_printable<U>) return false;
  else return true; // formatter args — be conservative; sub-string may contain '%'
}

#if FMT_SYCL_ACPP
// ============================================================
// print_string — consteval-validated format string for ACPP
// ============================================================

// Pre-parsed placeholder entry — populated at compile time, consumed at runtime.
struct ph_entry {
  uint8_t open;
  uint8_t close;
  int8_t arg_idx;
  bool has_spec;
  format_spec spec;
};

// print_string — consteval-validated format string for print("...", args...) syntax.
// Placeholders and specs are pre-parsed at compile time into phs[].
template <sycl_formattable... Args>
struct print_string {
  static constexpr int MAX_LEN = 256;
  static constexpr int MAX_PH = 16;
  char str[MAX_LEN]{};
  int len;
  ph_entry phs[MAX_PH]{};
  int ph_count = 0;
  bool needs_pct_escape = false;

  template <size_t N>
  consteval print_string(const char (&s)[N]) : len(static_cast<int>(N - 1)) {
    static_assert(N <= MAX_LEN, "format string too long");
    std::copy_n(s, N, str);
    int pos = 0, auto_idx = 0;
    constexpr int n_args = static_cast<int>(sizeof...(Args));
    while (true) {
      auto info = find_placeholder_rt(s, len, pos);
      if (!info.found) break;
      if (ph_count >= MAX_PH)
        throw "too many placeholders (max 16)";
      int arg_i = (info.index >= 0) ? info.index : auto_idx;
      if (arg_i < 0 || arg_i >= n_args)
        throw "format argument index out of range";
      ph_entry &e = phs[ph_count++];
      e.open = static_cast<uint8_t>(info.open);
      e.close = static_cast<uint8_t>(info.close);
      e.arg_idx = static_cast<int8_t>(arg_i);
      e.has_spec = info.has_spec && info.close > info.spec_beg;
      if (e.has_spec) {
        int dyn_auto = (info.index < 0) ? auto_idx + 1 : -1;
        e.spec = parse_spec_rt(s, static_cast<int>(info.spec_beg),
                               static_cast<int>(info.close), dyn_auto);
        if (e.spec.width_arg >= 0 && e.spec.width_arg >= n_args)
          throw "dynamic width argument index out of range";
        if (e.spec.prec_arg >= 0 && e.spec.prec_arg >= n_args)
          throw "dynamic precision argument index out of range";
        if (info.index < 0) auto_idx += 1 + e.spec.dyn_count;
      } else {
        if (info.index < 0) auto_idx++;
      }
      pos = static_cast<int>(info.close) + 1;
    }
    needs_pct_escape = (type_can_produce_pct<Args>() || ...);
    if (!needs_pct_escape) {
      for (int i = 0; i < len; i++) {
        if (str[i] == '%') { needs_pct_escape = true; break; }
      }
    }
  }
};
#endif // FMT_SYCL_ACPP

// ============================================================
// Device-side buffer (used by ACPP accumulator path)
// ============================================================

#ifndef KHX_SYCL_PRINT_BUFFER_SIZE
#define KHX_SYCL_PRINT_BUFFER_SIZE 128
#endif

template <int Cap, int ExtraPad = 0>
struct static_buf {
  static constexpr int cap = Cap;
  char data[Cap + ExtraPad]{};
  int len = 0;
  void push(char c) {
    if (len < Cap)
      data[len++] = c;
  }
  void push_n(char c, int n) {
    for (int i = 0; i < n && len < Cap; i++)
      data[len++] = c;
  }
  void push_data(const char *s, int n) {
    for (int i = 0; i < n && len < Cap; i++)
      data[len++] = s[i];
  }
  void push_str(const char *s) {
    while (*s)
      push(*s++);
  }
};

// Extra 32 bytes let dragonbox write directly into data[len] without a
// temporary buffer; len is clamped to cap afterwards.
using fmt_buf = static_buf<KHX_SYCL_PRINT_BUFFER_SIZE, 32>;

// Write an unsigned integer in any base into raw (data, len, cap) right-to-left.
template <int Base, bool Upper = false, typename U>
  requires (Base == 2 || Base == 8 || Base == 10 || Base == 16)
inline void write_uint_raw(char *data, int &len, int cap, U val) {
  if (val == 0) { if (len < cap) data[len++] = '0'; return; }
  int n = 0;
  for (U t = val; t > 0; t /= U(Base)) n++;
  int pos = len + n - 1;
  while (val > 0) {
    int d = static_cast<int>(val % U(Base));
    data[pos--] = Upper ? "0123456789ABCDEF"[d] : "0123456789abcdef"[d];
    val /= U(Base);
  }
  len += n;
  if (len > cap) len = cap;
}

template <int Base, bool Upper = false, typename U, typename Buf>
inline void write_uint_direct(Buf &buf, U val) {
  write_uint_raw<Base, Upper>(buf.data, buf.len, static_cast<int>(sizeof(buf.data)), val);
}

// Hex digit helper
inline char hex_digit(int d, bool upper) {
  if (d < 10)
    return static_cast<char>('0' + d);
  return static_cast<char>((upper ? 'A' : 'a') + d - 10);
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
  } else if constexpr (is_float_format(EffType)) {
#ifdef __OPTIMIZE__
    // At O1+, the host runtime enables DAZ/FTZ, so std::format treats float
    // subnormals as zero.  Match that on the device: without this, the float
    // is promoted to double (where the value is normal) and printf outputs a
    // non-zero result that disagrees with the host reference.
    if constexpr (std::same_as<U, float>) {
      auto bits = __builtin_bit_cast(uint32_t, arg);
      if ((bits & 0x7F800000u) == 0 && (bits & 0x007FFFFFu) != 0)
        return (bits & 0x80000000u) ? -0.0 : 0.0;
    }
#endif
    return static_cast<double>(arg);
  }
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
#ifdef FMT_SYCL_HOST
  ::printf("%s", s);
#else
  ::sycl::ext::oneapi::experimental::printf(s);
#endif
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
template <format_spec Spec, char EffType, bool Is64>
  requires (is_type_char(EffType) || EffType == 'u')
consteval printf_fmt_buf build_printf_fmt() {
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
#ifdef FMT_SYCL_HOST
  ::printf(s, args...);
#else
  ::sycl::ext::oneapi::experimental::printf(s, args...);
#endif
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

template <typename T, typename Buf> inline void write_decimal(Buf &out, T val) {
  using U = std::make_unsigned_t<T>;
  U uval;
  if constexpr (std::signed_integral<T>) {
    if (val < 0) {
      out.push('-');
      // Store in a typed variable: subtraction of two uint8_t/uint16_t promotes
      // to int, so passing the expression directly would deduce U=int instead of
      // the actual unsigned type, making t>0 false on a negative int.
      uval = U(0) - static_cast<U>(val);
    } else {
      uval = static_cast<U>(val);
    }
  } else {
    uval = static_cast<U>(val);
  }
  write_uint_direct<10>(out, uval);
}

template <typename T> inline void write_arg_default(fmt_buf &out, T arg) {
  using U = std::decay_t<T>;
  if constexpr (std::same_as<U, bool>) {
    out.push_str(arg ? "true" : "false");
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
    bits_t fbits = __builtin_bit_cast(bits_t, val);
    constexpr int exp_bits = std::is_same_v<U, float> ? 8 : 11;
    constexpr bits_t exp_mask = ((bits_t(1) << exp_bits) - 1) << (sizeof(bits_t) * 8 - 1 - exp_bits);
    constexpr bits_t mant_mask = (bits_t(1) << (sizeof(bits_t) * 8 - 1 - exp_bits)) - 1;
    if ((fbits & exp_mask) == exp_mask && (fbits & mant_mask) == 0) {
      out.push_str("inf");
    } else if ((fbits & exp_mask) == exp_mask) {
      out.push_str("nan");
    } else {
      out.len += dragonbox::format_shortest(out.data + out.len, val);
      if (out.len > fmt_buf::cap) out.len = fmt_buf::cap;
    }
  } else if constexpr (std::is_pointer_v<U>) {
    using Pointee = std::remove_cv_t<std::remove_pointer_t<U>>;
    if constexpr (std::same_as<Pointee, char>)
      out.push_str(arg);
    else
      out.push_str("<?p>");
  }
}


inline void apply_padding_data(fmt_buf &out, const char *data, int len,
                               char fill, char align, int width) {
  int pad = width > len ? width - len : 0;
  if (pad == 0) { out.push_data(data, len); }
  else if (align == '<') { out.push_data(data, len); out.push_n(fill, pad); }
  else if (align == '^') { out.push_n(fill, pad / 2); out.push_data(data, len); out.push_n(fill, pad - pad / 2); }
  else { out.push_n(fill, pad); out.push_data(data, len); }
}

// Pad/sign/zfill content already at out.data[content_start..out.len) in place.
// Final layout: [lpad fill][sign][prefix][zfill '0'][content][rpad fill].
// Avoids a stack temporary by shifting the already-written content right by
// `prepend` bytes, then filling the gap. Per-write `p < end` checks let the
// loops keep running past the buffer cap (matching push_n's silent-truncate).
inline void pad_in_place(fmt_buf &out, int content_start,
                         char sign_ch, const char *prefix, int prefix_n, int zfill,
                         char fill, char align, int width) {
  int content_len = out.len - content_start;
  int sign_n = sign_ch ? 1 : 0;
  int total = sign_n + prefix_n + zfill + content_len;
  int pad = width > total ? width - total : 0;
  int lpad = (align == '>') ? pad : (align == '^') ? pad / 2 : 0;
  int rpad = pad - lpad;
  int prepend = lpad + sign_n + prefix_n + zfill;

  if (prepend > 0) {
    int end = fmt_buf::cap;
    // Drop digits that wouldn't fit after the shift, then move what survives.
    int kept = content_len;
    if (content_start + prepend + kept > end) kept = end - content_start - prepend;
    if (kept < 0) kept = 0;
    for (int i = kept - 1; i >= 0; i--)
      out.data[content_start + prepend + i] = out.data[content_start + i];
    int p = content_start;
    for (int i = 0; i < lpad && p < end; i++) out.data[p++] = fill;
    if (sign_ch && p < end) out.data[p++] = sign_ch;
    for (int i = 0; i < prefix_n && p < end; i++) out.data[p++] = prefix[i];
    for (int i = 0; i < zfill && p < end; i++) out.data[p++] = '0';
    out.len = p + kept;
  }
  out.push_n(fill, rpad);
}

// ── Float formatting helpers ──────────────────────────────────────────────────

constexpr double pow10(int n) {
  double s = 1.0;
  for (int i = 0; i < n; i++) s *= 10.0;
  return s;
}

constexpr uint64_t ipow10(int n) {
  uint64_t s = 1;
  for (int i = 0; i < n; i++) s *= 10;
  return s;
}

inline int find_exponent(double val) {
  int exp = 0;
  if (val >= 10.0) {
    while (val >= 10.0) { val /= 10.0; exp++; }
  } else if (val < 1.0) {
    while (val < 1.0) { val *= 10.0; exp--; }
  }
  return exp;
}

// Round val*scale to nearest integer with correct midpoint handling.
// Non-template so __attribute__((target)) works; on x86-64 forces the FMA
// ISA so __builtin_fma compiles to a real vfmsub instruction instead of
// being decomposed into mul+sub (which folds to zero at -O2).
#ifdef __x86_64__
__attribute__((target("fma")))
#endif
inline uint64_t round_scaled(double val, double scale, double shifted) {
  auto total = static_cast<uint64_t>(shifted);
  double frac_part = shifted - static_cast<double>(total);
  if (frac_part > 0.5) {
    total++;
  } else if (frac_part == 0.5) {
    double err = __builtin_fma(val, scale, -shifted);
    if (err > 0)
      total++;
    else if (err == 0 && (total & 1) != 0)
      total++;
  }
  return total;
}

// Format a non-negative finite double in fixed notation into buf.
// Handles up to prec=15 safely (uint64_t scale limit ~1e15 for val<1e4).
template <typename Buf>
inline void fmt_fixed(Buf &out, double val, int prec, bool alt = false) {
  double scale = pow10(prec);
  uint64_t total = round_scaled(val, scale, val * scale);
  auto iscale = static_cast<uint64_t>(scale);
  uint64_t ipart = total / iscale;
  uint64_t frac = total % iscale;

  write_decimal(out, ipart);

  if (prec > 0 || alt) {
    out.push('.');
    if (prec > 0) {
      // prec is known, so advance len directly and write right-to-left.
      int pos = out.len + prec - 1;
      out.len += prec;
      for (int i = 0; i < prec; i++) {
        out.data[pos--] = '0' + static_cast<char>(frac % 10);
        frac /= 10;
      }
      if (out.len > Buf::cap) out.len = Buf::cap;
    }
  }
}

// Format a non-negative finite double in scientific notation into buf.
template <typename Buf>
inline void fmt_sci(Buf &out, double val, int prec, bool upper, bool alt = false) {
  int exp = 0;
  if (val == 0.0) {
    exp = 0;
  } else {
    double orig = val;
    exp = find_exponent(val);
    double tmp = val;
    if (exp > 0)
      for (int i = 0; i < exp; i++) tmp /= 10.0;
    else if (exp < 0)
      for (int i = 0; i < -exp; i++) tmp *= 10.0;
    val = tmp;
    // If rounding would make the mantissa overflow to 10 (e.g. 9.999... rounds
    // to 10.00000), detect and adjust.  For small |shift| use the original
    // un-normalized value to avoid accumulated error from repeated /10;
    // for extreme exponents (|shift|>22) check_scale would lose exactness or
    // overflow, so fall back to the normalized value which is accurate enough.
    {
      int shift = prec - exp;
      bool overflow = false;
      if (shift >= 0 && shift <= 22) {
        double check_scale = pow10(shift);
        uint64_t total = round_scaled(orig, check_scale, orig * check_scale);
        overflow = (total >= ipow10(prec + 1));
      } else {
        double scale = pow10(prec);
        uint64_t total = round_scaled(val, scale, val * scale);
        overflow = (total / static_cast<uint64_t>(scale) >= 10);
      }
      if (overflow) {
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

// Remove trailing zeros (and decimal point) from buf[start..len),
// stopping at 'e'/'E' if present (scientific notation).
template <typename Buf>
inline void trim_trailing_zeros(Buf &buf, int start = 0) {
  int dot_pos = -1;
  int e_pos = buf.len;
  for (int i = start; i < buf.len; i++) {
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
template <typename Buf>
inline void fmt_g(Buf &out, double val, int prec, bool upper, bool alt) {
  if (prec == 0)
    prec = 1;
  int exp = 0;
  if (val != 0.0)
    exp = find_exponent(val);
  bool use_fixed = (exp >= -4 && exp < prec);
  if (use_fixed && exp == prec - 1) {
    uint64_t rounded = static_cast<uint64_t>(val + 0.5);
    uint64_t t = rounded;
    int d = 0;
    do { d++; t /= 10; } while (t);
    if (d > exp + 1)
      use_fixed = false;
  }
  int start = out.len;
  if (use_fixed) {
    int f_prec = prec - (exp + 1);
    if (f_prec < 0)
      f_prec = 0;
    fmt_fixed(out, val, f_prec, alt);
  } else {
    fmt_sci(out, val, prec - 1, upper, alt);
  }
  if (!alt)
    trim_trailing_zeros(out, start);
}

// ============================================================
// Runtime-spec formatting (for print("...", args...) syntax)
// ============================================================

// hex_float_to_buf with runtime spec
template <typename T, typename Buf>
inline void hex_float_to_buf_rt(Buf &content, T arg, const format_spec &spec, bool upper) {
  double val = static_cast<double>(arg);
  uint64_t bits = __builtin_bit_cast(uint64_t, val);
  bool negative = (bits >> 63) != 0;
  if (negative) {
    content.push('-'); val = -val; bits &= 0x7FFFFFFFFFFFFFFFULL;
  } else if (spec.sign == '+') content.push('+');
  else if (spec.sign == ' ') content.push(' ');

  int biased_exp = static_cast<int>((bits >> 52) & 0x7FF);
  uint64_t mantissa = bits & 0x000FFFFFFFFFFFFFULL;
  if (biased_exp == 0x7FF) {
    content.push_str((mantissa == 0) ? (upper ? "INF" : "inf") : (upper ? "NAN" : "nan"));
    return;
  }
  int exponent;
  if (biased_exp == 0) {
    if (mantissa == 0) { content.push('0'); content.push(upper ? 'P' : 'p'); content.push('+'); content.push('0'); return; }
    content.push('0'); exponent = -1022;
  } else {
    content.push('1'); exponent = biased_exp - 1023;
  }
  if (mantissa != 0 || spec.alt) {
    content.push('.');
    if (mantissa != 0) {
      int last_nz = -1;
      for (int i = 0; i < 13; i++)
        if (((mantissa >> (48 - i * 4)) & 0xF) != 0) last_nz = i;
      for (int i = 0; i <= last_nz; i++)
        content.push(hex_digit(static_cast<int>((mantissa >> (48 - i * 4)) & 0xF), upper));
    }
  }
  content.push(upper ? 'P' : 'p');
  if (exponent >= 0) content.push('+');
  else { content.push('-'); exponent = -exponent; }
  if (exponent == 0) content.push('0');
  else write_uint_direct<10>(content, static_cast<unsigned>(exponent));
}

// write_int with runtime spec and etype — writes directly to out (no stack temp).
// Digits are written into out at content_start; pad_in_place then shifts them
// right to make room for sign/prefix/zfill/lpad. Saves the 68 B `dgt[68]`.
template <typename T>
inline void write_int_rt(fmt_buf &out, T arg, const format_spec &spec, char etype, int width) {
  using U = std::decay_t<T>;
  using Uns = std::conditional_t<(sizeof(U) <= 4), unsigned, unsigned long long>;

  bool neg = false;
  Uns uval;
  if constexpr (std::same_as<U, bool>) {
    uval = static_cast<Uns>(arg);
  } else if constexpr (std::signed_integral<U>) {
    if (arg < 0) { neg = true; uval = Uns(0) - static_cast<Uns>(arg); }
    else uval = static_cast<Uns>(arg);
  } else {
    uval = static_cast<Uns>(arg);
  }

  char sc = neg ? '-' : (spec.sign == '+') ? '+' : (spec.sign == ' ') ? ' ' : '\0';

  int base = (etype == 'b' || etype == 'B') ? 2 : (etype == 'o') ? 8
             : (etype == 'x' || etype == 'X') ? 16 : 10;
  bool upper = (etype == 'X' || etype == 'B');

  char pfx[3] = {};
  int pfx_n = 0;
  if (spec.alt) {
    if (base == 2)       { pfx[0] = '0'; pfx[1] = upper ? 'B' : 'b'; pfx_n = 2; }
    else if (base == 16) { pfx[0] = '0'; pfx[1] = upper ? 'X' : 'x'; pfx_n = 2; }
    else if (base == 8 && uval != 0) { pfx[0] = '0'; pfx_n = 1; }
  }

  int content_start = out.len;
  switch (base) {
    case 2:  write_uint_raw<2,  false>(out.data, out.len, fmt_buf::cap, uval); break;
    case 8:  write_uint_raw<8,  false>(out.data, out.len, fmt_buf::cap, uval); break;
    case 16: if (upper) write_uint_raw<16, true>(out.data, out.len, fmt_buf::cap, uval);
             else       write_uint_raw<16, false>(out.data, out.len, fmt_buf::cap, uval);
             break;
    default: write_uint_raw<10, false>(out.data, out.len, fmt_buf::cap, uval); break;
  }

  int dlen = out.len - content_start;
  int content_w = (sc ? 1 : 0) + pfx_n + dlen;
  bool zpad = spec.zero_pad && !spec.fill && !spec.align;
  int zfill = (zpad && width > content_w) ? width - content_w : 0;

  pad_in_place(out, content_start, sc, pfx, pfx_n, zfill,
               spec.fill_or(), spec.align_or(), width);
}

// write_float with runtime spec and etype — writes digits directly into out,
// then pad_in_place handles sign/zfill/alignment. No stack temp.
template <typename T>
inline void write_float_rt(fmt_buf &out, T arg, const format_spec &spec, char etype,
                           int dyn_w, int dyn_p) {
  bool upper = (etype == 'F' || etype == 'E' || etype == 'G' || etype == 'A');
  double val = static_cast<double>(arg);
  int prec = dyn_p >= 0 ? dyn_p : (spec.precision >= 0 ? spec.precision : 6);

  uint64_t bits = __builtin_bit_cast(uint64_t, val);
  bool neg = (bits >> 63) != 0;
  if (neg) val = -val;
  char sign_ch = neg ? '-' : (spec.sign == '+') ? '+' : (spec.sign == ' ') ? ' ' : '\0';

  int content_start = out.len;
  if (etype == 'a' || etype == 'A') {
    // hex_float_to_buf_rt emits its own sign (so we can include sign chars in
    // the padded content width). Suppress the outer sign to avoid duplication.
    sign_ch = '\0';
    hex_float_to_buf_rt(out, arg, spec, upper);
  } else {
    int biased_exp = static_cast<int>((bits >> 52) & 0x7FF);
    if (biased_exp == 0x7FF) {
      uint64_t mant = bits & 0x000FFFFFFFFFFFFFULL;
      out.push_str(mant == 0 ? (upper ? "INF" : "inf") : (upper ? "NAN" : "nan"));
    } else if (etype == 'f' || etype == 'F') {
      fmt_fixed(out, val, prec, spec.alt);
    } else if (etype == 'e' || etype == 'E') {
      fmt_sci(out, val, prec, upper, spec.alt);
    } else if (etype == 'g' || etype == 'G') {
      fmt_g(out, val, prec, upper, spec.alt);
    }
  }

  int dlen = out.len - content_start;
  int content_w = (sign_ch ? 1 : 0) + dlen;
  bool zpad = spec.zero_pad && !spec.fill && !spec.align;
  int zfill = (zpad && dyn_w > content_w) ? dyn_w - content_w : 0;

  pad_in_place(out, content_start, sign_ch, nullptr, 0, zfill,
               spec.fill_or(), spec.align_or(), dyn_w);
}

// Format one argument with runtime spec — dispatches based on type + etype.
// No fmt_buf temporaries for bool/char/string; writes directly to out.
template <typename T>
inline void write_arg_rt(fmt_buf &out, T arg, const format_spec &spec, int dyn_w, int dyn_p) {
  using U = std::decay_t<T>;

  if constexpr (std::same_as<U, bool>) {
    if (spec.type == '\0' || spec.type == 's') {
      const char *bs = arg ? "true" : "false";
      int blen = arg ? 4 : 5;
      apply_padding_data(out, bs, blen, spec.fill_or(), spec.align_or('<'), dyn_w);
      return;
    }
  }

  char etype = effective_type_rt<U>(spec.type);

  if (is_int_format(etype)) {
    if constexpr (std::integral<U>) write_int_rt(out, arg, spec, etype, dyn_w);
  } else if (etype == 'c') {
    if constexpr (std::integral<U>) {
      char ch = static_cast<char>(arg);
      apply_padding_data(out, &ch, 1, spec.fill_or(), spec.align_or('<'), dyn_w);
    }
  } else if (etype == 's') {
    if constexpr (std::same_as<U, bool>) {
      const char *bs = arg ? "true" : "false";
      int blen = arg ? 4 : 5;
      apply_padding_data(out, bs, blen, spec.fill_or(), spec.align_or('<'), dyn_w);
    } else if constexpr (std::is_pointer_v<U>) {
      using Pointee = std::remove_cv_t<std::remove_pointer_t<U>>;
      if constexpr (std::same_as<Pointee, char>) {
        const char *s = arg;
        int slen = 0;
        if (dyn_p >= 0) { for (; slen < dyn_p && s[slen]; slen++); }
        else { while (s[slen]) slen++; }
        apply_padding_data(out, s, slen, spec.fill_or(), spec.align_or('<'), dyn_w);
      }
    }
  } else if (is_float_format(etype)) {
    if constexpr (std::floating_point<U>) write_float_rt(out, arg, spec, etype, dyn_w, dyn_p);
  } else {
    out.push_str("<?>");
  }
}

// ============================================================
// Runtime pack dispatch + format loop
// ============================================================
// dispatch_pack folds over the args pack directly.
template <typename F, typename... Args>
inline void dispatch_pack(int idx, F &&fn, Args &&... args) {
  int i = 0;
  (((i++ == idx) ? (fn(args), void()) : void()), ...);
}

template <typename... Args>
inline void resolve_int_arg(int idx, int &out, Args... args) {
  dispatch_pack(idx,
    [&out](auto val) {
      if constexpr (std::integral<std::decay_t<decltype(val)>>)
        out = static_cast<int>(val);
    }, args...);
}

inline void write_literal_segment(fmt_buf &out, const char *str, int from, int to) {
  for (int i = from; i < to;) {
    if (i + 1 < to && str[i] == '{' && str[i + 1] == '{') { out.push('{'); i += 2; }
    else if (i + 1 < to && str[i] == '}' && str[i + 1] == '}') { out.push('}'); i += 2; }
    else out.push(str[i++]);
  }
}

// Forward decl: the literal-walking format loop is needed by dispatch_arg
// (for formatter inner sub-strings) but its definition wants dispatch_arg.
template <sycl_formattable... Args>
inline void format_lit_rt(fmt_buf &out, const char *fmt, int fmt_len, Args... args);

// Per-arg dispatch. Primitives use the spec-aware writers; formatter args
// recurse into format_lit_rt on the formatter's inner format string + values.
// Both branches are if-constexpr so the primitive path stays byte-identical.
template <typename... Args>
inline void dispatch_arg(fmt_buf &out, int idx, bool has_spec,
                         const format_spec &spec, int dyn_w, int dyn_p,
                         Args... args) {
  dispatch_pack(idx,
    [&]<typename T>(T arg) {
      if constexpr (sycl_printable<std::decay_t<T>>) {
        if (has_spec) write_arg_rt(out, arg, spec, dyn_w, dyn_p);
        else write_arg_default(out, arg);
      } else {
        // has_formatter<T> — specs/dynamic-width on custom args are not
        // supported on ACPP at runtime; the inner format literal is rewalked.
        auto inner = ::sycl::ext::khx::formatter<std::decay_t<T>>::format(arg);
        constexpr auto inner_fmt = decltype(inner)::format_string;
        std::apply(
            [&](auto... vs) {
              format_lit_rt(out, inner_fmt.data, static_cast<int>(flen(inner_fmt)), vs...);
            },
            inner.values);
      }
    }, args...);
}

// Top-level walker. Reuses the pre-parsed phs[] for the outer format string
// (so spec handling for primitives is unchanged) and falls through to
// dispatch_arg for both primitive and formatter args.
template <sycl_formattable... Args>
inline void format_rt(fmt_buf &out, const print_string<Args...> &ps, Args... args) {
  int pos = 0;
  for (int i = 0; i < ps.ph_count; i++) {
    const auto &e = ps.phs[i];
    write_literal_segment(out, ps.str, pos, e.open);
    int dyn_w = e.spec.width, dyn_p = e.spec.precision;
    if (e.spec.width_arg >= 0) resolve_int_arg(e.spec.width_arg, dyn_w, args...);
    if (e.spec.prec_arg >= 0)  resolve_int_arg(e.spec.prec_arg,  dyn_p, args...);
    dispatch_arg(out, e.arg_idx, e.has_spec, e.spec, dyn_w, dyn_p, args...);
    pos = e.close + 1;
  }
  write_literal_segment(out, ps.str, pos, ps.len);
}

// Inner walker for formatter sub-strings. These don't have a print_string
// (no compile-time pre-parse), so we re-parse with find_placeholder_rt.
// No spec/dyn-width support here — Stage 1 forbids them on custom args.
template <sycl_formattable... Args>
inline void format_lit_rt(fmt_buf &out, const char *fmt, int fmt_len, Args... args) {
  int pos = 0;
  int auto_idx = 0;
  format_spec empty{};
  while (true) {
    auto info = find_placeholder_rt(fmt, fmt_len, pos);
    if (!info.found) break;
    write_literal_segment(out, fmt, pos, static_cast<int>(info.open));
    int idx = (info.index >= 0) ? info.index : auto_idx++;
    dispatch_arg(out, idx, /*has_spec*/ false, empty, /*dyn_w*/ 0, /*dyn_p*/ -1, args...);
    pos = static_cast<int>(info.close) + 1;
  }
  write_literal_segment(out, fmt, pos, fmt_len);
}

// Flush buf: escape % → %% then output.
// ACPP SSCP __acpp_sscp_print has different backends:
//   PTX/CUDA: calls vprintf(msg, nullptr) — interprets % as format specifiers
//   Host/CPU: calls fputs(msg, stdout) — prints verbatim
// __acpp_sscp_is_host is false for BOTH backends in SSCP kernels, so we
// cannot distinguish at runtime. We escape % → %% here, which is correct
// for CUDA (vprintf renders %% as %). This breaks CPU SSCP (fputs prints
// %% literally), but we favor GPU correctness.
inline void flush_buf(fmt_buf &out, bool escape_pct = true) {
#ifdef FMT_SYCL_HOST_ACPP
  (void)escape_pct;
  out.data[out.len] = '\0';
  fputs(out.data, stdout);
#else
  if (__acpp_sscp_is_host) {
    out.data[out.len] = '\0';
    fputs(out.data, stdout);
  } else {
    if (escape_pct) {
      int pct = 0;
      for (int i = 0; i < out.len; i++)
        if (out.data[i] == '%') pct++;
      if (pct > 0) {
        int newlen = out.len + pct;
        if (newlen > (int)fmt_buf::cap) newlen = fmt_buf::cap;
        for (int src = out.len - 1, dst = newlen - 1; src >= 0 && dst >= 0; src--) {
          out.data[dst--] = out.data[src];
          if (out.data[src] == '%' && dst >= 0) out.data[dst--] = '%';
        }
        out.len = newlen;
      }
    }
    out.data[out.len] = '\0';
    __acpp_sscp_print(out.data);
  }
#endif
}

} // namespace buffer_path
#endif // FMT_SYCL_ACPP

} // namespace print_detail


// ============================================================
// formatter_expand — splice user `formatter<T>` results into
// the parent format string + arg pack at compile time. DPC++
// only: ACPP dispatches formatter args at runtime inside
// buffer_path::dispatch_arg, so this machinery would be dead
// weight there.
// ============================================================

#if !FMT_SYCL_ACPP
namespace print_detail {
namespace formatter_expand {

// Per-arg post-one-round tuple type: primitive stays as-is, formatter expands to its values.
// Specialize on the primitive vs custom case so the unused branch never instantiates
// formatter<T> for primitive T.
template <typename T, bool IsPrim = sycl_printable<T>> struct per_arg_round;
template <typename T> struct per_arg_round<T, true> {
  using type = std::tuple<T>;
};
template <typename T> struct per_arg_round<T, false> {
  using type = decltype(formatter<std::decay_t<T>>::format(std::declval<T>()).values);
};

template <typename... Args>
using expand_args_round_t =
    decltype(std::tuple_cat(std::declval<typename per_arg_round<Args>::type>()...));

// Inner format string of a custom formatter (only valid when has_formatter<T>).
template <typename T>
inline constexpr auto inner_format_string =
    decltype(formatter<std::decay_t<T>>::format(std::declval<T>()))::format_string;

template <typename... Args>
consteval bool all_primitive() {
  return (... && sycl_printable<std::decay_t<Args>>);
}

// Scan Fmt for any positional ({N}) placeholder.
template <fixed_string Fmt, size_t Pos = 0> consteval bool any_positional() {
  constexpr auto info = find_placeholder<Fmt, Pos>();
  if constexpr (!info.found) {
    return false;
  } else if constexpr (info.index >= 0) {
    return true;
  } else {
    return any_positional<Fmt, info.close + 1>();
  }
}

// Scan Fmt; for each placeholder mapping to a has_formatter arg, ensure no spec.
template <fixed_string Fmt, size_t Pos, size_t AutoIdx, typename... Args>
consteval bool no_spec_on_custom() {
  constexpr auto info = find_placeholder<Fmt, Pos>();
  if constexpr (!info.found) {
    return true;
  } else {
    using U = std::decay_t<std::tuple_element_t<AutoIdx, std::tuple<Args...>>>;
    if constexpr (has_formatter<U>) {
      if constexpr (info.has_spec && info.close > info.spec_beg) return false;
    }
    return no_spec_on_custom<Fmt, info.close + 1, AutoIdx + 1, Args...>();
  }
}

// Single-round splicer. out=nullptr → measure; non-null → write.
// For each placeholder: primitive → copy "{...}" verbatim;
// has_formatter → splice the inner fixed_string body in place of "{...}".
template <fixed_string Fmt, size_t Pos, size_t AutoIdx, typename... Args>
consteval size_t walk_expand(char *out, size_t op = 0) {
  constexpr size_t len = flen(Fmt);
  constexpr auto info = find_placeholder<Fmt, Pos>();
  if constexpr (!info.found) {
    if (out) {
      for (size_t i = Pos; i < len; i++) out[op + (i - Pos)] = Fmt[i];
    }
    return op + (len - Pos);
  } else {
    if (out) {
      for (size_t i = Pos; i < info.open; i++) out[op + (i - Pos)] = Fmt[i];
    }
    op += info.open - Pos;
    using U = std::decay_t<std::tuple_element_t<AutoIdx, std::tuple<Args...>>>;
    if constexpr (sycl_printable<U>) {
      if (out) {
        for (size_t i = info.open; i <= info.close; i++)
          out[op + (i - info.open)] = Fmt[i];
      }
      op += (info.close - info.open + 1);
    } else {
      constexpr auto inner = inner_format_string<U>;
      constexpr size_t inner_len = flen(inner);
      if (out) {
        for (size_t i = 0; i < inner_len; i++) out[op + i] = inner.data[i];
      }
      op += inner_len;
    }
    return walk_expand<Fmt, info.close + 1, AutoIdx + 1, Args...>(out, op);
  }
}

template <fixed_string Fmt, typename... Args>
consteval auto expand_format_one_round() {
  constexpr size_t N = walk_expand<Fmt, 0, 0, Args...>(nullptr) + 1; // +1 for '\0'
  fixed_string<N> result{};
  walk_expand<Fmt, 0, 0, Args...>(result.data);
  result.data[N - 1] = '\0';
  return result;
}

// Compile-time fixed-point on (Fmt, Args...).
template <fixed_string Fmt, int Depth, typename... Args>
consteval auto expand_format_full();

template <fixed_string Fmt, int Depth, typename Tup, size_t... Is>
consteval auto expand_format_full_apply(std::index_sequence<Is...>) {
  return expand_format_full<Fmt, Depth, std::tuple_element_t<Is, Tup>...>();
}

template <fixed_string Fmt, int Depth, typename... Args>
consteval auto expand_format_full() {
  static_assert(Depth < 8,
                "formatter expansion exceeded depth limit; check for cycles in formatter<T>");
  if constexpr (all_primitive<Args...>()) {
    return Fmt;
  } else {
    if constexpr (Depth == 0) {
      static_assert(!any_positional<Fmt>(),
                    "positional indices ({0}, {1}, ...) are not allowed in format strings "
                    "that contain custom-formatter args; use auto-indexed {} placeholders");
      static_assert(no_spec_on_custom<Fmt, 0, 0, std::decay_t<Args>...>(),
                    "format spec ({:...}) on a custom-formatter arg is not supported");
    }
    constexpr auto Fmt2 = expand_format_one_round<Fmt, std::decay_t<Args>...>();
    using NewTup = expand_args_round_t<std::decay_t<Args>...>;
    return expand_format_full_apply<Fmt2, Depth + 1, NewTup>(
        std::make_index_sequence<std::tuple_size_v<NewTup>>{});
  }
}

// Runtime per-arg expansion: primitive → tuple{a}; formatter → its .values tuple.
template <typename T>
constexpr auto per_arg_expand_rt(T arg) {
  if constexpr (sycl_printable<std::decay_t<T>>) {
    return std::tuple<std::decay_t<T>>{arg};
  } else {
    return formatter<std::decay_t<T>>::format(arg).values;
  }
}

// Runtime fixed-point: matches the type-level recursion of expand_format_full.
template <typename... Args>
constexpr auto expand_args_full_rt(Args... args) {
  if constexpr (all_primitive<Args...>()) {
    return std::tuple<std::decay_t<Args>...>{args...};
  } else {
    return std::apply([](auto... a) { return expand_args_full_rt(a...); },
                      std::tuple_cat(per_arg_expand_rt(args)...));
  }
}

// Forward to the public print<Fmt>(args...) by unpacking a tuple of expanded args.
// Kept as a free function (not a lambda) so Fmt2 can be used as an NTTP without
// the C++ "captureless lambda + non-type template param" pitfall that clang flags.
template <fixed_string Fmt2, typename Tup, size_t... Is>
inline void apply_print(Tup &t, std::index_sequence<Is...>);

} // namespace formatter_expand
} // namespace print_detail
#endif // !FMT_SYCL_ACPP

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


// ============================================================
// Public API
// ============================================================

#if !FMT_SYCL_ACPP

// DPC++ path: the format string must reach the consteval splicer as an NTTP,
// so the public entry point is a function template parameterized on it. The
// KHX_PRINT macro hides the angle-bracket call shape.
template <print_detail::fixed_string Fmt, sycl_formattable... Args>
inline void print(Args... args) {
  if constexpr ((print_detail::sycl_printable<std::decay_t<Args>> && ...)) {
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
  } else {
    // Formatter splicer — runs entirely at compile time, then forwards
    // through the primitive path with the expanded format + flattened args.
    constexpr auto Fmt2 =
        print_detail::formatter_expand::expand_format_full<Fmt, 0, std::decay_t<Args>...>();
    auto values = print_detail::formatter_expand::expand_args_full_rt(args...);
    print_detail::formatter_expand::apply_print<Fmt2>(
        values, std::make_index_sequence<std::tuple_size_v<decltype(values)>>{});
  }
}

template <print_detail::fixed_string Fmt, sycl_formattable... Args>
inline void println(Args... args) {
  print<print_detail::append_newline<Fmt>()>(args...);
}

#else // FMT_SYCL_ACPP

// ACPP path: keep the value-style public API
//   sycl::ext::khx::println("…", args…)
// for both primitive and formatter args. The format literal is captured by
// print_string's consteval ctor (so no NTTP at the call site). format_rt
// now dispatches both primitive and formatter args via if constexpr — the
// primitive path stays byte-identical, formatter args recurse into the
// inner walker on the formatter's sub-format-string.
template <sycl_formattable... Args>
inline void print(print_detail::print_string<std::type_identity_t<Args>...> ps, Args... args) {
  print_detail::fmt_buf out;
  print_detail::buffer_path::format_rt(out, ps, args...);
  print_detail::buffer_path::flush_buf(out, ps.needs_pct_escape);
}

template <sycl_formattable... Args>
inline void println(print_detail::print_string<std::type_identity_t<Args>...> ps, Args... args) {
  print_detail::fmt_buf out;
  print_detail::buffer_path::format_rt(out, ps, args...);
  out.push('\n');
  print_detail::buffer_path::flush_buf(out, ps.needs_pct_escape);
}

#endif // FMT_SYCL_ACPP

} // namespace khx
} // namespace ext
} // namespace sycl

// Definition of the apply_print helper forward-declared above.
// Lives outside the public API section because it calls back into
// sycl::ext::khx::print<Fmt2>(...) which must already be declared.
#if !FMT_SYCL_ACPP
namespace sycl::ext::khx::print_detail::formatter_expand {
template <::sycl::ext::khx::print_detail::fixed_string Fmt2, typename Tup, size_t... Is>
inline void apply_print(Tup &t, std::index_sequence<Is...>) {
  ::sycl::ext::khx::print<Fmt2>(std::get<Is>(t)...);
}
} // namespace sycl::ext::khx::print_detail::formatter_expand
#endif // !FMT_SYCL_ACPP

// ============================================================
// Built-in formatter specializations for SYCL types
// ============================================================
// Only enabled when <sycl/sycl.hpp> is in play (skips host-only coverage builds
// which include the header without SYCL).

#if !defined(FMT_SYCL_HOST) && !defined(FMT_SYCL_HOST_ACPP)

namespace sycl {
namespace ext {
namespace khx {
namespace print_detail {

// Build "{}", "{}x{}", "{}x{}x{}", ... with a chosen separator.
template <int N, char Sep> consteval auto make_sep_fmt() {
  static_assert(N >= 1, "make_sep_fmt requires N >= 1");
  // Output size: N copies of "{}" + (N-1) separators + '\0'
  constexpr size_t Sz = static_cast<size_t>(N) * 2 + (N - 1) + 1;
  fixed_string<Sz> r{};
  size_t p = 0;
  for (int i = 0; i < N; i++) {
    if (i > 0) r.data[p++] = Sep;
    r.data[p++] = '{';
    r.data[p++] = '}';
  }
  r.data[p] = '\0';
  return r;
}

// Build "({})", "({}, {})", "({}, {}, {})", ...
template <int N> consteval auto make_id_fmt() {
  static_assert(N >= 1, "make_id_fmt requires N >= 1");
  // "(" + N×"{}" + (N-1)×", " + ")" + '\0'
  constexpr size_t Sz = 1 + static_cast<size_t>(N) * 2 + (N - 1) * 2 + 1 + 1;
  fixed_string<Sz> r{};
  size_t p = 0;
  r.data[p++] = '(';
  for (int i = 0; i < N; i++) {
    if (i > 0) { r.data[p++] = ','; r.data[p++] = ' '; }
    r.data[p++] = '{';
    r.data[p++] = '}';
  }
  r.data[p++] = ')';
  r.data[p] = '\0';
  return r;
}

} // namespace print_detail

// sycl::range<N> -> "AxBxC"
template <int N>
struct formatter<::sycl::range<N>> {
  static constexpr auto format(::sycl::range<N> r) {
    return [&]<size_t... Is>(std::index_sequence<Is...>) {
      return formatted<print_detail::make_sep_fmt<N, 'x'>(),
                       std::decay_t<decltype(r[Is])>...>{ {r[Is]...} };
    }(std::make_index_sequence<N>{});
  }
};

// sycl::id<N> -> "(A, B, C)"
template <int N>
struct formatter<::sycl::id<N>> {
  static constexpr auto format(::sycl::id<N> id) {
    return [&]<size_t... Is>(std::index_sequence<Is...>) {
      return formatted<print_detail::make_id_fmt<N>(),
                       std::decay_t<decltype(id[Is])>...>{ {id[Is]...} };
    }(std::make_index_sequence<N>{});
  }
};

// sycl::item<N, WithOffset> -> "item(global=(...), range=...)" — recursively
// resolved through the formatters for sycl::id and sycl::range. The second
// template parameter exists in both DPC++ and ACPP; a single specialization
// covers both backends and both WithOffset values.
template <int N, bool WithOffset>
struct formatter<::sycl::item<N, WithOffset>> {
  static constexpr auto format(::sycl::item<N, WithOffset> it) {
    return formatted<print_detail::fixed_string{"item(global={}, range={})"},
                     ::sycl::id<N>, ::sycl::range<N>>{
      {it.get_id(), it.get_range()}
    };
  }
};

// sycl::nd_item<N> -> "nd_item(global=..., local=..., range=...)"
template <int N>
struct formatter<::sycl::nd_item<N>> {
  static constexpr auto format(::sycl::nd_item<N> nd) {
    return formatted<print_detail::fixed_string{"nd_item(global={}, local={}, range={})"},
                     ::sycl::id<N>, ::sycl::id<N>, ::sycl::range<N>>{
      {nd.get_global_id(), nd.get_local_id(), nd.get_global_range()}
    };
  }
};

} // namespace khx
} // namespace ext
} // namespace sycl

#endif // !FMT_SYCL_HOST && !FMT_SYCL_HOST_ACPP

// Convenience macro — nicer syntax without explicit template angle brackets
#if FMT_SYCL_ACPP
#define KHX_PRINT(fmtstr, ...) ::sycl::ext::khx::print(fmtstr __VA_OPT__(,) __VA_ARGS__)
#define KHX_PRINTLN(fmtstr, ...) ::sycl::ext::khx::println(fmtstr __VA_OPT__(,) __VA_ARGS__)
#else
#define KHX_PRINT(fmtstr, ...) ::sycl::ext::khx::print<fmtstr>(__VA_ARGS__)
#define KHX_PRINTLN(fmtstr, ...) ::sycl::ext::khx::println<fmtstr>(__VA_ARGS__)
#endif
