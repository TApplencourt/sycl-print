// fuzz.cpp — File-based fuzzer for fmt-sycl
//
// Build & run:
//   icpx -std=c++20 -O2 -DUSE_STD fuzz.cpp -o fuzz_std
//   icpx -fsycl -std=c++20 -O2 fuzz.cpp -o fuzz_sycl
//   ./fuzz_std  > /tmp/fuzz_std.txt
//   ONEAPI_DEVICE_SELECTOR=*:cpu ./fuzz_sycl > /tmp/fuzz_sycl.txt
//   diff /tmp/fuzz_std.txt /tmp/fuzz_sycl.txt && echo "PASS" || echo "FAIL"
//
// Both binaries use the same deterministic PRNG seed, so they produce
// identical random values. Any diff means a formatting mismatch.

#include <cstdint>
#include <climits>
#include <cmath>

#ifdef USE_STD
  #include <format>
  #include <iostream>
  #define P(fmt_str, ...) \
    std::cout << std::format(fmt_str __VA_OPT__(,) __VA_ARGS__)
#else
  #include "fmt_sycl.hpp"
  #define P(fmt_str, ...) \
    ::sycl::khr::print<fmt_str>(__VA_ARGS__)
#endif

// Simple deterministic PRNG (xorshift64)
static uint64_t rng_state = 0xDEADBEEFCAFE1234ULL;
static uint64_t rng() {
  rng_state ^= rng_state << 13;
  rng_state ^= rng_state >> 7;
  rng_state ^= rng_state << 17;
  return rng_state;
}
static int rand_int() { return static_cast<int>(rng()); }
static unsigned rand_uint() { return static_cast<unsigned>(rng()); }
static int64_t rand_i64() { return static_cast<int64_t>(rng()); }
static uint64_t rand_u64() { return rng(); }
static double rand_double() {
  // Build a double from bits to avoid FP computation differences between
  // compiler modes (-fsycl may use different FP model than host).
  uint64_t r1 = rng();
  uint64_t r2 = rng();
  // Exponent: bias 1023, range [1013..1033] → values in ~[2^-10, 2^10]
  uint64_t exp = 1013 + (r1 % 21);
  // Mantissa: 52 random bits
  uint64_t mant = r2 & 0x000FFFFFFFFFFFFFULL;
  // Sign: from r1
  uint64_t sign = (r1 >> 32) & 1;
  uint64_t bits = (sign << 63) | (exp << 52) | mant;
  return __builtin_bit_cast(double, bits);
}
static char rand_char() { return static_cast<char>(33 + rng() % 94); } // printable ASCII

// For each format spec, run N random values.
// The PRINT macro dispatches to either std::format or fmt::sycl::print.
#define N_ITER 200

// Macro to run a format spec over N random values of a given generator.
// GEN must be a callable returning the right type.
#define FUZZ_INT(spec, gen)          \
  for (int _i = 0; _i < N_ITER; _i++) { \
    auto _v = gen();                \
    P(spec, _v);                    \
  }

#define FUZZ_UINT(spec, gen)         \
  for (int _i = 0; _i < N_ITER; _i++) { \
    auto _v = gen();                \
    P(spec, _v);                    \
  }

#define FUZZ_DBL(spec)               \
  for (int _i = 0; _i < N_ITER; _i++) { \
    auto _v = rand_double();        \
    P(spec, _v);                    \
  }

#define FUZZ_CHAR(spec)              \
  for (int _i = 0; _i < N_ITER; _i++) { \
    auto _v = rand_char();          \
    P(spec, _v);                    \
  }

int main() {
  // ============================================================
  // 1. Default format — integers
  // ============================================================
  FUZZ_INT("{}\n", rand_int)
  FUZZ_INT("{}\n", rand_i64)
  FUZZ_UINT("{}\n", rand_uint)
  FUZZ_UINT("{}\n", rand_u64)

  // ============================================================
  // 2. Integer type specifiers
  // ============================================================
  FUZZ_INT("{:d}\n", rand_int)
  FUZZ_INT("{:x}\n", rand_int)
  FUZZ_INT("{:X}\n", rand_int)
  FUZZ_INT("{:o}\n", rand_int)
  FUZZ_INT("{:b}\n", rand_int)
  FUZZ_INT("{:B}\n", rand_int)
  FUZZ_UINT("{:x}\n", rand_uint)
  FUZZ_UINT("{:b}\n", rand_u64)

  // ============================================================
  // 3. Alternate form
  // ============================================================
  FUZZ_INT("{:#x}\n", rand_int)
  FUZZ_INT("{:#X}\n", rand_int)
  FUZZ_INT("{:#o}\n", rand_int)
  FUZZ_INT("{:#b}\n", rand_int)
  FUZZ_UINT("{:#x}\n", rand_uint)

  // ============================================================
  // 4. Sign
  // ============================================================
  FUZZ_INT("{:+d}\n", rand_int)
  FUZZ_INT("{: d}\n", rand_int)
  FUZZ_INT("{:+x}\n", rand_int)

  // ============================================================
  // 5. Width + alignment (integers)
  // ============================================================
  FUZZ_INT("{:20d}\n", rand_int)
  FUZZ_INT("{:<20d}\n", rand_int)
  FUZZ_INT("{:>20d}\n", rand_int)
  FUZZ_INT("{:^20d}\n", rand_int)
  FUZZ_INT("{:020d}\n", rand_int)
  FUZZ_INT("{:020x}\n", rand_int)

  // ============================================================
  // 6. Fill + alignment (integers)
  // ============================================================
  FUZZ_INT("{:*<20d}\n", rand_int)
  FUZZ_INT("{:*>20d}\n", rand_int)
  FUZZ_INT("{:*^20d}\n", rand_int)
  FUZZ_INT("{:#^20x}\n", rand_int)
  FUZZ_INT("{:*>20b}\n", rand_int)

  // ============================================================
  // 7. Combined integer specs
  // ============================================================
  FUZZ_INT("{:+020d}\n", rand_int)
  FUZZ_INT("{:#012x}\n", rand_int)
  FUZZ_INT("{:+#20x}\n", rand_int)
  FUZZ_INT("{:#020b}\n", rand_int)

  // ============================================================
  // 8. Float type specifiers
  // ============================================================
  FUZZ_DBL("{:f}\n")
  FUZZ_DBL("{:e}\n")
  FUZZ_DBL("{:E}\n")
  FUZZ_DBL("{:g}\n")
  FUZZ_DBL("{:G}\n")
  FUZZ_DBL("{:a}\n")
  FUZZ_DBL("{:A}\n")

  // ============================================================
  // 9. Float with precision
  // ============================================================
  FUZZ_DBL("{:.2f}\n")
  FUZZ_DBL("{:.0f}\n")
  FUZZ_DBL("{:.10f}\n")
  FUZZ_DBL("{:.4e}\n")
  FUZZ_DBL("{:.8g}\n")

  // ============================================================
  // 10. Float with width + precision
  // ============================================================
  FUZZ_DBL("{:20.4f}\n")
  FUZZ_DBL("{:<20.4f}\n")
  FUZZ_DBL("{:+20.4f}\n")
  FUZZ_DBL("{:020.4f}\n")

  // ============================================================
  // 11. Float with fill
  // ============================================================
  FUZZ_DBL("{:*>20.4f}\n")
  FUZZ_DBL("{:*<20.4f}\n")
  FUZZ_DBL("{:*^20.4f}\n")

  // ============================================================
  // 12. Char
  // ============================================================
  FUZZ_CHAR("{}\n")
  FUZZ_CHAR("{:c}\n")
  FUZZ_CHAR("{:d}\n")

  // ============================================================
  // 13. Bool
  // ============================================================
  for (int _i = 0; _i < N_ITER; _i++) {
    bool v = (rng() & 1) != 0;
    P("{}\n", v);
  }
  for (int _i = 0; _i < N_ITER; _i++) {
    bool v = (rng() & 1) != 0;
    P("{:d}\n", v);
  }
  for (int _i = 0; _i < N_ITER; _i++) {
    bool v = (rng() & 1) != 0;
    P("{:>10}\n", v);
  }
  for (int _i = 0; _i < N_ITER; _i++) {
    bool v = (rng() & 1) != 0;
    P("{:s}\n", v);
  }

  // ============================================================
  // 14. Hex float edge values
  // ============================================================
  FUZZ_DBL("{:a}\n")
  FUZZ_DBL("{:#a}\n")
  FUZZ_DBL("{:A}\n")
  FUZZ_DBL("{:20a}\n")

  // ============================================================
  // 15. Special float values mixed in
  // ============================================================
  P("{:f}\n", 1.0 / 0.0);
  P("{:f}\n", -1.0 / 0.0);
  P("{:e}\n", 1.0 / 0.0);
  P("{:g}\n", 0.0 / 0.0);
  P("{:a}\n", 0.0);
  P("{:a}\n", -0.0);
  P("{:+a}\n", 0.0);
  P("{:a}\n", 1.0 / 0.0);
  P("{:A}\n", 0.0 / 0.0);

  // ============================================================
  // 16. Multiple args — same type
  // ============================================================
  for (int _i = 0; _i < N_ITER; _i++) {
    int a = rand_int(); int b = rand_int();
    P("{} + {} = {}\n", a, b, a + b);
  }

  // ============================================================
  // 17. Multiple args — mixed types
  // ============================================================
  for (int _i = 0; _i < N_ITER; _i++) {
    int i = rand_int();
    double f = rand_double();
    P("i={:+d} f={:.4f}\n", i, f);
  }
  for (int _i = 0; _i < N_ITER; _i++) {
    P("Hello {}, coucou {}\n", "world", rand_int());
  }
  for (int _i = 0; _i < N_ITER; _i++) {
    int i = rand_int();
    double d = rand_double();
    char c = rand_char();
    bool b = (rng() & 1) != 0;
    P("int={} dbl={:.3f} chr={} bool={}\n", i, d, c, b);
  }
  for (int _i = 0; _i < N_ITER; _i++) {
    unsigned u = rand_uint();
    int64_t big = rand_i64();
    P("u={:#x} big={:d}\n", u, big);
  }
  for (int _i = 0; _i < N_ITER; _i++) {
    int a = rand_int();
    double b = rand_double();
    char c = rand_char();
    P("{:*>15d} | {:<+15.2f} | {:^5c}\n", a, b, c);
  }
  for (int _i = 0; _i < N_ITER; _i++) {
    int v = rand_int();
    P("hex={:#010x} bin={:#020b} dec={:+d}\n", v, v, v);
  }
  for (int _i = 0; _i < N_ITER; _i++) {
    bool b1 = (rng() & 1) != 0;
    bool b2 = (rng() & 1) != 0;
    int i = rand_int();
    P("{} and {} => {:d}\n", b1, b2, i);
  }
  for (int _i = 0; _i < N_ITER; _i++) {
    double d = rand_double();
    int i = rand_int();
    uint64_t u = rand_u64();
    char c = rand_char();
    bool b = (rng() & 1) != 0;
    P("{:.2e} {} {:x} {} {}\n", d, i, u, c, b);
  }
  for (int _i = 0; _i < N_ITER; _i++) {
    int v1 = rand_int(); double v2 = rand_double(); char v3 = rand_char();
    bool v4 = (rng() & 1) != 0; unsigned v5 = rand_uint();
    int64_t v6 = rand_i64();
    P("{} {} {} {} {} {} {}\n", v1, v2, v3, v4, v5, v6, "lit");
  }

  // ============================================================
  // 18. Escaped braces with mixed types
  // ============================================================
  for (int _i = 0; _i < N_ITER; _i++) {
    int v = rand_int();
    P("{{{:d}}}\n", v);
  }
  for (int _i = 0; _i < N_ITER; _i++) {
    double d = rand_double();
    int i = rand_int();
    P("result={{{}}} value={{{:+.2f}}}\n", i, d);
  }

  return 0;
}
