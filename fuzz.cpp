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
    ::fmt::sycl::print<fmt_str>(__VA_ARGS__)
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
  // Generate a double in a useful range: [-1e6, 1e6]
  int64_t r = static_cast<int64_t>(rng() % 2000001) - 1000000;
  uint64_t frac = rng() % 1000000;
  return static_cast<double>(r) + static_cast<double>(frac) / 1000000.0;
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
  // 16. Multiple args
  // ============================================================
  for (int _i = 0; _i < N_ITER; _i++) {
    int a = rand_int();
    int b = rand_int();
    P("{} + {} = {}\n", a, b, a + b);
  }
  for (int _i = 0; _i < N_ITER; _i++) {
    int x = rand_int();
    double f = rand_double();
    P("i={:+d} f={:.4f}\n", x, f);
  }

  // ============================================================
  // 17. Escaped braces with values
  // ============================================================
  for (int _i = 0; _i < N_ITER; _i++) {
    int v = rand_int();
    P("{{{:d}}}\n", v);
  }

  return 0;
}
