// fuzz.cpp — Deterministic fuzzer for fmt-sycl
// Single binary: runs each format spec through both std::format and
// sycl::khr::print with identical PRNG seeds, then diffs the output.

#ifndef FUZZ_BODY
#define FUZZ_BODY "fuzz.cpp"

#include "capture.hpp"

static constexpr uint64_t SEED = 0xDEADBEEFCAFE1234ULL;
static uint64_t rng_state;
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
  uint64_t r1 = rng();
  uint64_t r2 = rng();
  uint64_t exp = 1013 + (r1 % 21);
  uint64_t mant = r2 & 0x000FFFFFFFFFFFFFULL;
  uint64_t sign = (r1 >> 32) & 1;
  uint64_t bits = (sign << 63) | (exp << 52) | mant;
  return __builtin_bit_cast(double, bits);
}
static char rand_char() { return static_cast<char>(33 + rng() % 94); }

#define N_ITER 200

#define FUZZ_INT(spec, gen) \
  for (int _i = 0; _i < N_ITER; _i++) { auto _v = gen(); P(spec, _v); }
#define FUZZ_UINT(spec, gen) \
  for (int _i = 0; _i < N_ITER; _i++) { auto _v = gen(); P(spec, _v); }
#define FUZZ_DBL(spec) \
  for (int _i = 0; _i < N_ITER; _i++) { auto _v = rand_double(); P(spec, _v); }
#define FUZZ_CHAR(spec) \
  for (int _i = 0; _i < N_ITER; _i++) { auto _v = rand_char(); P(spec, _v); }

int main() {
  rng_state = SEED;
  auto std_out = capture_stdout([&]() {
#define FMT_STD_PATH
#define P(fmt_str, ...) std::cout << std::format(fmt_str __VA_OPT__(,) __VA_ARGS__)
#include FUZZ_BODY
#undef P
#undef FMT_STD_PATH
  });

  rng_state = SEED;
  auto sycl_out = capture_stdout([&]() {
#define P(fmt_str, ...) ::sycl::khr::print<fmt_str>(__VA_ARGS__)
#include FUZZ_BODY
#undef P
  });

  return diff_output("fuzz.cpp", std_out, sycl_out) ? 0 : 1;
}

#else

// ── Fuzz body (included twice with different P definitions) ──

FUZZ_INT("{}\n", rand_int)
FUZZ_INT("{}\n", rand_i64)
FUZZ_UINT("{}\n", rand_uint)
FUZZ_UINT("{}\n", rand_u64)

FUZZ_INT("{:d}\n", rand_int)
FUZZ_UINT("{:x}\n", rand_uint)
FUZZ_INT("{:+d}\n", rand_int)
FUZZ_INT("{: d}\n", rand_int)

FUZZ_INT("{:20d}\n", rand_int)
FUZZ_INT("{:<20d}\n", rand_int)
FUZZ_INT("{:>20d}\n", rand_int)
FUZZ_INT("{:020d}\n", rand_int)
FUZZ_INT("{:+020d}\n", rand_int)

FUZZ_DBL("{:f}\n")
FUZZ_DBL("{:e}\n")
FUZZ_DBL("{:E}\n")
FUZZ_DBL("{:g}\n")
FUZZ_DBL("{:G}\n")

FUZZ_DBL("{:.2f}\n")
FUZZ_DBL("{:.0f}\n")
FUZZ_DBL("{:.10f}\n")
FUZZ_DBL("{:.4e}\n")
FUZZ_DBL("{:.8g}\n")

FUZZ_DBL("{:20.4f}\n")
FUZZ_DBL("{:<20.4f}\n")
FUZZ_DBL("{:+20.4f}\n")
FUZZ_DBL("{:020.4f}\n")

FUZZ_CHAR("{}\n")
FUZZ_CHAR("{:c}\n")
FUZZ_CHAR("{:d}\n")

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

P("{:f}\n", 1.0 / 0.0);
P("{:f}\n", -1.0 / 0.0);
P("{:e}\n", 1.0 / 0.0);
P("{:g}\n", 0.0 / 0.0);

for (int _i = 0; _i < N_ITER; _i++) {
  int a = rand_int(); int b = rand_int();
  P("{} + {} = {}\n", a, b, a + b);
}

for (int _i = 0; _i < N_ITER; _i++) {
  int i = rand_int();
  double f = rand_double();
  P("i={:+d} f={:.4f}\n", i, f);
}
#ifndef FMT_SYCL_WA_STR
for (int _i = 0; _i < N_ITER; _i++) {
  P("Hello {}, coucou {}\n", "world", rand_int());
}
#endif
for (int _i = 0; _i < N_ITER; _i++) {
  int i = rand_int();
  double d = rand_double();
  char c = rand_char();
  bool b = (rng() & 1) != 0;
  P("int={} dbl={:.3f} chr={} bool={}\n", i, d, c, b);
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
  int v = rand_int();
  P("{{{:d}}}\n", v);
}
for (int _i = 0; _i < N_ITER; _i++) {
  double d = rand_double();
  int i = rand_int();
  P("result={{{}}} value={{{:+.2f}}}\n", i, d);
}

#ifdef FMT_SYCL_BUFFER_PATH_ONLY
FUZZ_INT("{:x}\n", rand_int)
FUZZ_INT("{:X}\n", rand_int)
FUZZ_INT("{:o}\n", rand_int)
FUZZ_INT("{:b}\n", rand_int)
FUZZ_INT("{:B}\n", rand_int)
FUZZ_UINT("{:b}\n", rand_u64)
FUZZ_INT("{:#x}\n", rand_int)
FUZZ_INT("{:#X}\n", rand_int)
FUZZ_INT("{:#o}\n", rand_int)
FUZZ_INT("{:#b}\n", rand_int)
FUZZ_UINT("{:#x}\n", rand_uint)
FUZZ_INT("{:+x}\n", rand_int)
FUZZ_INT("{:^20d}\n", rand_int)
FUZZ_INT("{:020x}\n", rand_int)
FUZZ_INT("{:*<20d}\n", rand_int)
FUZZ_INT("{:*>20d}\n", rand_int)
FUZZ_INT("{:*^20d}\n", rand_int)
FUZZ_INT("{:#^20x}\n", rand_int)
FUZZ_INT("{:*>20b}\n", rand_int)
FUZZ_INT("{:#012x}\n", rand_int)
FUZZ_INT("{:+#20x}\n", rand_int)
FUZZ_INT("{:#020b}\n", rand_int)
FUZZ_DBL("{:a}\n")
FUZZ_DBL("{:A}\n")
FUZZ_DBL("{:*>20.4f}\n")
FUZZ_DBL("{:*<20.4f}\n")
FUZZ_DBL("{:*^20.4f}\n")
FUZZ_DBL("{:#a}\n")
FUZZ_DBL("{:20a}\n")
P("{:a}\n", 0.0);
P("{:a}\n", -0.0);
P("{:+a}\n", 0.0);
P("{:a}\n", 1.0 / 0.0);
P("{:A}\n", 0.0 / 0.0);
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
#endif

#if !defined(FMT_SYCL_WA_STR) && defined(FMT_SYCL_BUFFER_PATH_ONLY)
for (int _i = 0; _i < N_ITER; _i++) {
  int v1 = rand_int(); double v2 = rand_double(); char v3 = rand_char();
  bool v4 = (rng() & 1) != 0; unsigned v5 = rand_uint();
  int64_t v6 = rand_i64();
  P("{} {} {} {} {} {} {}\n", v1, v2, v3, v4, v5, v6, "lit");
}
#endif

#endif
