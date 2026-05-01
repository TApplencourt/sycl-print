// fuzz.cpp — Deterministic fuzzer for fmt-sycl
// Single binary: runs each format spec through both std::format and
// sycl::ext::khx::print with identical PRNG seeds, then diffs the output.
// The SYCL path submits kernels via parallel_for, like the other tests.

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
  for (int _i = 0; _i < N_ITER; _i++) { auto _v = gen(); RUN(P(spec, _v)); }
#define FUZZ_UINT(spec, gen) \
  for (int _i = 0; _i < N_ITER; _i++) { auto _v = gen(); RUN(P(spec, _v)); }
#define FUZZ_DBL(spec) \
  for (int _i = 0; _i < N_ITER; _i++) { auto _v = rand_double(); RUN(P(spec, _v)); }
#define FUZZ_CHAR(spec) \
  for (int _i = 0; _i < N_ITER; _i++) { auto _v = rand_char(); RUN(P(spec, _v)); }

#if defined(FMT_SYCL_HOST) || defined(FMT_SYCL_HOST_ACPP)
bool test_fuzz() {
  rng_state = SEED;
  auto std_out = capture_stdout([&]() {
#define FMT_STD_PATH
#define P(fmt_str, ...) std::cout << std::format(fmt_str __VA_OPT__(,) __VA_ARGS__)
#define RUN(...) for (int _r = 0; _r < N; _r++) { __VA_ARGS__; }
#include FUZZ_BODY
#undef RUN
#undef P
#undef FMT_STD_PATH
  });

  rng_state = SEED;
  auto sycl_out = capture_stdout([&]() {
#define P(fmt_str, ...) KHX_PRINT(fmt_str __VA_OPT__(,) __VA_ARGS__)
#define RUN(...) for (int _r = 0; _r < N; _r++) { __VA_ARGS__; }
#include FUZZ_BODY
#undef RUN
#undef P
  });

  return diff_output("fuzz.cpp", std_out, sycl_out);
}
#ifndef TEST_NO_MAIN
int main() { return test_fuzz() ? 0 : 1; }
#endif
#else
int main() {
  ::sycl::queue q;

  rng_state = SEED;
  auto std_out = capture_stdout([&]() {
#define FMT_STD_PATH
#define P(fmt_str, ...) std::cout << std::format(fmt_str __VA_OPT__(,) __VA_ARGS__)
#define RUN(...) for (int _r = 0; _r < N; _r++) { __VA_ARGS__; }
#include FUZZ_BODY
#undef RUN
#undef P
#undef FMT_STD_PATH
  });

  rng_state = SEED;
  auto sycl_out = capture_stdout([&]() {
#define P(fmt_str, ...) KHX_PRINT(fmt_str __VA_OPT__(,) __VA_ARGS__)
#define RUN(...) q.parallel_for(N, [=](::sycl::id<1>) { __VA_ARGS__; }).wait()
#include FUZZ_BODY
#undef RUN
#undef P
  });

  return diff_output("fuzz.cpp", std_out, sycl_out) ? 0 : 1;
}
#endif

#else

// ── Fuzz body (included twice with different P/RUN definitions) ──

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

FUZZ_CHAR("{:d}\n")

#ifndef FMT_SYCL_WA_STR
for (int _i = 0; _i < N_ITER; _i++) {
  bool v = (rng() & 1) != 0;
  RUN(P("{}\n", v));
}
#endif
for (int _i = 0; _i < N_ITER; _i++) {
  bool v = (rng() & 1) != 0;
  RUN(P("{:d}\n", v));
}
#ifndef FMT_SYCL_WA_STR
for (int _i = 0; _i < N_ITER; _i++) {
  bool v = (rng() & 1) != 0;
  RUN(P("{:>10}\n", v));
}
for (int _i = 0; _i < N_ITER; _i++) {
  bool v = (rng() & 1) != 0;
  RUN(P("{:s}\n", v));
}
#endif

RUN(P("{:f}\n", 1.0 / 0.0));
RUN(P("{:f}\n", -1.0 / 0.0));
RUN(P("{:e}\n", 1.0 / 0.0));
RUN(P("{:g}\n", 0.0 / 0.0));

for (int _i = 0; _i < N_ITER; _i++) {
  int a = rand_int(); int b = rand_int();
  int sum = static_cast<int>(static_cast<unsigned>(a) + static_cast<unsigned>(b));
  RUN(P("{} + {} = {}\n", a, b, sum));
}

for (int _i = 0; _i < N_ITER; _i++) {
  int i = rand_int();
  double f = rand_double();
  RUN(P("i={:+d} f={:.4f}\n", i, f));
}
#ifndef FMT_SYCL_WA_STR
for (int _i = 0; _i < N_ITER; _i++) {
  auto _v = rand_int();
  RUN(P("Hello {}, coucou {}\n", "world", _v));
}
#endif
#ifndef FMT_SYCL_WA_STR
for (int _i = 0; _i < N_ITER; _i++) {
  bool b1 = (rng() & 1) != 0;
  bool b2 = (rng() & 1) != 0;
  int i = rand_int();
  RUN(P("{} and {} => {:d}\n", b1, b2, i));
}
#endif

for (int _i = 0; _i < N_ITER; _i++) {
  int v = rand_int();
  RUN(P("{{{:d}}}\n", v));
}
for (int _i = 0; _i < N_ITER; _i++) {
  double d = rand_double();
  int i = rand_int();
  RUN(P("result={{{}}} value={{{:+.2f}}}\n", i, d));
}

#if FMT_SYCL_ACPP
FUZZ_INT("{:x}\n", rand_int)
FUZZ_INT("{:X}\n", rand_int)
FUZZ_INT("{:o}\n", rand_int)
FUZZ_INT("{:b}\n", rand_int)
FUZZ_INT("{:B}\n", rand_int)
FUZZ_UINT("{:b}\n", rand_u64)
FUZZ_UINT("{:x}\n", rand_u64)
FUZZ_UINT("{:X}\n", rand_u64)
FUZZ_UINT("{:o}\n", rand_u64)
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
RUN(P("{:a}\n", 0.0));
RUN(P("{:a}\n", -0.0));
RUN(P("{:+a}\n", 0.0));
RUN(P("{:a}\n", 1.0 / 0.0));
RUN(P("{:A}\n", 0.0 / 0.0));
for (int _i = 0; _i < N_ITER; _i++) {
  unsigned u = rand_uint();
  int64_t big = rand_i64();
  RUN(P("u={:#x} big={:d}\n", u, big));
}
for (int _i = 0; _i < N_ITER; _i++) {
  int v = rand_int();
  RUN(P("hex={:#010x} bin={:#020b} dec={:+d}\n", v, v, v));
}

// Bool with width/alignment (covers ACPP bool padding path)
for (int _i = 0; _i < N_ITER; _i++) {
  bool v = (rng() & 1) != 0;
  RUN(P("{:>10s}\n", v));
}
for (int _i = 0; _i < N_ITER; _i++) {
  bool v = (rng() & 1) != 0;
  RUN(P("{:<10s}\n", v));
}
for (int _i = 0; _i < N_ITER; _i++) {
  bool v = (rng() & 1) != 0;
  RUN(P("{:^10s}\n", v));
}

// Dynamic width (covers resolve_int_arg path)
for (int _i = 0; _i < N_ITER; _i++) {
  int v = rand_int();
  int w = 5 + static_cast<int>(rng() % 20);
  RUN(P("{:{}d}\n", v, w));
}
for (int _i = 0; _i < N_ITER; _i++) {
  double v = rand_double();
  int w = 10 + static_cast<int>(rng() % 15);
  RUN(P("{:{}.4f}\n", v, w));
}

// Dynamic precision
for (int _i = 0; _i < N_ITER; _i++) {
  double v = rand_double();
  int p = static_cast<int>(rng() % 10);
  RUN(P("{:.{}f}\n", v, p));
}

#endif

#endif
