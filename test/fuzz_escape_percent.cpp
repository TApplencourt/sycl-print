// fuzz_escape_percent.cpp — Fuzz tests that can produce % in output.
// Separated because % escaping favors GPU (see sycl_khr_print.hpp comment).

#ifndef FUZZ_BODY
#define FUZZ_BODY "fuzz_escape_percent.cpp"

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

#define FUZZ_CHAR(spec) \
  for (int _i = 0; _i < N_ITER; _i++) { auto _v = rand_char(); RUN(P(spec, _v)); }

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
#define P(fmt_str, ...) KHR_PRINT(fmt_str __VA_OPT__(,) __VA_ARGS__)
#define RUN(...) q.parallel_for(N, [=](::sycl::id<1>) { __VA_ARGS__; }).wait()
#include FUZZ_BODY
#undef RUN
#undef P
  });

  return diff_output("fuzz_escape_percent.cpp", std_out, sycl_out) ? 0 : 1;
}

#else

// Tests that can produce % in output (rand_char range includes ASCII 37 '%')
FUZZ_CHAR("{}\n")
FUZZ_CHAR("{:c}\n")

for (int _i = 0; _i < N_ITER; _i++) {
  int i = rand_int();
  double d = rand_double();
  char c = rand_char();
  bool b = (rng() & 1) != 0;
  RUN(P("int={} dbl={:.3f} chr={} bool={}\n", i, d, c, b));
}
for (int _i = 0; _i < N_ITER; _i++) {
  double d = rand_double();
  int i = rand_int();
  uint64_t u = rand_u64();
  char c = rand_char();
  bool b = (rng() & 1) != 0;
  RUN(P("{:.2e} {} {:x} {} {}\n", d, i, u, c, b));
}

#ifdef FMT_SYCL_ACPP
for (int _i = 0; _i < N_ITER; _i++) {
  int a = rand_int();
  double b = rand_double();
  char c = rand_char();
  RUN(P("{:*>15d} | {:<+15.2f} | {:^5c}\n", a, b, c));
}
#endif

#ifndef FMT_SYCL_WA_STR
for (int _i = 0; _i < N_ITER; _i++) {
  int v1 = rand_int(); double v2 = rand_double(); char v3 = rand_char();
  bool v4 = (rng() & 1) != 0; unsigned v5 = rand_uint();
  int64_t v6 = rand_i64();
  RUN(P("{} {} {} {} {} {} {}\n", v1, v2, v3, v4, v5, v6, "lit"));
}
#endif

#endif
