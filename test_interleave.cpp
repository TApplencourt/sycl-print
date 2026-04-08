// test_interleave.cpp — Multi-work-item atomicity test for fmt-sycl
//
// Each println must produce a complete, non-interleaved line.
// All work-items print the same output, so we can diff against std.
//
// Build & diff:
//   icpx -std=c++20 -DUSE_STD test_interleave.cpp -o interleave_std
//   icpx -fsycl -std=c++20 test_interleave.cpp -o interleave_sycl
//   diff <(./interleave_std) <(ONEAPI_DEVICE_SELECTOR=*:cpu ./interleave_sycl 2>/dev/null)

#include <cstdint>

constexpr int N = 8;

#ifdef USE_STD
  #include <format>
  #include <iostream>
  #define TEST(fmt_str, ...) \
    for (int _i = 0; _i < N; _i++) \
      std::cout << std::format(fmt_str __VA_OPT__(,) __VA_ARGS__) << '\n'
#else
  #include "sycl_khr_print.hpp"
  #include <sycl/sycl.hpp>
  static ::sycl::queue _q;
  #define TEST(fmt_str, ...) \
    _q.parallel_for(N, [=](::sycl::id<1>) { \
      ::sycl::khr::println<fmt_str>(__VA_ARGS__); \
    }); \
    _q.wait()
#endif

int main() {
  // 1. Plain literal
  TEST("hello world");

  // 2. String with {}
  TEST("hello {}", "world");

  // 3. Integer
  TEST("val={}", 42);
  TEST("{} + {} = {}", 1, 2, 3);

  // 4. Mixed string + int
  TEST("{} has {} items", "cart", 7);

  // 5. Default float
  TEST("pi={}", 3.14f);

  // 6. Bool
  TEST("flag={}", true);
  TEST("a={} b={}", true, false);

  // 7. Mixed string + int + float + bool
  TEST("name={} age={} score={} ok={}", "alice", 30, 9.5f, true);

  // 8. Width and alignment
  TEST("{:10d}", 42);
  TEST("{:<10d}", 42);
  TEST("{:>10d}", 42);
  TEST("{:010d}", -42);

  // 9. Float with precision and width
  TEST("{:.2f}", 3.14159);
  TEST("{:+15.6f}", 3.14);
  TEST("{:010.3e}", 0.000123456);

  // 10. Mixed specs
  TEST("x={:06x} f={:+.2f} d={:>8d}", 255u, -3.14, 42);

  return 0;
}
