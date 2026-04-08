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
  TEST("name={} age={} city={}", "alice", 30, "paris");

  return 0;
}
