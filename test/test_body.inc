#include "capture.hpp"

#define CONCAT2(a, b) a##b
#define CONCAT(a, b) CONCAT2(a, b)

bool CONCAT(test_, TEST_NAME)(::sycl::queue& q) {
  int _test_id = 0;

  auto std_out = capture_stdout([&]() {
#define FMT_STD_PATH
#define PRINT(fmt_str, ...) std::cout << std::format(fmt_str __VA_OPT__(,) __VA_ARGS__)
#define PRINTLN(fmt_str, ...) std::cout << std::format(fmt_str __VA_OPT__(,) __VA_ARGS__) << '\n'
#define RUN(...) do { printf("Test %d\n", ++_test_id); \
  for (int _i = 0; _i < N; _i++) { __VA_ARGS__; } } while (0)
#include TEST_INC
#undef RUN
#undef PRINTLN
#undef PRINT
#undef FMT_STD_PATH
  });

  _test_id = 0;

  auto sycl_out = capture_stdout([&]() {
#define PRINT(fmt_str, ...) ::sycl::khr::print<fmt_str>(__VA_ARGS__)
#define PRINTLN(fmt_str, ...) ::sycl::khr::println<fmt_str>(__VA_ARGS__)
#define RUN(...) do { printf("Test %d\n", ++_test_id); \
  q.parallel_for(N, [=](::sycl::id<1>) { __VA_ARGS__; }).wait(); } while (0)
#include TEST_INC
#undef RUN
#undef PRINTLN
#undef PRINT
  });

  return diff_output(TEST_INC, std_out, sycl_out);
}
