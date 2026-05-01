#include "capture.hpp"

#define CONCAT2(a, b) a##b
#define CONCAT(a, b) CONCAT2(a, b)

bool CONCAT(test_, TEST_NAME)() {
  int _test_id = 0;

  auto std_out = capture_stdout([&]() {
#define FMT_STD_PATH
#if FMT_SYCL_ACPP
    // ACPP path uses a fixed fmt_buf and silently truncates per print call.
    // Mirror that on the std reference so tests can include long output.
#define PRINT(fmt_str, ...)                                                   \
  std::cout << std::format(fmt_str __VA_OPT__(,) __VA_ARGS__)                 \
                   .substr(0, KHX_SYCL_PRINT_BUFFER_SIZE)
#define PRINTLN(fmt_str, ...)                                                 \
  std::cout << std::format(fmt_str __VA_OPT__(,) __VA_ARGS__)                 \
                   .substr(0, KHX_SYCL_PRINT_BUFFER_SIZE)                     \
            << '\n'
#else
#define PRINT(fmt_str, ...) std::cout << std::format(fmt_str __VA_OPT__(,) __VA_ARGS__)
#define PRINTLN(fmt_str, ...) std::cout << std::format(fmt_str __VA_OPT__(,) __VA_ARGS__) << '\n'
#endif
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
#define PRINT(fmt_str, ...) KHX_PRINT(fmt_str __VA_OPT__(,) __VA_ARGS__)
#define PRINTLN(fmt_str, ...) KHX_PRINTLN(fmt_str __VA_OPT__(,) __VA_ARGS__)
#define RUN(...) do { printf("Test %d\n", ++_test_id); \
  for (int _i = 0; _i < N; _i++) { __VA_ARGS__; } } while (0)
#include TEST_INC
#undef RUN
#undef PRINTLN
#undef PRINT
  });

  return diff_output(TEST_INC, std_out, sycl_out);
}

#ifndef TEST_NO_MAIN
int main() {
  return CONCAT(test_, TEST_NAME)() ? 0 : 1;
}
#endif
