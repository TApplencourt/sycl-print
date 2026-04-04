// examples.cpp — Compilation test suite for fmt-sycl (Approach A)
//
// This file exercises the fmt-sycl API surface. Each test function runs
// inside a SYCL kernel and calls fmt::sycl::print(), which should
// compile-time convert "{}" syntax to printf format specifiers and
// forward to sycl::ext::oneapi::experimental::printf.
//
// Compile: icpx -fsycl -std=c++20 examples.cpp -o examples
//
// NOTE: We only verify compilation here. Runtime correctness is tested
//       separately.

#include "fmt_sycl.hpp"

#include <sycl/sycl.hpp>

// Helper: submit a single_task kernel with a unique name
#define KERNEL(name, body)                                                     \
  q.submit([&](sycl::handler &cgh) {                                          \
    cgh.single_task<class name>([=]() { body; });                              \
  }).wait()

int main() {
  sycl::queue q;

  // ===================================================================
  // 1. Basic types — no format spec, just "{}"
  // ===================================================================

  // 1a. Integers
  KERNEL(test_int, {
    int val = 42;
    FMT_PRINT("{}\n", val);
  });

  KERNEL(test_negative_int, {
    int val = -123;
    FMT_PRINT("{}\n", val);
  });

  KERNEL(test_unsigned, {
    unsigned val = 100u;
    FMT_PRINT("{}\n", val);
  });

  KERNEL(test_long, {
    long val = 1234567890L;
    FMT_PRINT("{}\n", val);
  });

  KERNEL(test_long_long, {
    long long val = 9876543210LL;
    FMT_PRINT("{}\n", val);
  });

  KERNEL(test_unsigned_long_long, {
    unsigned long long val = 18446744073709551615ULL;
    FMT_PRINT("{}\n", val);
  });

  KERNEL(test_short, {
    short val = 32767;
    FMT_PRINT("{}\n", val);
  });

  KERNEL(test_int8, {
    int8_t val = -128;
    FMT_PRINT("{}\n", val);
  });

  KERNEL(test_uint8, {
    uint8_t val = 255;
    FMT_PRINT("{}\n", val);
  });

  KERNEL(test_int16, {
    int16_t val = -32768;
    FMT_PRINT("{}\n", val);
  });

  KERNEL(test_int32, {
    int32_t val = 2147483647;
    FMT_PRINT("{}\n", val);
  });

  KERNEL(test_int64, {
    int64_t val = -9223372036854775807LL;
    FMT_PRINT("{}\n", val);
  });

  KERNEL(test_uint64, {
    uint64_t val = 18446744073709551615ULL;
    FMT_PRINT("{}\n", val);
  });

  KERNEL(test_size_t, {
    size_t val = 42;
    FMT_PRINT("{}\n", val);
  });

  // 1b. Floating point
  KERNEL(test_float, {
    float val = 3.14f;
    FMT_PRINT("{}\n", val);
  });

  KERNEL(test_double, {
    double val = 2.718281828459045;
    FMT_PRINT("{}\n", val);
  });

  KERNEL(test_float_negative, {
    float val = -0.001f;
    FMT_PRINT("{}\n", val);
  });

  KERNEL(test_float_large, {
    double val = 1.0e+300;
    FMT_PRINT("{}\n", val);
  });

  KERNEL(test_float_small, {
    double val = 1.0e-300;
    FMT_PRINT("{}\n", val);
  });

  // 1c. Characters
  KERNEL(test_char, {
    char val = 'A';
    FMT_PRINT("{}\n", val);
  });

  // 1d. Booleans
  KERNEL(test_bool_true, {
    bool val = true;
    FMT_PRINT("{}\n", val);
  });

  KERNEL(test_bool_false, {
    bool val = false;
    FMT_PRINT("{}\n", val);
  });

  // 1e. Pointers
  KERNEL(test_pointer, {
    int x = 42;
    const void *ptr = &x;
    FMT_PRINT("{}\n", ptr);
  });

  // 1f. String literals
  KERNEL(test_string_literal, {
    FMT_PRINT("{}\n", "hello world");
  });

  // ===================================================================
  // 2. Integer format specs — {:d}, {:x}, {:X}, {:o}, {:b}, {:B}
  // ===================================================================

  KERNEL(test_decimal, {
    int val = 255;
    FMT_PRINT("{:d}\n", val);
  });

  KERNEL(test_hex_lower, {
    int val = 255;
    FMT_PRINT("{:x}\n", val);
  });

  KERNEL(test_hex_upper, {
    int val = 255;
    FMT_PRINT("{:X}\n", val);
  });

  KERNEL(test_octal, {
    int val = 255;
    FMT_PRINT("{:o}\n", val);
  });

  KERNEL(test_binary_lower, {
    int val = 255;
    FMT_PRINT("{:b}\n", val);
  });

  KERNEL(test_binary_upper, {
    int val = 255;
    FMT_PRINT("{:B}\n", val);
  });

  // ===================================================================
  // 3. Alternate form — {:#x}, {:#o}, {:#b}
  // ===================================================================

  KERNEL(test_alt_hex, {
    int val = 255;
    FMT_PRINT("{:#x}\n", val);
  });

  KERNEL(test_alt_oct, {
    int val = 255;
    FMT_PRINT("{:#o}\n", val);
  });

  KERNEL(test_alt_bin, {
    int val = 255;
    FMT_PRINT("{:#b}\n", val);
  });

  // ===================================================================
  // 4. Floating-point format specs — {:f}, {:e}, {:E}, {:g}, {:G}, {:a}
  // ===================================================================

  KERNEL(test_fixed, {
    double val = 3.14159;
    FMT_PRINT("{:f}\n", val);
  });

  KERNEL(test_scientific_lower, {
    double val = 3.14159;
    FMT_PRINT("{:e}\n", val);
  });

  KERNEL(test_scientific_upper, {
    double val = 3.14159;
    FMT_PRINT("{:E}\n", val);
  });

  KERNEL(test_general_lower, {
    double val = 3.14159;
    FMT_PRINT("{:g}\n", val);
  });

  KERNEL(test_general_upper, {
    double val = 3.14159;
    FMT_PRINT("{:G}\n", val);
  });

  KERNEL(test_hex_float_lower, {
    double val = 3.14159;
    FMT_PRINT("{:a}\n", val);
  });

  KERNEL(test_hex_float_upper, {
    double val = 3.14159;
    FMT_PRINT("{:A}\n", val);
  });

  // ===================================================================
  // 5. Width and alignment
  // ===================================================================

  KERNEL(test_width_right, {
    int val = 42;
    FMT_PRINT("{:10d}\n", val);
  });

  KERNEL(test_width_left, {
    int val = 42;
    FMT_PRINT("{:<10d}\n", val);
  });

  KERNEL(test_width_center, {
    int val = 42;
    FMT_PRINT("{:^10d}\n", val);
  });

  KERNEL(test_width_right_explicit, {
    int val = 42;
    FMT_PRINT("{:>10d}\n", val);
  });

  // ===================================================================
  // 6. Fill character + alignment
  // ===================================================================

  KERNEL(test_fill_left, {
    int val = 42;
    FMT_PRINT("{:*<10d}\n", val);
  });

  KERNEL(test_fill_right, {
    int val = 42;
    FMT_PRINT("{:*>10d}\n", val);
  });

  KERNEL(test_fill_center, {
    int val = 42;
    FMT_PRINT("{:*^10d}\n", val);
  });

  // ===================================================================
  // 7. Zero-padding
  // ===================================================================

  KERNEL(test_zero_pad_int, {
    int val = 42;
    FMT_PRINT("{:010d}\n", val);
  });

  KERNEL(test_zero_pad_hex, {
    int val = 255;
    FMT_PRINT("{:08x}\n", val);
  });

  KERNEL(test_zero_pad_float, {
    double val = 3.14;
    FMT_PRINT("{:012f}\n", val);
  });

  // ===================================================================
  // 8. Sign control
  // ===================================================================

  KERNEL(test_sign_plus, {
    int val = 42;
    FMT_PRINT("{:+d}\n", val);
  });

  KERNEL(test_sign_minus, {
    int val = 42;
    FMT_PRINT("{:-d}\n", val);
  });

  KERNEL(test_sign_space, {
    int val = 42;
    FMT_PRINT("{: d}\n", val);
  });

  // ===================================================================
  // 9. Precision (floats)
  // ===================================================================

  KERNEL(test_precision_2, {
    double val = 3.14159265;
    FMT_PRINT("{:.2f}\n", val);
  });

  KERNEL(test_precision_10, {
    double val = 3.14159265;
    FMT_PRINT("{:.10f}\n", val);
  });

  KERNEL(test_precision_0, {
    double val = 3.14159265;
    FMT_PRINT("{:.0f}\n", val);
  });

  KERNEL(test_precision_sci, {
    double val = 0.000123456;
    FMT_PRINT("{:.3e}\n", val);
  });

  // ===================================================================
  // 10. Combined specs
  // ===================================================================

  KERNEL(test_combined_int, {
    int val = 42;
    FMT_PRINT("{:+010d}\n", val);
  });

  KERNEL(test_combined_hex, {
    int val = 255;
    FMT_PRINT("{:#010x}\n", val);
  });

  KERNEL(test_combined_float, {
    double val = 3.14;
    FMT_PRINT("{:+15.6f}\n", val);
  });

  KERNEL(test_combined_fill_float, {
    double val = 3.14;
    FMT_PRINT("{:*>15.6f}\n", val);
  });

  // ===================================================================
  // 11. Multiple arguments
  // ===================================================================

  KERNEL(test_multi_2args, {
    int a = 1;
    int b = 2;
    FMT_PRINT("{} + {} = {}\n", a, b, a + b);
  });

  KERNEL(test_multi_mixed, {
    int i = 42;
    float f = 3.14f;
    char c = '!';
    FMT_PRINT("int={} float={} char={}\n", i, f, c);
  });

  KERNEL(test_multi_formatted, {
    int x = 255;
    double pi = 3.14159;
    FMT_PRINT("hex={:#x} pi={:.4f}\n", x, pi);
  });

  KERNEL(test_multi_many_args, {
    FMT_PRINT("{} {} {} {} {} {} {}\n", 1, 2, 3, 4, 5, 6, 7);
  });

  // ===================================================================
  // 12. No arguments — plain text
  // ===================================================================

  KERNEL(test_no_args, {
    FMT_PRINT("Hello from GPU!\n");
  });

  // ===================================================================
  // 13. Escaped braces
  // ===================================================================

  KERNEL(test_escaped_braces, {
    FMT_PRINT("{{}} is literal braces\n");
  });

  KERNEL(test_escaped_mixed, {
    int val = 42;
    FMT_PRINT("value={{{}}} done\n", val);
  });

  // ===================================================================
  // 14. Boolean as string vs integer
  // ===================================================================

  KERNEL(test_bool_default, {
    bool val = true;
    FMT_PRINT("{}\n", val);  // should print "true"
  });

  KERNEL(test_bool_as_int, {
    bool val = true;
    FMT_PRINT("{:d}\n", val);  // should print "1"
  });

  // ===================================================================
  // 15. Char as character vs integer
  // ===================================================================

  KERNEL(test_char_default, {
    char val = 'Z';
    FMT_PRINT("{}\n", val);  // should print "Z"
  });

  KERNEL(test_char_as_int, {
    char val = 'Z';
    FMT_PRINT("{:d}\n", val);  // should print "90"
  });

  // ===================================================================
  // 16. Edge cases
  // ===================================================================

  KERNEL(test_zero_int, {
    FMT_PRINT("{}\n", 0);
  });

  KERNEL(test_zero_float, {
    FMT_PRINT("{}\n", 0.0);
  });

  KERNEL(test_max_int, {
    FMT_PRINT("{}\n", 2147483647);
  });

  KERNEL(test_min_int, {
    FMT_PRINT("{}\n", -2147483647 - 1);
  });

  q.wait();
  return 0;
}
