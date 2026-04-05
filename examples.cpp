// examples.cpp — Dual-mode test suite for fmt-sycl
//
// Build & diff:
//   icpx -std=c++20 -DUSE_STD examples.cpp -o examples_std
//   icpx -fsycl -std=c++20 -Wall -Werror examples.cpp -o examples_sycl
//   diff <(./examples_std) <(ONEAPI_DEVICE_SELECTOR=*:cpu ./examples_sycl)

#include <cstdint>

#ifdef USE_STD
  #include <format>
  #include <iostream>
  #define PRINT(fmt_str, ...) \
    std::cout << std::format(fmt_str __VA_OPT__(,) __VA_ARGS__)
  #define RUN(...) do { __VA_ARGS__; } while (0)
#else
  #include "fmt_sycl.hpp"
  #include <sycl/sycl.hpp>
  #define PRINT(fmt_str, ...) \
    ::fmt::sycl::print<fmt_str>(__VA_ARGS__)
  #define RUN(...)                                         \
    q.submit([&](::sycl::handler &cgh) {                   \
      cgh.single_task([=]() { __VA_ARGS__; });             \
    }).wait()
#endif

int main() {
#ifndef USE_STD
  ::sycl::queue q;
#endif

  // =================================================================
  // 1. Integers
  // =================================================================
  RUN(PRINT("{}\n", 42));
  RUN(PRINT("{}\n", -123));
  RUN(PRINT("{}\n", 100u));
  RUN(PRINT("{}\n", 1234567890L));
  RUN(PRINT("{}\n", 9876543210LL));
  RUN(PRINT("{}\n", 18446744073709551615ULL));
  RUN(PRINT("{}\n", static_cast<short>(32767)));
  RUN(PRINT("{}\n", static_cast<int8_t>(-128)));
  RUN(PRINT("{}\n", static_cast<uint8_t>(255)));
  RUN(PRINT("{}\n", static_cast<int16_t>(-32768)));
  RUN(PRINT("{}\n", 2147483647));
  RUN(PRINT("{}\n", static_cast<int64_t>(-9223372036854775807LL)));
  RUN(PRINT("{}\n", static_cast<uint64_t>(18446744073709551615ULL)));

  // =================================================================
  // 2. Floats — use explicit specs so output matches between std/printf
  // =================================================================
  RUN(PRINT("{:g}\n", 3.14));
  RUN(PRINT("{:g}\n", 2.718281828459045));
  RUN(PRINT("{:g}\n", -0.001));
  RUN(PRINT("{:g}\n", 1.0e+300));
  RUN(PRINT("{:g}\n", 1.0e-300));

  // =================================================================
  // 3. Char, bool, string
  // =================================================================
  RUN(PRINT("{}\n", 'A'));
  RUN(PRINT("{}\n", true));
  RUN(PRINT("{}\n", false));
  RUN(PRINT("{}\n", "hello world"));

  // =================================================================
  // 4. Integer type specifiers
  // =================================================================
  RUN(PRINT("{:d}\n", 255));
  RUN(PRINT("{:x}\n", 255));
  RUN(PRINT("{:X}\n", 255));
  RUN(PRINT("{:o}\n", 255));
  // {:b} and {:B} — Phase 4 (will diff)
  RUN(PRINT("{:b}\n", 255));
  RUN(PRINT("{:B}\n", 255));

  // =================================================================
  // 5. Alternate form
  // =================================================================
  RUN(PRINT("{:#x}\n", 255));
  RUN(PRINT("{:#o}\n", 255));
  // {:#b} — Phase 4 (will diff)
  RUN(PRINT("{:#b}\n", 255));

  // =================================================================
  // 6. Float type specifiers
  // =================================================================
  RUN(PRINT("{:f}\n", 3.14159));
  RUN(PRINT("{:e}\n", 3.14159));
  RUN(PRINT("{:E}\n", 3.14159));
  RUN(PRINT("{:g}\n", 3.14159));
  RUN(PRINT("{:G}\n", 3.14159));
  RUN(PRINT("{:a}\n", 3.14159));
  RUN(PRINT("{:A}\n", 3.14159));

  // =================================================================
  // 7. Width and alignment
  // =================================================================
  RUN(PRINT("{:10d}\n", 42));
  RUN(PRINT("{:<10d}\n", 42));
  RUN(PRINT("{:>10d}\n", 42));
  // {:^10d} — center, Phase 4 (will diff)
  RUN(PRINT("{:^10d}\n", 42));

  // =================================================================
  // 8. Fill character + alignment — Phase 4 (will diff)
  // =================================================================
  RUN(PRINT("{:*<10d}\n", 42));
  RUN(PRINT("{:*>10d}\n", 42));
  RUN(PRINT("{:*^10d}\n", 42));

  // =================================================================
  // 9. Zero-padding
  // =================================================================
  RUN(PRINT("{:010d}\n", 42));
  RUN(PRINT("{:08x}\n", 255));
  RUN(PRINT("{:012f}\n", 3.14));

  // =================================================================
  // 10. Sign control
  // =================================================================
  RUN(PRINT("{:+d}\n", 42));
  RUN(PRINT("{:-d}\n", 42));
  RUN(PRINT("{: d}\n", 42));

  // =================================================================
  // 11. Precision
  // =================================================================
  RUN(PRINT("{:.2f}\n", 3.14159265));
  RUN(PRINT("{:.10f}\n", 3.14159265));
  RUN(PRINT("{:.0f}\n", 3.14159265));
  RUN(PRINT("{:.3e}\n", 0.000123456));

  // =================================================================
  // 12. Combined specs
  // =================================================================
  RUN(PRINT("{:+010d}\n", 42));
  RUN(PRINT("{:#010x}\n", 255));
  RUN(PRINT("{:+15.6f}\n", 3.14));
  // Fill char — Phase 4 (will diff)
  RUN(PRINT("{:*>15.6f}\n", 3.14));

  // =================================================================
  // 13. Multiple arguments
  // =================================================================
  RUN({
    int a = 1; int b = 2;
    PRINT("{} + {} = {}\n", a, b, a + b);
  });
  RUN(PRINT("hex={:#x} pi={:.4f}\n", 255, 3.14159));
  RUN(PRINT("{} {} {} {} {} {} {}\n", 1, 2, 3, 4, 5, 6, 7));

  // =================================================================
  // 14. Plain text and escaped braces
  // =================================================================
  RUN(PRINT("Hello from GPU!\n"));
  RUN(PRINT("{{}} is literal braces\n"));
  RUN(PRINT("value={{{}}} done\n", 42));

  // =================================================================
  // 15. Bool / char as alternate types
  // =================================================================
  RUN(PRINT("{}\n", true));
  RUN(PRINT("{:d}\n", true));
  RUN(PRINT("{}\n", 'Z'));
  RUN(PRINT("{:d}\n", 'Z'));

  // =================================================================
  // 16. Edge cases — basic
  // =================================================================
  RUN(PRINT("{}\n", 0));
  RUN(PRINT("{:g}\n", 0.0));
  RUN(PRINT("{}\n", 2147483647));
  RUN(PRINT("{}\n", -2147483647 - 1));

  // =================================================================
  // 17. Integer extremes
  // =================================================================
  RUN(PRINT("{:d}\n", -2147483647 - 1));
  RUN(PRINT("{:x}\n", -2147483647 - 1));
  RUN(PRINT("{:b}\n", -2147483647 - 1));
  RUN(PRINT("{:b}\n", static_cast<uint64_t>(18446744073709551615ULL)));
  RUN(PRINT("{:#b}\n", 0));
  RUN(PRINT("{:x}\n", static_cast<int>(-1)));
  RUN(PRINT("{:#x}\n", 0));

  // =================================================================
  // 18. Float specials (inf, nan, -0)
  // =================================================================
  RUN(PRINT("{:g}\n", 1.0 / 0.0));             // inf
  RUN(PRINT("{:g}\n", -1.0 / 0.0));            // -inf
  RUN(PRINT("{:f}\n", 1.0 / 0.0));             // inf
  RUN(PRINT("{:e}\n", 1.0 / 0.0));             // inf
  RUN(PRINT("{:+f}\n", 3.14));                  // +3.140000
  RUN(PRINT("{:a}\n", 0.0));                    // 0p+0
  RUN(PRINT("{:a}\n", -0.0));                   // -0p+0
  RUN(PRINT("{:a}\n", 1.0));                    // 1p+0
  RUN(PRINT("{:#a}\n", 3.14159));               // 0x1.921f...

  // =================================================================
  // 19. Bool / char edge cases
  // =================================================================
  RUN(PRINT("{:>10}\n", true));                  // "      true"
  RUN(PRINT("{:<10}\n", false));                 // "false     "
  RUN(PRINT("{:*^10}\n", true));                 // "***true***"
  RUN(PRINT("{:s}\n", true));                    // "true"
  RUN(PRINT("{:s}\n", false));                   // "false"
  RUN(PRINT("{:c}\n", 65));                      // A

  // =================================================================
  // 20. Width/fill combos
  // =================================================================
  RUN(PRINT("{:#^20b}\n", 255));                 // binary+center+fill+alt
  RUN(PRINT("{:+010x}\n", -1));                  // sign-aware zero-pad hex
  RUN(PRINT("{:>20b}\n", static_cast<uint8_t>(255))); // right-align binary
  RUN(PRINT("{:*>5d}\n", -42));                  // fill + negative

  // =================================================================
  // 21. Hex float edge cases
  // =================================================================
  RUN(PRINT("{:a}\n", 1.0 / 0.0));              // inf
  RUN(PRINT("{:A}\n", 0.0 / 0.0));              // NAN
  RUN(PRINT("{:a}\n", 0.125));                    // exact power of 2

  // =================================================================
  // 22. Positional arguments
  // =================================================================
  RUN(PRINT("{0} {1} {0}\n", 42, 99));
  RUN(PRINT("{1} {0}\n", "hello", "world"));
  RUN(PRINT("{0:x} {0:d} {0:b}\n", 255));
  RUN(PRINT("{0:>10} {1:<10}\n", 42, 99));
  RUN(PRINT("{2} {0} {1}\n", 'a', 'b', 'c'));
  RUN(PRINT("{0} {0} {0}\n", 7));
  RUN(PRINT("{0:.2f} {1:+d}\n", 3.14, -42));

  // =================================================================
  // 23. Dynamic width and precision
  // =================================================================
  RUN(PRINT("{:{}d}\n", 42, 10));                // width=10
  RUN(PRINT("{:{}d}\n", 42, 1));                 // width=1 (smaller than content)
  RUN(PRINT("{:.{}f}\n", 3.14159, 2));           // precision=2
  RUN(PRINT("{:.{}f}\n", 3.14159, 0));           // precision=0
  RUN(PRINT("{:{}.{}f}\n", 3.14, 15, 6));        // width=15, precision=6
  RUN(PRINT("{0:{1}x}\n", 255, 10));             // positional + dynamic width
  RUN(PRINT("{0:{1}.{2}f}\n", 3.14, 20, 4));     // positional + both dynamic

  return 0;
}
