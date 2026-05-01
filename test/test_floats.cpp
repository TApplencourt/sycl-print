#ifndef TEST_INC
#define TEST_NAME floats
#define TEST_INC "test_floats.cpp"
#include "test_select_body.inc"
#else

// Floats — default format ({}) for float and double
RUN(PRINT("{}\n", 3.14f));
RUN(PRINT("{}\n", 0.1f));
RUN(PRINT("{}\n", 1.0f));
RUN(PRINT("{}\n", -2.5f));
RUN(PRINT("{}\n", 3.14));
RUN(PRINT("{}\n", 0.1));
RUN(PRINT("{}\n", 1.0));
RUN(PRINT("{}\n", -2.5));

// Floats — explicit specs
RUN(PRINT("{:g}\n", 3.14));
RUN(PRINT("{:g}\n", 2.718281828459045));
RUN(PRINT("{:g}\n", -0.001));
RUN(PRINT("{:g}\n", 1.0e+300));
RUN(PRINT("{:g}\n", 1.0e-300));

// Float type specifiers
RUN(PRINT("{:f}\n", 3.14159));
RUN(PRINT("{:e}\n", 3.14159));
RUN(PRINT("{:E}\n", 3.14159));
RUN(PRINT("{:g}\n", 3.14159));
RUN(PRINT("{:G}\n", 3.14159));

// Precision
RUN(PRINT("{:.2f}\n", 3.14159265));
RUN(PRINT("{:.10f}\n", 3.14159265));
RUN(PRINT("{:.0f}\n", 3.14159265));
RUN(PRINT("{:.3e}\n", 0.000123456));

// Float specials (inf, nan, -0)
RUN(PRINT("{:g}\n", 1.0 / 0.0));
RUN(PRINT("{:g}\n", -1.0 / 0.0));
RUN(PRINT("{:f}\n", 1.0 / 0.0));
RUN(PRINT("{:e}\n", 1.0 / 0.0));
RUN(PRINT("{:+f}\n", 3.14));

// {:F} — uppercase INF/NAN
RUN(PRINT("{:F}\n", 7.14));
RUN(PRINT("{:F}\n", 1.0 / 0.0));
RUN(PRINT("{:F}\n", -1.0 / 0.0));

// {:#} — alternate form on floats
RUN(PRINT("{:#g}\n", 1.0));
RUN(PRINT("{:#g}\n", 100.0));
RUN(PRINT("{:#.0f}\n", 3.0));
RUN(PRINT("{:#e}\n", 1.0));

// Precision on g
RUN(PRINT("{:.1g}\n", 3.14159));
RUN(PRINT("{:.2g}\n", 3.14159));
RUN(PRINT("{:.4g}\n", 3.14159));
RUN(PRINT("{:.1g}\n", 0.00123));
RUN(PRINT("{:.6g}\n", 1234567.0));

// Alt form forces decimal point with precision 0
RUN(PRINT("{:#.0f}\n", 3.0));
RUN(PRINT("{:#.0f}\n", 100.0));
RUN(PRINT("{:#.0f}\n", 0.0));
RUN(PRINT("{:#.0e}\n", 1.0));

// Float edge values
RUN(PRINT("{:e}\n", 2.2250738585072014e-308));
RUN(PRINT("{:e}\n", 1.7976931348623157e+308));
RUN(PRINT("{:g}\n", 1.7976931348623157e+308));
RUN(PRINT("{:f}\n", 0.0));
RUN(PRINT("{:f}\n", -0.0));

// Float with explicit spec on float (not double)
RUN(PRINT("{:f}\n", 3.14f));
RUN(PRINT("{:e}\n", 3.14f));
RUN(PRINT("{:g}\n", 3.14f));
RUN(PRINT("{:.2f}\n", 3.14f));
RUN(PRINT("{:+.4e}\n", 0.001f));
RUN(PRINT("{:012.3f}\n", 42.0f));

// Float extremes with explicit specs (float 32-bit)
RUN(PRINT("{:g}\n", std::numeric_limits<float>::max()));
RUN(PRINT("{:e}\n", std::numeric_limits<float>::max()));
RUN(PRINT("{:g}\n", -std::numeric_limits<float>::max()));
RUN(PRINT("{:g}\n", std::numeric_limits<float>::min()));   // smallest normal
RUN(PRINT("{:e}\n", std::numeric_limits<float>::min()));
RUN(PRINT("{:g}\n", std::numeric_limits<float>::denorm_min()));

// Negative zero
RUN(PRINT("{:f}\n", -0.0f));
RUN(PRINT("{:e}\n", -0.0f));
RUN(PRINT("{:g}\n", -0.0f));
RUN(PRINT("{:g}\n", -0.0));
RUN(PRINT("{:e}\n", -0.0));

// NaN (quiet)
RUN(PRINT("{:f}\n", std::numeric_limits<float>::quiet_NaN()));
RUN(PRINT("{:g}\n", std::numeric_limits<float>::quiet_NaN()));
RUN(PRINT("{:F}\n", std::numeric_limits<float>::quiet_NaN()));
RUN(PRINT("{:f}\n", std::numeric_limits<double>::quiet_NaN()));

// Negative inf (float)
RUN(PRINT("{:f}\n", -1.0f / 0.0f));
RUN(PRINT("{:g}\n", -1.0f / 0.0f));
RUN(PRINT("{:F}\n", -1.0f / 0.0f));

// Default format for float/double specials (covers ACPP dragonbox inf/nan path)
RUN(PRINT("{}\n", 1.0 / 0.0));
RUN(PRINT("{}\n", -1.0 / 0.0));
RUN(PRINT("{}\n", 0.0 / 0.0));
RUN(PRINT("{}\n", 1.0f / 0.0f));
RUN(PRINT("{}\n", -1.0f / 0.0f));
RUN(PRINT("{}\n", 0.0f / 0.0f));

// {:g} with precision that triggers prec=1 fallback
RUN(PRINT("{:.0g}\n", 3.14));
RUN(PRINT("{:.0g}\n", 0.5));

// More float values with explicit spec (exercises format_float paths)
RUN(PRINT("{:g}\n", 0.0));
RUN(PRINT("{:g}\n", -0.0));
RUN(PRINT("{:e}\n", 0.0f));
RUN(PRINT("{:f}\n", 0.0f));
RUN(PRINT("{:g}\n", 0.0f));
RUN(PRINT("{:g}\n", 1.0e-7));
RUN(PRINT("{:g}\n", 9.99e+4));

// {:g} rounding at fixed/scientific boundary
RUN(PRINT("{:g}\n", 9.999995e+5));
RUN(PRINT("{:g}\n", 9.9999950000001e+5));
RUN(PRINT("{:g}\n", 9.9e+5));

// fmt_fixed rounding: midpoint ties with IEEE double-rounding
RUN(PRINT("{:.2g}\n", 9.95));
RUN(PRINT("{:.1g}\n", 0.95));
RUN(PRINT("{:.4g}\n", 9999.5));
RUN(PRINT("{:.0f}\n", 0.5));
RUN(PRINT("{:.0f}\n", 1.5));
RUN(PRINT("{:.0f}\n", 2.5));
RUN(PRINT("{:.1f}\n", 9.95));

// Default format for zero (exercises dragonbox format_shortest zero path)
RUN(PRINT("{}\n", 0.0));
RUN(PRINT("{}\n", -0.0));
RUN(PRINT("{}\n", 0.0f));
RUN(PRINT("{}\n", -0.0f));

// Default format for small values (format_shortest leading zeros path)
RUN(PRINT("{}\n", 0.001));
RUN(PRINT("{}\n", 0.001f));

// ACPP-only: dragonbox paths that round numbers don't hit. These default
// `{}` outputs use shortest-round-trip representation which std::format
// matches but printf("%g") (DPC++ path) does not (capped at 6 sig digits).
//   - round-up fallback in shorter_interval_case<double> (large powers of 2)
//   - small_divisor path → check_divisibility_and_divide_by_pow10
//     + compute_mul_parity<double> + umul192_lower128
//     + remove_trailing_zeros<uint64_t> inner loop
#if FMT_SYCL_ACPP
RUN(PRINT("{}\n", 1.2676506002282294e+30));  // 2^100 (shorter_interval round-up)
RUN(PRINT("{}\n", 7.888609052210118e-31));   // 2^-100 (shorter_interval round-up)
RUN(PRINT("{}\n", 0.30000000000000004));     // small_divisor, full precision
RUN(PRINT("{}\n", 1234567.89));              // small_divisor
RUN(PRINT("{}\n", 1.234));                   // small_divisor
RUN(PRINT("{}\n", 7.89));                    // small_divisor
#endif


#endif
