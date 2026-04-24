#ifndef TEST_INC
#define TEST_NAME floats
#define TEST_INC "test_floats.cpp"
#include "test_body.inc"
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


#endif
