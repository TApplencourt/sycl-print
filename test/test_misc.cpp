#ifndef TEST_INC
#define TEST_NAME misc
#define TEST_INC "test_misc.cpp"
#include "test_body.inc"
#else

// Combined specs
RUN(PRINT("{:+010d}\n", 42));
RUN(PRINT("{:+15.6f}\n", 3.14));

// Multiple arguments
RUN({
  int a = 1; int b = 2;
  PRINT("{} + {} = {}\n", a, b, a + b);
});
RUN(PRINT("{} {} {} {} {} {} {}\n", 1, 2, 3, 4, 5, 6, 7));

// Plain text and escaped braces
RUN(PRINT("Hello from GPU!\n"));
RUN(PRINT("{{}} is literal braces\n"));
RUN(PRINT("value={{{}}} done\n", 42));


// Edge cases
RUN(PRINT("{}\n", 0));
RUN(PRINT("{:g}\n", 0.0));
RUN(PRINT("{}\n", 2147483647));
RUN(PRINT("{}\n", -2147483647 - 1));

// Positional arguments
RUN(PRINT("{0} {1} {0}\n", 42, 99));
#ifndef FMT_SYCL_WA_STR
RUN(PRINT("{1} {0}\n", "hello", "world"));
#endif
RUN(PRINT("{0:>10} {1:<10}\n", 42, 99));
RUN(PRINT("{2} {0} {1}\n", 'a', 'b', 'c'));
RUN(PRINT("{0} {0} {0}\n", 7));
RUN(PRINT("{0:.2f} {1:+d}\n", 3.14, -42));

// Zero with every type specifier
RUN(PRINT("{:d}\n", 0u));
RUN(PRINT("{:x}\n", 0u));
RUN(PRINT("{:X}\n", 0u));
RUN(PRINT("{:o}\n", 0u));
RUN(PRINT("{:f}\n", 0.0));
RUN(PRINT("{:e}\n", 0.0));
RUN(PRINT("{:g}\n", 0.0));



#endif
