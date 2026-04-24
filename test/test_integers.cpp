#ifndef TEST_INC
#define TEST_NAME integers
#define TEST_INC "test_integers.cpp"
#include "test_body.inc"
#else

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

// Type specifiers
RUN(PRINT("{:d}\n", 255));
RUN(PRINT("{:x}\n", 255u));
RUN(PRINT("{:X}\n", 255u));
RUN(PRINT("{:o}\n", 255u));
RUN(PRINT("{:#o}\n", 255u));

// Extremes
RUN(PRINT("{:d}\n", -2147483647 - 1));
RUN(PRINT("{:d}\n", static_cast<int64_t>(-9223372036854775807LL - 1)));
RUN(PRINT("{}\n", static_cast<int64_t>(-9223372036854775807LL - 1)));


#endif
