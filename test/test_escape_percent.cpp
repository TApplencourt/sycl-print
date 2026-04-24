#ifndef TEST_INC
#define TEST_NAME escape_percent
#define TEST_INC "test_escape_percent.cpp"
#include "test_body.inc"
#else

RUN(PRINT("100% done\n"));
RUN(PRINT("100%% done\n"));
RUN(PRINT("{}% complete\n", 50));
RUN(PRINT("{} is 100% of {}\n", 42, 42));
RUN(PRINT("%\n"));
RUN(PRINT("%%\n"));
RUN(PRINT("{}%%{}\n", 1, 2));

#endif
