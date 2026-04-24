#ifndef TEST_INC
#define TEST_NAME layout
#define TEST_INC "test_layout.cpp"
#include "test_body.inc"
#else

// Width and alignment
RUN(PRINT("{:10d}\n", 42));
RUN(PRINT("{:<10d}\n", 42));
RUN(PRINT("{:>10d}\n", 42));

// Zero-padding
RUN(PRINT("{:010d}\n", 42));
RUN(PRINT("{:08x}\n", 255u));
RUN(PRINT("{:012f}\n", 3.14));

// Sign control
RUN(PRINT("{:+d}\n", 42));
RUN(PRINT("{:-d}\n", 42));
RUN(PRINT("{: d}\n", 42));

// Width smaller than content
RUN(PRINT("{:3d}\n", 123456));
RUN(PRINT("{:2f}\n", 3.14));
RUN(PRINT("{:1d}\n", -42));

// Zero-pad interactions
RUN(PRINT("{:010d}\n", -42));
RUN(PRINT("{:010f}\n", -3.14));
RUN(PRINT("{:<010d}\n", 42));
RUN(PRINT("{:<010d}\n", -42));
RUN(PRINT("{:010e}\n", -1.23));


#endif
