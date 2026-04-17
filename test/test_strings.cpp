#ifndef TEST_INC
#define TEST_NAME strings
#define TEST_INC "test_strings.cpp"
#include "test_body.inc"
#else

// Char, bool
RUN(PRINT("{}\n", 'A'));
RUN(PRINT("{}\n", false));
RUN(PRINT("{}\n", true));
RUN(PRINT("{:d}\n", true));
RUN(PRINT("{}\n", 'Z'));
RUN(PRINT("{:d}\n", 'Z'));
RUN(PRINT("{:>10}\n", true));
RUN(PRINT("{:<10}\n", false));
RUN(PRINT("{:s}\n", true));
RUN(PRINT("{:s}\n", false));
RUN(PRINT("{:c}\n", 65));

// println
RUN(PRINTLN("hello println"));
RUN(PRINTLN("{} + {} = {}", 1, 2, 3));
RUN(PRINTLN("{:08x}", 255u));

// Dynamic char
RUN({
  volatile char c = 'Q';
  PRINT("{}\n", static_cast<char>(c));
});

// Width on char
RUN(PRINT("{:<10c}\n", 'B'));
RUN(PRINT("{:>10c}\n", 'C'));

// String tests (guarded for DPC++ O0 string literal bug)
#ifndef FMT_SYCL_WA_STR
RUN(PRINT("{}\n", "hello world"));
{
  const char *env = "cpu-char-* copied";
#ifdef FMT_STD_PATH
  RUN(PRINT("{}\n", env));
#else
  size_t len = std::strlen(env) + 1;
  char *shared = ::sycl::malloc_shared<char>(len, q);
  std::memcpy(shared, env, len);
  RUN(PRINT("{}\n", shared));
  ::sycl::free(shared, q);
#endif
}
RUN(PRINT("{:<20s}\n", "hello"));
RUN(PRINT("{:>20s}\n", "hello"));
RUN(PRINT("{:.5s}\n", "hello world"));
RUN(PRINT("{:.0s}\n", "hello"));
RUN(PRINT("{:.100s}\n", "hello"));
RUN(PRINT("{:>10.5s}\n", "hello world"));
RUN(PRINT("{:<10.3s}\n", "hello"));
#endif

#ifdef FMT_SYCL_BUFFER_PATH_ONLY
RUN(PRINT("{:*^10}\n", true));
RUN(PRINTLN("{:08x}", 255));
RUN(PRINT("{:10c}\n", 'A'));
RUN(PRINT("{:^10c}\n", 'D'));
RUN(PRINT("{:*>5c}\n", 'X'));
RUN(PRINT("{:*<5c}\n", 'X'));
RUN(PRINT("{:*^5c}\n", 'X'));
#endif

#if !defined(FMT_SYCL_WA_STR) && defined(FMT_SYCL_BUFFER_PATH_ONLY)
RUN(PRINT("{:20s}\n", "hello"));
RUN(PRINT("{:*^10.5s}\n", "hello world"));
RUN(PRINT("{:*<20s}\n", "hello"));
RUN(PRINT("{:*>20s}\n", "hello"));
RUN(PRINT("{:*^20s}\n", "hello"));
#endif

#endif
