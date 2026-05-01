#ifndef TEST_INC
#define TEST_NAME formatter
#define TEST_INC "test_formatter.cpp"

#include "capture.hpp" // pulls in <format> and sycl_khx_print.hpp

#if !defined(FMT_SYCL_HOST) && !defined(FMT_SYCL_HOST_ACPP)
#include "sycl_std_formatters.hpp"
#endif

// Non-SYCL custom type with a formatter — exercises the customization point
// in coverage builds (which don't include <sycl/sycl.hpp>) as well as device
// builds. Specializations must be visible before the test body is included.
namespace test_fmt {
struct point { int x; int y; };
struct boxed { int v; };
}

template <>
struct std::formatter<test_fmt::point> {
  constexpr auto parse(std::format_parse_context& ctx) { return ctx.begin(); }
  auto format(const test_fmt::point& p, std::format_context& ctx) const {
    return std::format_to(ctx.out(), "({}, {})", p.x, p.y);
  }
};
template <>
struct std::formatter<test_fmt::boxed> {
  constexpr auto parse(std::format_parse_context& ctx) { return ctx.begin(); }
  auto format(const test_fmt::boxed& b, std::format_context& ctx) const {
    return std::format_to(ctx.out(), "[{}]", b.v);
  }
};

template <>
struct sycl::ext::khx::formatter<test_fmt::point> {
  static constexpr auto format(test_fmt::point p) {
    return formatted<print_detail::fixed_string{"({}, {})"}, int, int>{ {p.x, p.y} };
  }
};
template <>
struct sycl::ext::khx::formatter<test_fmt::boxed> {
  static constexpr auto format(test_fmt::boxed b) {
    return formatted<print_detail::fixed_string{"[{}]"}, int>{ {b.v} };
  }
};

#include "test_select_body.inc"
#else

// Custom non-SYCL formatter
RUN(PRINTLN("p = {}", test_fmt::point{1, 2}));
RUN(PRINTLN("b = {}", test_fmt::boxed{42}));

// Mixed primitive + custom (auto-indexed) — exercises the splicer with
// non-trivial Fmt2 expansion and the runtime args-tuple flatten.
RUN(PRINTLN("step {} of {}: p={}", 3, 100, test_fmt::point{4, 8}));
#ifndef FMT_SYCL_WA_STR
RUN(PRINTLN("{} -> {} ({})", test_fmt::boxed{1}, test_fmt::boxed{2}, "ok"));
#endif

// All-primitive path on the same entry point — should bypass expansion
RUN(PRINTLN("plain {} works too", 42));

// PRINT (no newline) variant
RUN(PRINT("[{}]\n", test_fmt::point{0, 0}));

#if !defined(FMT_SYCL_HOST) && !defined(FMT_SYCL_HOST_ACPP)
// SYCL types — only available in real device builds
RUN(PRINTLN("range = {}", sycl::range<1>{4}));
RUN(PRINTLN("range = {}", sycl::range<2>{4, 8}));
RUN(PRINTLN("range = {}", sycl::range<3>{4, 8, 16}));

RUN(PRINTLN("id = {}", sycl::id<1>{2}));
RUN(PRINTLN("id = {}", sycl::id<2>{2, 5}));
RUN(PRINTLN("id = {}", sycl::id<3>{1, 2, 3}));

RUN(PRINTLN("step {} of {}: range={}", 3, 100, sycl::range<3>{4, 8, 16}));

RUN(PRINT("[{}]\n", sycl::id<2>{0, 0}));
#endif

#endif
