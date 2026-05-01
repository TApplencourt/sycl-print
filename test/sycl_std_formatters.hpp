#pragma once
// Host-side std::formatter specializations for SYCL composite types.
// Mirror exactly the device-side output produced by sycl::ext::khx::formatter
// so test reference output (std::format) can be diffed against device output.
//
// Test-only — kept here, not in the main header, to avoid pulling SYCL into
// the std namespace for users who only want device printing.

#include <format>
#include <sycl/sycl.hpp>

template <int N>
struct std::formatter<::sycl::range<N>> : std::formatter<std::string> {
  auto format(const ::sycl::range<N>& r, std::format_context& ctx) const {
    std::string s;
    for (int i = 0; i < N; i++) {
      if (i) s += 'x';
      s += std::to_string(r[i]);
    }
    return std::formatter<std::string>::format(s, ctx);
  }
};

template <int N>
struct std::formatter<::sycl::id<N>> : std::formatter<std::string> {
  auto format(const ::sycl::id<N>& id, std::format_context& ctx) const {
    std::string s = "(";
    for (int i = 0; i < N; i++) {
      if (i) s += ", ";
      s += std::to_string(id[i]);
    }
    s += ")";
    return std::formatter<std::string>::format(s, ctx);
  }
};
