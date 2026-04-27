#pragma once

#include <cstdint>
#include <cstdio>
#include <limits>
#include <cstdlib>
#include <cstring>
#include <format>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/mman.h>
#include <unistd.h>

#include "../sycl_khr_print.hpp"
#if !defined(FMT_SYCL_HOST) && !defined(FMT_SYCL_HOST_ACPP)
#include <sycl/sycl.hpp>
#endif

constexpr int N = 2;

static std::string capture_stdout(auto&& fn) {
  std::cout.flush();
  fflush(stdout);

  int mem_fd = memfd_create("capture", 0);
  int saved_fd = dup(STDOUT_FILENO);
  dup2(mem_fd, STDOUT_FILENO);

  fn();

  std::cout.flush();
  fflush(stdout);
  dup2(saved_fd, STDOUT_FILENO);
  close(saved_fd);

  auto size = lseek(mem_fd, 0, SEEK_END);
  lseek(mem_fd, 0, SEEK_SET);
  std::string result(size, '\0');
  ::read(mem_fd, result.data(), size);
  close(mem_fd);
  return result;
}

// unused in test_main.cpp which includes this header but only calls test_*()
[[maybe_unused]] static bool diff_output(const char* name,
                        const std::string& expected,
                        const std::string& actual) {
  if (expected == actual)
    return true;
  std::istringstream a(expected), b(actual);
  std::string la, lb;
  int test_id = 0;
  bool ga, gb, mismatch = false;
  std::string exp_block, act_block;
  auto flush = [&]() {
    if (mismatch) {
      fprintf(stderr, "[  FAILED  ] %s / test %d\n", name, test_id);
      fprintf(stderr, "  Expected:\n%s", exp_block.c_str());
      fprintf(stderr, "  Actual:\n%s", act_block.c_str());
    }
    exp_block.clear();
    act_block.clear();
    mismatch = false;
  };
  while (ga = bool(std::getline(a, la)),
         gb = bool(std::getline(b, lb)),
         ga || gb) {
    if (la.starts_with("Test ") && la == lb) {
      flush();
      test_id = std::atoi(la.c_str() + 5);
    } else {
      if (la != lb) mismatch = true;
      exp_block += "    " + la + "\n";
      act_block += "    " + lb + "\n";
    }
    la.clear();
    lb.clear();
  }
  flush();
  return false;
}
