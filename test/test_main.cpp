#include "capture.hpp"

bool test_integers(::sycl::queue& q);
bool test_floats(::sycl::queue& q);
bool test_strings(::sycl::queue& q);
bool test_layout(::sycl::queue& q);
bool test_misc(::sycl::queue& q);

int main() {
  ::sycl::queue q;

  bool ok = true;
  ok &= test_integers(q);
  ok &= test_floats(q);
  ok &= test_strings(q);
  ok &= test_layout(q);
  ok &= test_misc(q);

  printf(ok ? "PASS\n" : "FAIL\n");
  return ok ? 0 : 1;
}
