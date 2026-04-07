#include <sycl/sycl.hpp>

int main() {
  sycl::queue q;

  // String literal through a local pointer — segfaults at -O0
  q.submit([&](sycl::handler &cgh) {
    cgh.single_task([=]() {
      const char *s = "hello";
      ::sycl::ext::oneapi::experimental::printf("%s\n", s);
    });
  }).wait();

  return 0;
}
