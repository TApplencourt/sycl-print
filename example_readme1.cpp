#include "sycl_khx_print.hpp"
#include <sycl/sycl.hpp>

int main() {
  sycl::queue q;
  q.parallel_for(4, [=](sycl::id<1> i) {
    KHX_PRINTLN("work-item {} says {}",
        static_cast<int>(i), "hello");
  }).wait();
}
