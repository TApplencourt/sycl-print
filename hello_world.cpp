#include <cstdlib>
#include <format>
#include <sycl/sycl.hpp>
#include "sycl_khr_print.hpp"

int main(int argc, char *argv[]) {
  sycl::queue Q;
  std::cout << "Running on "
            << Q.get_device().get_info<sycl::info::device::name>() << std::endl;
  int global_range = std::atoi(argv[1]);

  float f = 3.14f;

  std::cout << std::format("{}\n",  f) << std::endl;

  Q.parallel_for(global_range, [=](sycl::id<1> idx) {
    int id =  static_cast<int>(idx);
#ifdef REF_IMPL
   sycl::ext::oneapi::experimental::printf("Hello %s: id: %d: value %f\n", "World", id, f);
#else
   // Dragon box, will print 3.14, and not 3.14000.
   sycl::khr::println<"Hello {}: id {}: value {}">("World", id, f);
#endif
  });
  Q.wait();
}
