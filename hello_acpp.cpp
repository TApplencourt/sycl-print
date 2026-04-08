#include <sycl/sycl.hpp>
#include "sycl_khr_print.hpp"
#include <iostream>

int main() {
    sycl::queue q;
    std::cout << "Running on: " << q.get_device().get_info<sycl::info::device::name>() << "\n";

    // Atomicity test: 8 work-items, each prints "{} {} {}" with double + string
    q.parallel_for(sycl::range<1>{8}, [=](sycl::id<1> idx) {
        int id = static_cast<int>(idx[0]);
        sycl::khr::print<"{} {} {}\n">(id, 3.14 * id, "hello");
    }).wait();
}
