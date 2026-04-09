#include <sycl/sycl.hpp>
#include "sycl_khr_print.hpp"
#include <iostream>

int main() {
    sycl::queue q;
    std::cout << "Running on: " << q.get_device().get_info<sycl::info::device::name>() << "\n";

    // README quick example
    q.parallel_for(4, [=](sycl::id<1> i) {
        KHR_PRINTLN("work-item {} says {}", static_cast<int>(i), "hello");
    }).wait();

    std::cout << "---\n";

    // README advanced example
    q.parallel_for(4, [=](sycl::id<1> i) {
        int id = static_cast<int>(i);
        float v = 3.14159f * (id + 1);
        KHR_PRINTLN("format used: 'id: {{0}}, v2dp={{1:6.2f}}, v={{1:8.5f}}' -> id: {0}, v2dp={1:6.2f}, v={1:8.5f}", id, v);
    }).wait();
}
