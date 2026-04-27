# fmt-sycl

- `std::print`-like formatting for SYCL device kernels.
- Required C++20

- Tested with DPC++ and AdaptiveCpp (ACPP, generic/SSCP backend)

## Quick example

> Source: [`example_readme1.cpp`](example_readme1.cpp)

```cpp
#include "sycl_khr_print.hpp"
#include <sycl/sycl.hpp>

int main() {
  sycl::queue q;
  q.parallel_for(4, [=](sycl::id<1> i) {
    KHR_PRINTLN("work-item {} says {}",
        static_cast<int>(i), "hello");
  }).wait();
}
```

One possible ordering of the output:
```bash
work-item 0 says hello
work-item 2 says hello
work-item 1 says hello
work-item 3 says hello
```

## Advanced example

> Source: [`example_readme2.cpp`](example_readme2.cpp)

```cpp
#include "sycl_khr_print.hpp"
#include <sycl/sycl.hpp>

int main() {
  sycl::queue q;
  q.parallel_for(4, [=](sycl::id<1> i) {
    int id = static_cast<int>(i);
    float v = 3.14159f * (id + 1);
    KHR_PRINTLN("format used: 'id: {{0}}, v2dp={{1:6.2f}}, v={{1:8.5f}}' -> id: {0}, v2dp={1:6.2f}, v={1:8.5f}", id, v);
  }).wait();
}
```

One possible ordering of the output:
```bash
format used: 'id: {0}, v2dp={1:6.2f}, v={1:8.5f}' -> id: 0, v2dp=  3.14, v= 3.14159
format used: 'id: {0}, v2dp={1:6.2f}, v={1:8.5f}' -> id: 2, v2dp=  9.42, v= 9.42477
format used: 'id: {0}, v2dp={1:6.2f}, v={1:8.5f}' -> id: 1, v2dp=  6.28, v= 6.28318
format used: 'id: {0}, v2dp={1:6.2f}, v={1:8.5f}' -> id: 3, v2dp= 12.57, v=12.56636
```

## API

```cpp
sycl::khr::print<"format string">(args...);    // no trailing newline
sycl::khr::println<"format string">(args...);  // appends \n

// Convenience macros (avoid angle-bracket syntax)
KHR_PRINT("format string", args...);
KHR_PRINTLN("format string", args...);
```

### Backend differences

**AdaptiveCpp (ACPP)** supports the full `std::format` spec. The entire output is accumulated into a buffer before printing, so all features work atomically.

**DPC++** uses a single `printf` call with format specifiers. This is atomic but limits which format features are available. Unsupported features produce a compile-time error:

```
error: static assertion failed:
  This format string uses features not supported on DPC++
  ({:b}, {:a}, {:^}, custom fill, {:#x} with signed int,
  dynamic width/precision, dragonbox default float).
  These features are only available on ACPP.
```

Features only available on ACPP:
- Binary format (`{:b}`, `{:B}`)
- Hex float (`{:a}`, `{:A}`)
- Center alignment (`{:^}`)
- Custom fill characters (`{:*>10}`)
- Signed integers with hex/oct (`{:x}` with `int`)
- Alternate hex (`{:#x}` with signed int)
- Dynamic width/precision (`{:{}}`, `{:.{}}`)
- Dragonbox shortest-decimal float (default `{}` with floats)

### ACPP buffer limit

The output buffer defaults to 128 characters. Output longer than that per `KHR_PRINT` call is silently truncated. Override with:

```cpp
#define KHR_SYCL_PRINT_BUFFER_SIZE 512
#include "sycl_khr_print.hpp"
```

## Build

```bash
# DPC++ (Intel)
icpx -fsycl -std=c++20 my_kernel.cpp -o my_kernel

# AdaptiveCpp (generic/SSCP backend)
acpp --acpp-targets=generic -std=c++20 my_kernel.cpp -o my_kernel
```

## Tests

```bash
make test              # Run all tests (format + fuzz + ffast-math)
```
Or individually:
```bash
make test-format       # Format correctness (diff against std::format)
make test-fuzz         # Fuzz with random values
make test-ffast        # Fuzz with -ffast-math
```

For ACPP builds (`make ... USE_ACPP=1`):
- `test_buffer_path` tests ACPP-only features (binary, hex float, center align, custom fill, etc.) and is only compiled with `USE_ACPP=1`.
- `test_escape_percent` and `fuzz_escape_percent` test `%` in formatted output. These are expected to fail on CPU because `%` → `%%` escaping is applied to work around CUDA's `vprintf` interpreting `%` as format specifiers. They should pass on GPU.
