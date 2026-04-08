- To run do: ~/project/p26.06/argo-shim
- See Makefile to verify that everything is correct ( std::format, same output as sycl impl)

0/ Fix example (it should compile with `icpx -fsycl -std=c++20`)
1/ Dragonbox is broken for: `{}`, 3.14
Print  `3.140000104904175`, where it should print 3.14
- See `hello-world` running with `./hello-world 1` (comapre the first print to the `hello` one)
- We should add a new example like this in `test_example.cpp` (and for double too)
- And then fix dragon box 

2/ Because we don't print everything in one go (`sycl::khr::println` print per "word-format")
   in case of multiple work-item, the output is unreadable because totaly interleaved.
   - See `hello-world` running with `./hello-world 2` and compared whem compiled `REF_IMPL` versus no `REF_IMPL`
   - We hould print everything in one go.
   - The problem is for when `sycl::khr::println` need to print `per char` (in order to workarround `sycl::ext::oneapi::experimental::printf` doesn' support %s for stack string, but `%c` is supported)
      - In the case, the workarround is append one `%c` for each char we want to print in the format
	  (test_printf_workarround_dynamic_string_with_know_max_length.cpp)
      - This require a unper bound on the number `char`.  But In case of float / double dragon box have a upper limit, so maybe we are good.
