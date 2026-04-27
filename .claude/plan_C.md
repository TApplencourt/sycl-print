# Plan C — Tag-based dispatch (decision: measure first, then implement)

This file is the entry point if you (Claude) are picked up in a future
session to work on the C optimization. **Read this whole file before
starting.** Then run the experiment in part 2 to decide whether to
proceed at all.

---

## Context (what's already done)

Three register-pressure changes already landed (commit `43485df`):
- **A** — dropped `char dgt[68]` and two 48 B `small_buf` temporaries from
  `write_int_rt` / `write_float_rt` / `hex_float_to_buf_rt`. Replaced by
  `pad_in_place` which shifts content right in `out`.
- **B** — `KHR_SYCL_PRINT_BUFFER_SIZE` default 255 → 128.
- **D** — Hoisted dragonbox cache tables to namespace-scope
  `inline constexpr` so they reliably land in `__constant`.

Coverage and ASAN/UBSAN are clean: 84.56% regions, 82.17% branches, no
sanitizer findings (the only UBSAN noise was a fuzz int-overflow,
fixed in `7f62e19`).

Plan C is the **biggest potential GPU win still on the table**, but it
is also the **highest-risk and most uncertain** change. Don't do it
without numbers.

---

## Part 1 — The C change itself

### Problem

`buffer_path::dispatch_by_index` (sycl_khr_print.hpp ~line 1901):

```cpp
template <typename Tuple, typename F, size_t... Is>
inline void dispatch_by_index(const Tuple &t, int idx, F &&fn, std::index_sequence<Is...>) {
  ((static_cast<int>(Is) == idx ? (fn(std::get<Is>(t)), void()) : void()), ...);
}
```

For `print("{} {} {} {} {} {} {}", 1,2,3,4,5,6,7)`:
- 7 placeholder loop iterations × 7 dispatch arms = **49 inlined formatter
  call sites** in one kernel.
- Every tuple slot stays live until the dispatch resolves.
- Per-type formatter (`write_arg_default<int>`) is instantiated once per
  arg position even though the body is identical.

### Goal

Collapse the fanout from **N×M** (positions × types) to **M** (types).

### Design

1. **Type tag enum** (one byte):
   ```cpp
   enum class arg_tag : uint8_t {
     i32, u32, i64, u64, f32, f64, ch, bl, cstr, voidp,
   };
   ```

2. **Variant slot** (16 B, tag + 8-byte payload):
   ```cpp
   struct arg_value {
     arg_tag tag;
     union {
       int32_t  i32;  uint32_t u32;
       int64_t  i64;  uint64_t u64;
       float    f32;  double   f64;
       char     ch;   bool     bl;
       const char *cstr;
       const void *voidp;
     } v;
     int as_int() const noexcept; // returns 0 for non-integral tags
   };
   ```

3. **`make_arg_value<T>(T)`** — `constexpr if`-chain mapping the C++
   type to the right tag. Distinguishes `const char*` (cstr) from
   generic pointers (voidp) the same way `effective_type` does today.

4. **Construct at the entry** (`buffer_path::print`):
   ```cpp
   inline void print(print_string<...> ps, Args... args) {
     fmt_buf out;
     arg_value vals[] = { make_arg_value(args)... };
     format_rt(out, ps, vals, sizeof...(Args));
     flush_buf(out, ps.needs_pct_escape);
   }
   ```

5. **Formatters become non-templated** (one body each):
   ```cpp
   inline void write_arg_default(fmt_buf &out, const arg_value &a) {
     switch (a.tag) {
       case arg_tag::i32: write_decimal(out, a.v.i32); break;
       // ... one arm per tag
     }
   }
   ```
   Same for `write_arg_rt`.

6. **`format_rt` loses the tuple**:
   ```cpp
   inline void format_rt(fmt_buf &out, const print_string_view &ps,
                         const arg_value *vals) {
     // walks ps.phs[], dispatches via vals[e.arg_idx].tag
   }
   ```
   **Optional but high-value**: drop the template parameter entirely so
   `format_rt` is a single function in the binary regardless of call
   site. Requires extracting a non-templated view of the placeholder
   table from `print_string<Args...>`.

### Risks / open questions

1. **Variant size on stack**: 16 B × 16 args (MAX_PH) = 256 B, vs ~128 B
   for an `int` tuple of 16. Could be net worse on the stack. If this
   matters: parallel-arrays variant (`tags[N]`, `values[N*8]`) drops to
   144 B at the cost of indexing complexity.

2. **At -O2 with full inlining**, the compiler may already prune the
   dispatch ladder for known arg types — making C a wash or slight loss
   (extra runtime switch vs pruned static branches). **This is exactly
   why we measure first.**

3. **DPC++ path (`specifiers_path`) is untouched.** C only affects ACPP.

4. **`print_string` template params unchanged** — `Args...` still needed
   for consteval validation in the constructor.

5. **`needs_pct_escape` computation uses `Args...`** — still works
   because the consteval constructor still has the pack.

### Order of work (when implementing)

1. Add `arg_tag`, `arg_value`, `make_arg_value` (~50 lines).
2. Rewrite `write_arg_default` / `write_arg_rt` as switch-on-tag (~80
   lines changed).
3. Add `as_int()` (~10 lines). Drop `resolve_int_arg`, `dispatch_arg`,
   `dispatch_by_index`.
4. Rewrite `format_rt` to take `arg_value[]` (~30 lines).
5. Update `print` and `println` public entries (~10 lines).
6. **Re-run `make -j coverage`** — must stay ≥ 82.17% branches.
7. **Re-run ASAN+UBSAN** (see ".claude/run_sanitizers.sh" if it exists,
   else replicate the manual build from session notes).
8. **Re-run the experiment from part 2** — must show measurable PTX
   size and/or register-count improvement, otherwise revert.

---

## Part 2 — The experiment (run this FIRST)

The point: get an actual number from ACPP for a representative kernel
**before** committing to the rewrite. If the number says C buys
nothing (or hurts), don't do C — instead look at:
- **E**: replace recursive `collect_printf_args` with one
  index-sequence fold (DPC++ path).
- **uint128 trim**: dragonbox `compute_mul_parity` /
  `umul192_lower128` materialize a full 128-bit value to read one
  parity bit. Could return only what's needed.

### What to measure

For each kernel: **PTX size in bytes** and **register count** reported
by `ptxas -v` (or AMDGPU equivalent). Both are proxies for register
pressure. PTX size also captures code-bloat from the fold expansion.

### Kernels to compile

Create one file `experiment/c_baseline.cpp` with kernels parameterized
by N (number of args):

```cpp
#include "../sycl_khr_print.hpp"
#include <sycl/sycl.hpp>

template <int N> void launch(sycl::queue &q);

template <> void launch<1>(sycl::queue &q) {
  q.parallel_for(1, [=](sycl::id<1> i) {
    KHR_PRINT("a={}\n", int(i[0]));
  }).wait();
}
template <> void launch<3>(sycl::queue &q) {
  q.parallel_for(1, [=](sycl::id<1> i) {
    int a = i[0], b = a+1, c = a+2;
    KHR_PRINT("{} {} {}\n", a, b, c);
  }).wait();
}
template <> void launch<7>(sycl::queue &q) {
  q.parallel_for(1, [=](sycl::id<1> i) {
    int a=i[0], b=a+1, c=a+2, d=a+3, e=a+4, f=a+5, g=a+6;
    KHR_PRINT("{} {} {} {} {} {} {}\n", a, b, c, d, e, f, g);
  }).wait();
}
template <> void launch<7 mixed>(sycl::queue &q) {
  q.parallel_for(1, [=](sycl::id<1> i) {
    int a = i[0]; double b = a*0.5; const char *c = "x";
    bool d = a&1; char e = 'q'; unsigned f = a; long long g = a;
    KHR_PRINT("{} {} {} {} {} {} {}\n", a, b, c, d, e, f, g);
  }).wait();
}

int main() {
  sycl::queue q;
  launch<1>(q); launch<3>(q); launch<7>(q); launch<7mixed>(q);
}
```

(Pseudo-syntax — fix the `<7 mixed>` to a valid template name like
`launch_mixed7`.)

### How to extract the numbers

ACPP with PTX backend (CUDA target):

```bash
acpp --acpp-targets=cuda:sm_80 -O2 \
  -Xcompiler -ptxas-options=-v \
  experiment/c_baseline.cpp -o build/c_baseline 2> build/c_baseline.ptxas.txt
grep -E "Used [0-9]+ registers|^ptxas info" build/c_baseline.ptxas.txt
```

For AMD targets, replace with the equivalent flag (probably
`--save-temps` and inspecting `.s` for `.amdhsa_kernel.*sgpr_count`).

If you can't get per-kernel breakdowns, the **PTX/AMDGPU asm file size**
is a coarser but still useful proxy — bigger means more inlined code.

### Decision rule

Compile **once** with current `main` (post-A+B+D), record numbers.
Then implement C, compile again, compare:

| Outcome | Action |
|---|---|
| C reduces register count by ≥4 OR PTX size by ≥10% on the 7-arg kernel | Land C. |
| C is roughly flat (±5% PTX, ±2 registers) | Revert C. Pursue E + dragonbox uint128 trim instead. |
| C makes things *worse* | Definitely revert. The static dispatch ladder was already being pruned by the compiler; the runtime switch is overhead. |

### Files to commit alongside the experiment

- `experiment/c_baseline.cpp` — the kernels above.
- `experiment/Makefile` or shell script — build with ACPP, capture
  ptxas output.
- `experiment/RESULTS.md` — table with the numbers from each run.
  Even a "C didn't help" result is valuable: it tells future-you not
  to revisit this.

---

## Part 3 — If C lands, follow-up wins

Even after C, these remain on the table from the original audit:

- **E** (DPC++ path): replace recursive `collect_printf_args` with one
  index-sequence fold. No growing tuple, helps -O0/-O1.
- **dragonbox `uint128` simplification**: `compute_mul_parity` /
  `umul192_lower128` build a full 128-bit value to read 1 parity bit
  and do 1 zero-test. Could return `{bool parity; bool is_integer;}`
  directly.
- **`__uint128_t` for dragonbox `umul128`**: on NVPTX/AMDGPU/SPIR-V,
  `__uint128_t` lowers to `mul.wide.u64`/`mad.wide.u64` natively.
  Worth checking if the device path supports it; if so, replace the
  4-multiply schoolbook.

---

## How to resume in a fresh session

1. `cd /home/applenco/project/p26.06/fmt-sycl`
2. `git log --oneline -3` — confirm you're at or after `43485df`.
3. Read this file end-to-end. Then read sycl_khr_print.hpp lines
   1900–1965 (the dispatch path) so you have the current code in
   context.
4. **Do part 2 first.** No code changes until you have ACPP numbers.
5. Fill in `experiment/RESULTS.md` with the baseline numbers, then
   either implement C and re-measure, or write up why you skipped it.
