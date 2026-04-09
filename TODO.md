# fmt-sycl Refactoring Plan

## Refactor code flow

### Remove `FMT_SYCL_BUFFER_PATH_ONLY`

This flag is removed entirely. Atomicity is always guaranteed:
- DPC++: single combined `DEVICE_PRINTF` call.
- ACPP: buffer accumulation + single `sycl::detail::print` call.

### DPC++ path (printf-format path)

```
sycl::print<"format">(args...)
  -> printf_format = std_format_to_printf_format("format")   // compile-time
     - static_assert if something is not convertible to printf format
     - No buffer/%c workaround needed (atomicity is mandatory)
  -> DEVICE_PRINTF(printf_format, cast(args)...)              // single call
```

### ACPP path (buffer path)

```
sycl::print<"format">(args...)
  -> char buffer[256]
  -> buffer = format<"format">(args...)   // internal helper, formats into buffer
  -> sycl::detail::print("%s", buffer)
```

- `format()` is an internal helper (not public API).
- Dragonbox stays in the header, guarded by `#if FMT_SYCL_ACPP`.

### What gets deleted

- `FMT_SYCL_BUFFER_PATH_ONLY` and all associated `#ifdef` branching
- `print_arg_default()` (DPC++ per-arg printf calls)
- `print_arg_with_spec()` (DPC++ per-arg with spec)
- `print_impl()` (DPC++ recursive per-arg walker)
- All `needs_buf` / `print_with_fill` / `format_int_buf` DPC++ fallback logic

### What stays

- Dragonbox (ACPP only, under `#if FMT_SYCL_ACPP`)
- Format string parsing (`find_placeholder`, `parse_spec`, `format_spec`, etc.)
- `build_combined_printf_fmt` + `walk_and_emit` (DPC++ core path)
- ACPP buffer formatting (`acpp_print_decimal`, `acpp_write_*`, `acpp_fmt_*`, etc.)
- ACPP accumulator path (`acpp_print_impl`, `acpp_write_one_arg`, etc.)
- Public API (`print`, `println`, `KHR_PRINT`, `KHR_PRINTLN`)

### Testing strategy

- ACPP: full feature set tested (dragonbox, binary, hex-float, center-align, etc.)
- DPC++: printf-compatible subset only
  - Features not supported on DPC++ are commented out
  - Default float `{}` uses `%g` on DPC++; tests that produce different output
    between `%g` and dragonbox are commented out for DPC++
  - `FMT_SYCL_WA_STR` guards remain for string tests (DPC++ O0 bug)
