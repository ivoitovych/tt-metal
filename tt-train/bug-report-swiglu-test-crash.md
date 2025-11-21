# Bug Report: SwiGLU test crash due to xtensor lazy evaluation

## Describe the bug

The `SwiGLUOpTest.SwiGLU_Basic_1x1x32x32` test crashes with segmentation faults or `std::bad_alloc` / `std::bad_array_new_length` errors when running the tt-train test suite.

The root cause is that the `silu` lambda in the `swiglu_forward_reference` function returns an unevaluated xtensor lazy expression that holds references to local variables. When the lambda returns, these references become dangling, causing crashes when the expression is later evaluated.

### Error Output

When run in isolation, the test fails with:
```
[ RUN      ] SwiGLUOpTest.SwiGLU_Basic_1x1x32x32
unknown file: Failure
C++ exception with description "std::bad_array_new_length" thrown in the test body.
[  FAILED  ] SwiGLUOpTest.SwiGLU_Basic_1x1x32x32 (110 ms)
```

Or crashes with a segmentation fault:
```
[ RUN      ] SwiGLUOpTest.SwiGLU_Basic_1x1x32x32
[fb0ad2b2ef86:00016] *** Process received signal ***
[fb0ad2b2ef86:00016] Signal: Segmentation fault (11)
[fb0ad2b2ef86:00016] Signal code: Address not mapped (1)
[fb0ad2b2ef86:00016] Failing at address: 0x1
[fb0ad2b2ef86:00016] [ 0] /lib/x86_64-linux-gnu/libc.so.6(+0x42520)[0x799cc9bfb520]
[fb0ad2b2ef86:00016] [ 1] ./build/tests/ttml_tests(+0x260a5f)[0x63eeffb6ba5f]
...
[fb0ad2b2ef86:00016] [ 8] ./build/tests/ttml_tests(_ZN2xt21throw_broadcast_errorINS_7svectorImLm4ESaImELb1EEES3_EEvRKT_RKT0_+0x25)[0x63eeffb6b365]
...
Segmentation fault (core dumped)
```

The stack trace shows `xt::throw_broadcast_error` which indicates xtensor is trying to operate on arrays with corrupted shape data due to dangling references.

## Steps to reproduce the issue

1. Build tt-train with Debug configuration:
```bash
cd ${TT_METAL_HOME}/tt-train
cmake -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_C_COMPILER_LAUNCHER=ccache \
      -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
      -B build -GNinja
cmake --build build --config Debug --clean-first
```

2. Run the SwiGLU test:
```bash
./build/tests/ttml_tests --gtest_filter=SwiGLUOpTest.SwiGLU_Basic_1x1x32x32
```

3. Observe the crash (segfault or std::bad_alloc exception)

Note: The crash may manifest differently on different runs - sometimes as `std::bad_array_new_length`, sometimes as `std::bad_alloc`, and sometimes as a segmentation fault.

## Expected behavior

The test should pass successfully, comparing the SwiGLU kernel implementation against the reference implementation without any crashes or memory errors.

Expected output:
```
[ RUN      ] SwiGLUOpTest.SwiGLU_Basic_1x1x32x32
[       OK ] SwiGLUOpTest.SwiGLU_Basic_1x1x32x32 (115 ms)
```

## Please complete the following environment information

- **OS version**: Ubuntu 22.04 (running in Docker container)
- **Python version**: N/A (C++ test)
- **Framework version**: tt-train (part of tt-metal)
- **Version of software**: tt-metal main branch, commit `ed2c8a1d20` or later
- **Hardware**: Tenstorrent Wormhole device
- **KMD version**: 2.2.0
- **Firmware version**: 18.5.0

## Additional context

### Problematic code location

File: `tt-train/tests/ops/swiglu_op_test.cpp`

The issue is in the `swiglu_forward_reference` function around line 62:

```cpp
// SiLU activation: z * sigmoid(z)
auto silu = [](const xt::xarray<float>& t) {
    auto sigmoid = 1.0f / (1.0f + xt::exp(-t));
    return t * sigmoid;  // Returns lazy expression with dangling references!
};
```

### Fix

The fix is to explicitly specify the return type to force evaluation:

```cpp
// SiLU activation: z * sigmoid(z)
// Note: Must return xt::xarray to force evaluation, not a lazy expression
auto silu = [](const xt::xarray<float>& t) -> xt::xarray<float> {
    auto sigmoid = 1.0f / (1.0f + xt::exp(-t));
    return t * sigmoid;
};
```

By adding `-> xt::xarray<float>`, xtensor is forced to evaluate the expression and return an actual array instead of a lazy expression that holds references to the local `sigmoid` variable.
