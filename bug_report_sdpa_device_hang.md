# [tt-train/SDPA]: Device hang during NIGHTLY_SDPAForwardTest_Batch_12Heads_6Group test

## Issue Metadata

**Component / Area:** tt-train, kernels, SDPA ops

**Issue Type:** Device hang / PCIe failure

## Issue Description

### Observed

Running the full `ttml_tests` test suite causes a device hang during the `SDPAForwardTest.NIGHTLY_SDPAForwardTest_Batch_12Heads_6Group` test. The test runs for approximately 50 seconds before the device becomes unresponsive with a PCIe read error.

The error occurs after the SDPA kernel prints:
```
0:(x=0,y=0):NC: SDPA FW: num_rows_to_process=96, start_row=0, qWt=2, kWt=2, Ht=32, q_heads=12, scaler=0.125, minus_one=-1, custom_inf=1e+09
```

The crash results in:
```
terminate called after throwing an instance of 'std::runtime_error'
  what():  Read 0xffffffff from PCIE: you should reset the board.
```

After the crash, `tt-smi -r 0` fails to reset the device:
```
Failed to map bar0_uc for 0 with error Invalid argument (os error 22)
```

The device requires a kernel module reload or full system reboot to recover.

### Expected

The test should complete successfully, or fail gracefully without hanging the device. If the test is too intensive for single-device configurations, it should be skipped or have appropriate guards.

## Steps to Reproduce the Issue

### 1. Steps (exact commands)

```bash
cd ~/tt/tt-metal
time ./build_Debug/tt-train/tests/ttml_tests
```

Or to run the specific failing test:
```bash
./build_Debug/tt-train/tests/ttml_tests --gtest_filter="SDPAForwardTest.NIGHTLY_SDPAForwardTest_Batch_12Heads_6Group"
```

### 2. Input data / link or description

No external input data required. The test uses internally generated tensors with the following SDPA configuration:
- num_rows_to_process: 96
- qWt: 2, kWt: 2
- Ht: 32
- q_heads: 12
- scaler: 0.125

Note: The smaller test `NIGHTLY_SDPAForwardTest_SmallBatch_12Heads_6Group` (6 rows) passes after ~50 seconds.

### 3. Frequency

Intermittent - occurred 1 out of 2 attempts:
- **Run 1 (full suite):** Crashed with PCIe error after ~16 minutes
- **Run 2 (SDPA tests only):** Passed after ~14 minutes (826 seconds for the large batch test)

The smaller SDPA tests always pass; only the large batch (96 rows) version is flaky.

## System Details

### 1. Software Versions

- **OS version:** Ubuntu 22.04.5 LTS
- **Kernel:** 6.8.0-87-generic
- **Python:** 3.10.12
- **TT-KMD (driver):** 2.2.0
- **tt-smi:** 3.0.32
- **pyluwen:** 0.7.11
- **Firmware bundle:** 18.5.0
- **ETH FW:** 6.15.0
- **tt-metal:** Built from source (build_Debug)

### 2. Hardware Details

- **Product:** Wormhole
- **Card/System:** n150 L (single device)
- **Board ID:** 0100018611902024
- **PCIe:** Gen4 x16
- **DRAM:** Trained, 12G speed

## Regression Info (Optional)

**Is this a regression?** Unknown - first time running full test suite on this configuration.

## Logs & Diagnostics (Optional)

### Test crash log (SDPAForwardTest suite)

```
[----------] 7 tests from SDPAForwardTest
[ RUN      ] SDPAForwardTest.SDPAForwardTest_SmallBatch
2025-12-13 17:29:43.795 | info     |           Metal | DPRINT Server detached device 0 (dprint_server.cpp:823)
2025-12-13 17:29:43.803 | info     |          Fabric | TopologyMapper mapping start (mesh=0): n_log=1, n_phys=1, log_deg_hist={0:1}, phys_deg_hist={0:1} (topology_mapper.cpp:587)
2025-12-13 17:29:43.808 | info     |           Metal | DPRINT enabled on device 0, worker core (x=0,y=0) (virtual (x=18,y=18)). (dprint_server.cpp:688)
2025-12-13 17:29:43.808 | info     |           Metal | DPRINT Server attached device 0 (dprint_server.cpp:735)
2025-12-13 17:29:43.825 | info     |           Metal | Profiler started on device 0 (device_pool.cpp:203)
0:(x=0,y=0):NC: SDPA FW: num_rows_to_process=1, start_row=0, qWt=2, kWt=2, Ht=4, q_heads =2, scaler=0.125, minus_one=-1, custom_inf=1e+09
[       OK ] SDPAForwardTest.SDPAForwardTest_SmallBatch (656 ms)
[ RUN      ] SDPAForwardTest.SDPAForwardTest_SingleHead
2025-12-13 17:29:44.452 | info     |           Metal | DPRINT Server detached device 0 (dprint_server.cpp:823)
2025-12-13 17:29:44.460 | info     |          Fabric | TopologyMapper mapping start (mesh=0): n_log=1, n_phys=1, log_deg_hist={0:1}, phys_deg_hist={0:1} (topology_mapper.cpp:587)
2025-12-13 17:29:44.464 | info     |           Metal | DPRINT enabled on device 0, worker core (x=0,y=0) (virtual (x=18,y=18)). (dprint_server.cpp:688)
2025-12-13 17:29:44.464 | info     |           Metal | DPRINT Server attached device 0 (dprint_server.cpp:735)
2025-12-13 17:29:44.481 | info     |           Metal | Profiler started on device 0 (device_pool.cpp:203)
0:(x=0,y=0):NC: SDPA FW: num_rows_to_process=1, start_row=0, qWt=4, kWt=4, Ht=4, q_heads =1, scaler=0.0883883, minus_one=-1, custom_inf=1e+09
[       OK ] SDPAForwardTest.SDPAForwardTest_SingleHead (400 ms)
[ RUN      ] SDPAForwardTest.SDPAForwardTest_SmallBatch_2Heads_1Group
2025-12-13 17:29:44.852 | info     |           Metal | DPRINT Server detached device 0 (dprint_server.cpp:823)
2025-12-13 17:29:44.859 | info     |          Fabric | TopologyMapper mapping start (mesh=0): n_log=1, n_phys=1, log_deg_hist={0:1}, phys_deg_hist={0:1} (topology_mapper.cpp:587)
2025-12-13 17:29:44.863 | info     |           Metal | DPRINT enabled on device 0, worker core (x=0,y=0) (virtual (x=18,y=18)). (dprint_server.cpp:688)
2025-12-13 17:29:44.864 | info     |           Metal | DPRINT Server attached device 0 (dprint_server.cpp:735)
2025-12-13 17:29:44.881 | info     |           Metal | Profiler started on device 0 (device_pool.cpp:203)
0:(x=0,y=0):NC: SDPA FW: num_rows_to_process=1, start_row=0, qWt=2, kWt=2, Ht=4, q_heads =2, scaler=0.125, minus_one=-1, custom_inf=1e+09
[       OK ] SDPAForwardTest.SDPAForwardTest_SmallBatch_2Heads_1Group (430 ms)
[ RUN      ] SDPAForwardTest.NIGHTLY_SDPAForwardTest_SmallBatch_12Heads_6Group
2025-12-13 17:29:45.282 | info     |           Metal | DPRINT Server detached device 0 (dprint_server.cpp:823)
2025-12-13 17:29:45.291 | info     |          Fabric | TopologyMapper mapping start (mesh=0): n_log=1, n_phys=1, log_deg_hist={0:1}, phys_deg_hist={0:1} (topology_mapper.cpp:587)
2025-12-13 17:29:45.296 | info     |           Metal | DPRINT enabled on device 0, worker core (x=0,y=0) (virtual (x=18,y=18)). (dprint_server.cpp:688)
2025-12-13 17:29:45.296 | info     |           Metal | DPRINT Server attached device 0 (dprint_server.cpp:735)
2025-12-13 17:29:45.314 | info     |           Metal | Profiler started on device 0 (device_pool.cpp:203)
2025-12-13 17:29:45.394 | info     |             UMD | Starting topology discovery. (topology_discovery.cpp:69)
2025-12-13 17:29:45.401 | info     |             UMD | Established firmware bundle version: 18.5.0 (topology_discovery.cpp:368)
2025-12-13 17:29:45.401 | info     |             UMD | Established ETH FW version: 6.15.0 (topology_discovery_wormhole.cpp:324)
2025-12-13 17:29:45.401 | info     |             UMD | Completed topology discovery. (topology_discovery.cpp:73)
0:(x=0,y=0):NC: SDPA FW: num_rows_to_process=6, start_row=0, qWt=2, kWt=2, Ht=32, q_heads =12, scaler=0.125, minus_one=-1, custom_inf=1e+09
[       OK ] SDPAForwardTest.NIGHTLY_SDPAForwardTest_SmallBatch_12Heads_6Group (50616 ms)
[ RUN      ] SDPAForwardTest.NIGHTLY_SDPAForwardTest_Batch_12Heads_6Group
2025-12-13 17:30:35.898 | info     |           Metal | DPRINT Server detached device 0 (dprint_server.cpp:823)
2025-12-13 17:30:35.907 | info     |          Fabric | TopologyMapper mapping start (mesh=0): n_log=1, n_phys=1, log_deg_hist={0:1}, phys_deg_hist={0:1} (topology_mapper.cpp:587)
2025-12-13 17:30:35.912 | info     |           Metal | DPRINT enabled on device 0, worker core (x=0,y=0) (virtual (x=18,y=18)). (dprint_server.cpp:688)
2025-12-13 17:30:35.912 | info     |           Metal | DPRINT Server attached device 0 (dprint_server.cpp:735)
2025-12-13 17:30:35.931 | info     |           Metal | Profiler started on device 0 (device_pool.cpp:203)
2025-12-13 17:30:36.019 | info     |             UMD | Starting topology discovery. (topology_discovery.cpp:69)
2025-12-13 17:30:36.026 | info     |             UMD | Established firmware bundle version: 18.5.0 (topology_discovery.cpp:368)
2025-12-13 17:30:36.026 | info     |             UMD | Established ETH FW version: 6.15.0 (topology_discovery_wormhole.cpp:324)
2025-12-13 17:30:36.026 | info     |             UMD | Completed topology discovery. (topology_discovery.cpp:73)
0:(x=0,y=0):NC: SDPA FW: num_rows_to_process=96, start_row=0, qWt=2, kWt=2, Ht=32, q_heads =12, scaler=0.125, minus_one=-1, custom_inf=1e+09
terminate called after throwing an instance of 'std::runtime_error'
  what():  Read 0xffffffff from PCIE: you should reset the board.
[movsianikov-tt:1424694] *** Process received signal ***
[movsianikov-tt:1424694] Signal: Aborted (6)
[movsianikov-tt:1424694] Signal code:  (-6)
[movsianikov-tt:1424694] [ 0] /lib/x86_64-linux-gnu/libc.so.6(+0x42520)[0x7d23a9242520]
[movsianikov-tt:1424694] [ 1] /lib/x86_64-linux-gnu/libc.so.6(pthread_kill+0x12c)[0x7d23a92969fc]
[movsianikov-tt:1424694] [ 2] /lib/x86_64-linux-gnu/libc.so.6(raise+0x16)[0x7d23a9242476]
[movsianikov-tt:1424694] [ 3] /lib/x86_64-linux-gnu/libc.so.6(abort+0xd3)[0x7d23a92287f3]
[movsianikov-tt:1424694] [ 4] /lib/x86_64-linux-gnu/libstdc++.so.6(+0xa2b9e)[0x7d23a96a2b9e]
[movsianikov-tt:1424694] [ 5] /lib/x86_64-linux-gnu/libstdc++.so.6(+0xae20c)[0x7d23a96ae20c]
[movsianikov-tt:1424694] [ 6] /lib/x86_64-linux-gnu/libstdc++.so.6(+0xae277)[0x7d23a96ae277]
[movsianikov-tt:1424694] [ 7] /lib/x86_64-linux-gnu/libstdc++.so.6(+0xae4d8)[0x7d23a96ae4d8]
[movsianikov-tt:1424694] [ 8] /home/ivoitovych/tt/tt-metal/build_Debug/tt_metal/libtt_metal.so(_ZN2tt8tt_metal12DPrintServer4Impl15poll_print_dataEv+0x4b4)[0x7d23ac18e744]
[movsianikov-tt:1424694] [ 9] /home/ivoitovych/tt/tt-metal/build_Debug/tt_metal/libtt_metal.so(+0x11906c8)[0x7d23ac1906c8]
[movsianikov-tt:1424694] [10] /home/ivoitovych/tt/tt-metal/build_Debug/tt_metal/libtt_metal.so(+0x11906a5)[0x7d23ac1906a5]
[movsianikov-tt:1424694] [11] /home/ivoitovych/tt/tt-metal/build_Debug/tt_metal/libtt_metal.so(+0x1190665)[0x7d23ac190665]
[movsianikov-tt:1424694] [12] /home/ivoitovych/tt/tt-metal/build_Debug/tt_metal/libtt_metal.so(+0x119063d)[0x7d23ac19063d]
[movsianikov-tt:1424694] [13] /home/ivoitovych/tt/tt-metal/build_Debug/tt_metal/libtt_metal.so(+0x1190615)[0x7d23ac190615]
[movsianikov-tt:1424694] [14] /home/ivoitovych/tt/tt-metal/build_Debug/tt_metal/libtt_metal.so(+0x1190579)[0x7d23ac190579]
[movsianikov-tt:1424694] [15] /lib/x86_64-linux-gnu/libstdc++.so.6(+0xdc253)[0x7d23a96dc253]
[movsianikov-tt:1424694] [16] /lib/x86_64-linux-gnu/libc.so.6(+0x94ac3)[0x7d23a9294ac3]
[movsianikov-tt:1424694] [17] /lib/x86_64-linux-gnu/libc.so.6(+0x1268c0)[0x7d23a93268c0]
[movsianikov-tt:1424694] *** End of error message ***
Aborted (core dumped)

real    16m19.032s
user    22m25.604s
sys     0m41.868s
```

### Post-crash tt-smi failure (pyluwen panic)

After the crash, tt-smi cannot access the device. First two attempts show normal exit, third attempt shows the PCIe is inaccessible:

```
$ ~/tt/tt-smi/.venv/bin/tt-smi
 Detected Chips: 1
 Detecting ARC: |
 Detecting DRAM: |
 [] [16/16] ETH: |
Gathering Information ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00
Exiting TT-SMI.

$ ~/tt/tt-smi/.venv/bin/tt-smi
 Detected Chips: 1
 Detecting ARC: |
 Detecting DRAM: |
 [] [16/16] ETH: |
Gathering Information ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00
Exiting TT-SMI.

$ ~/tt/tt-smi/.venv/bin/tt-smi
WARNING: Failed to map bar0_wc for 0 with error Invalid argument (os error 22)

thread '<unnamed>' panicked at crates/ttkmd-if/src/lib.rs:294:17:
Failed to map bar0_uc for 0 with error Invalid argument (os error 22)
stack backtrace:
   0:     0x739a52113a72 - <std::sys::backtrace::BacktraceLock::print::DisplayBacktrace as core::fmt::Display>::fmt::hf435e8e9347709a8
   1:     0x739a52135923 - core::fmt::write::h0a51fad3804c5e7c
   2:     0x739a521113d3 - std::io::Write::write_fmt::h9759e4151bf4a45e
   3:     0x739a521138c2 - std::sys::backtrace::BacktraceLock::print::h1ec5ce5bb8ee285e
   4:     0x739a52114b36 - std::panicking::default_hook::{{closure}}::h5ffefe997a3c75e4
   5:     0x739a52114939 - std::panicking::default_hook::h820c77ba0601d6bb
   6:     0x739a521154c2 - std::panicking::rust_panic_with_hook::h8b29cbe181d50030
   7:     0x739a5211527a - std::panicking::begin_panic_handler::{{closure}}::h9f5b6f6dc6fde83e
   8:     0x739a52113f79 - std::sys::backtrace::__rust_end_short_backtrace::hd7b0c344383b0b61
   9:     0x739a52114f0d - __rustc[5224e6b81cd82a8f]::rust_begin_unwind
  10:     0x739a52033900 - core::panicking::panic_fmt::hc49fc28484033487
  11:     0x739a520a13c9 - ttkmd_if::PciDevice::open::hd449c4415dce98ac
  12:     0x739a5209646f - luwen_ref::ExtendedPciDevice::open::he558e07d7cd2800b
  13:     0x739a5203a1b1 - pyluwen::PciChip::new::h6e6603d5efb65c10
  14:     0x739a520627c1 - pyluwen::detect_chips_fallible::h954cf6d27bdc8e32
  15:     0x739a520644dc - pyluwen::_::__pyfunction_detect_chips_fallible::h23bf1875822302a1
  16:     0x739a52035803 - pyo3::impl_::trampoline::trampoline::h9e7ecef4889e864b
  17:     0x739a52063f60 - pyluwen::_::<impl pyluwen::detect_chips_fallible::MakeDef>::DEF::trampoline::h498b30b573a35695
  18:     0x5db1b5aae21f - <unknown>
  19:     0x5db1b5aa6739 - _PyEval_EvalFrameDefault
  20:     0x5db1b5abb0ac - _PyFunction_Vectorcall
  21:     0x5db1b5aa6739 - _PyEval_EvalFrameDefault
  22:     0x5db1b5abb0ac - _PyFunction_Vectorcall
  23:     0x5db1b5aa5460 - _PyEval_EvalFrameDefault
  24:     0x5db1b5b89be6 - <unknown>
  25:     0x5db1b5b89ab6 - PyEval_EvalCode
  26:     0x5db1b5bb0528 - <unknown>
  27:     0x5db1b5baab7f - <unknown>
  28:     0x5db1b5bb02c5 - <unknown>
  29:     0x5db1b5baf808 - _PyRun_SimpleFileObject
  30:     0x5db1b5baf4e7 - _PyRun_AnyFileObject
  31:     0x5db1b5ba3a8e - Py_RunMain
  32:     0x5db1b5b7da8d - Py_BytesMain
  33:     0x739a53229d90 - __libc_start_call_main
                               at ./csu/../sysdeps/nptl/libc_start_call_main.h:58:16
  34:     0x739a53229e40 - __libc_start_main_impl
                               at ./csu/../csu/libc-start.c:392:3
  35:     0x5db1b5b7d985 - _start
  36:                0x0 - <unknown>
Traceback (most recent call last):
  File "/home/ivoitovych/tt/tt-smi/.venv/bin/tt-smi", line 7, in <module>
    sys.exit(main())
  File "/home/ivoitovych/tt/tt-smi/.venv/lib/python3.10/site-packages/tt_smi/tt_smi.py", line 998, in main
    devices = detect_chips_with_callback(
  File "/home/ivoitovych/tt/tt-smi/.venv/lib/python3.10/site-packages/tt_tools_common/utils_common/tools_utils.py", line 340, in detect_chips_with_callback
    for device in detect_chips_fallible(
pyo3_runtime.PanicException: Failed to map bar0_uc for 0 with error Invalid argument (os error 22)
```

After reboot/kernel module reload, tt-smi works again:
```
$ ~/tt/tt-smi/.venv/bin/tt-smi
 Detected Chips: 1
 Detecting ARC: |
 Detecting DRAM: |
 [] [16/16] ETH: |
Gathering Information ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00
Exiting TT-SMI.
```

## Impact & Priority (Optional)

**Priority:** P2 (Medium) - NIGHTLY test, not blocking CI but prevents full test suite execution on n150

**Impact:**
- Cannot run complete ttml_tests suite on single n150 device without risk of device hang
- Requires system reboot or kernel module reload to recover
- May indicate underlying SDPA kernel stability issue with large batch sizes
