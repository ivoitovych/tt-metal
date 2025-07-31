# TT-Metal Arbitrary Data Structure Communication Pipeline

A complete working example demonstrating how to process arbitrary (non-tensor) data structures on Tenstorrent hardware using TT-Metal, enabling custom C++ data processing in device kernels.

## 🎯 Project Purpose

This example demonstrates end-to-end communication of arbitrary data structures between host and Tenstorrent device, enabling:

- **Custom data processing** in device kernels using unpredicted C++ code
- **Arbitrary structure transformation** (not limited to tensors)
- **Direct memory manipulation** of complex data layouts
- **Foundation for HDL simulation acceleration** and custom compute workloads

## 📋 Current Implementation

### Data Flow Architecture
```
Host (C++) → DRAM → L1 Memory → Device Kernel (C++) → L1 Memory → DRAM → Host (C++)
```

### Pipeline Components
1. **Host Application** (`minimal_struct_dataflow.cpp`)
   - Creates interleaved DRAM buffers for input/output
   - Generates test data with custom structure patterns
   - Launches single data movement kernel
   - Performs comprehensive verification with tile-by-tile analysis

2. **Device Kernel** (`kernels/dataflow/all_in_one_struct.cpp`)
   - Reads arbitrary data structures from DRAM to L1
   - Processes data using custom C++ transformation logic
   - Writes transformed structures back to DRAM
   - Uses proper TT-Metal addressing for interleaved buffers

### Data Structures
```cpp
// Input: 16-byte aligned structure
struct InputStruct {
    float float_value;      // Primary data
    int32_t int_value;      // Counter/ID
    uint16_t short_value;   // Metadata
    int8_t char_value;      // Flags
    uint8_t padding[5];     // Alignment padding
};

// Output: Different 16-byte structure
struct OutputStruct {
    float result_value;     // Transformed result
    int32_t status_code;    // Status information
    uint16_t iteration_count; // Derived metadata
    uint8_t flags;          // Processed flags
    uint8_t padding[5];     // Alignment padding
};
```

## ✨ Key Features Achieved

### ✅ **Arbitrary Data Processing**
- Processes custom 16-byte structures (not tensors)
- Handles mixed data types (float, int32, uint16, int8)
- Maintains data integrity through full pipeline

### ✅ **Custom Transformation Logic**
- **Float transformation**: `result = input * 2.0f`
- **Integer passthrough**: `status = input_int`
- **Modulo operation**: `iterations = input_short % 100`
- **Bitwise operation**: `flags = input_char & 0xFF`

### ✅ **Performance Characteristics**
- **Data throughput**: 32 KB input → 32 KB output
- **Tile processing**: 2 tiles × 1024 structures each
- **Memory efficiency**: Zero-copy L1 buffer reuse
- **Verification**: 100% data integrity maintained

### ✅ **Robust Architecture**
- **Single kernel approach**: Avoids NOC conflicts
- **Proper addressing**: Uses `InterleavedAddrGenFast` for correct DRAM access
- **Memory safety**: Separate L1 buffers for input/output
- **Comprehensive debugging**: Extensive DPRINT logging for development

## 🚀 Technical Implementation

### Memory Layout
```
DRAM Buffers (Interleaved):
├── Input Buffer  (32KB): 2048 × InputStruct (16 bytes each)
└── Output Buffer (32KB): 2048 × OutputStruct (16 bytes each)

L1 Memory (Per Core):
├── Input Buffer  (0x10000): 16KB temporary storage
└── Output Buffer (0x20000): 16KB temporary storage
```

### Addressing Solution
The key breakthrough was using TT-Metal's proper interleaved buffer addressing:

```cpp
// CORRECT: Use InterleavedAddrGenFast for proper tile addressing
const InterleavedAddrGenFast<true> src_gen = {
    .bank_base_address = src_addr,
    .page_size = tile_bytes,
    .data_format = DataFormat::UInt8
};

// Read/write using tile-based addressing
noc_async_read_tile(tile_idx, src_gen, l1_input_buffer_addr);
noc_async_write_tile(tile_idx, dst_gen, l1_output_buffer_addr);
```

### Verification Results
```
=== FINAL VERIFICATION RESULTS ===
✓ Total elements processed: 2048
✓ Perfect matches: 2048 (100.0%)
✓ Data corruption errors: 0 (0.0%)
✓ Edge case verification: PASS
✓ Overall verification: PASS ✓
```

## 🛠️ Build & Run

### Prerequisites
- TT-Metal development environment
- Tenstorrent hardware (N300 or similar)
- GCC/Clang with C++20 support

### Build Instructions
```bash
cd ~/setup/tt-metal
./build_metal.sh --build-programming-examples --debug

# Run the example
./build_Debug/programming_examples/hdl_simulation/minimal_struct_dataflow

# Enable detailed kernel debugging
export TT_METAL_DPRINT_CORES=0,0
./build_Debug/programming_examples/hdl_simulation/minimal_struct_dataflow
```

### Expected Output
```
🎉 SUCCESS: TT-Metal arbitrary data structure pipeline verified!
✓ Data integrity maintained through full pipeline
✓ Reader kernel: DRAM → L1 CB transfer working with custom structures
✓ Compute kernel: Structure transformation working
✓ Writer kernel: L1 CB → DRAM transfer working with different structures
✓ Pipeline ready for arbitrary data processing workloads
```

## 📁 Project Structure
```
tt_metal/programming_examples/hdl_simulation/minimal_struct_dataflow/
├── minimal_struct_dataflow.cpp           # Host application
├── kernels/dataflow/all_in_one_struct.cpp # Device kernel
└── README.md                              # This file
```

## 🔧 Key Technical Learnings

### 1. **Avoid Tensor APIs for Arbitrary Data**
- TT-Metal's compute kernels are designed for tensor operations
- For arbitrary data structures, use **data movement kernels** with direct memory access
- APIs like `get_read_ptr()` and `get_write_ptr()` work in data movement, not compute kernels

### 2. **Proper Interleaved Buffer Addressing**
- Manual address calculation (`base + offset`) fails with interleaved buffers
- Use `InterleavedAddrGenFast` with `noc_async_read_tile()`/`noc_async_write_tile()`
- TT-Metal handles complex DRAM bank interleaving internally

### 3. **NOC Resource Management**
- Multiple data movement kernels on same core can cause NOC conflicts
- Single kernel approach avoids "Illegal NOC usage" errors
- Each kernel should use dedicated NOC (NOC_0 vs NOC_1)

### 4. **Memory Layout Considerations**
- 16-byte alignment is crucial for struct layout
- Use separate L1 buffers for input/output to avoid memory corruption
- `volatile` pointers ensure proper memory access in device code

## 🚀 Next Steps & Future Enhancements

### **Multi-Core Scaling**
- Extend to multiple cores for larger datasets
- Implement work distribution across core grid
- Add inter-core communication for complex algorithms

### **Advanced Transformations**
- Implement sorting, filtering, aggregation operations
- Add conditional processing based on struct content
- Support variable-length data structures

### **HDL Simulation Integration**
- Add Verilator/HDL simulation support
- Create testbenches for hardware verification
- Enable co-simulation with RTL models

### **Performance Optimizations**
- Implement double-buffering for continuous processing
- Add DMA-style bulk transfers
- Optimize memory access patterns

### **API Extensions**
- Create reusable template for arbitrary struct processing
- Add compile-time struct validation
- Support nested/complex data structures

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| **Data Throughput** | 32 KB → 32 KB |
| **Processing Elements** | 2,048 structures |
| **Tile Size** | 16 KB (1,024 structures) |
| **Memory Efficiency** | Zero-copy L1 processing |
| **Data Integrity** | 100% verified |
| **Transformation Types** | 4 (float, int, mod, bitwise) |

## 🐛 Debugging Tips

### Enable Kernel Debug Output
```bash
export TT_METAL_DPRINT_CORES=0,0
```

### Common Issues & Solutions
1. **"Illegal NOC usage"** → Use single kernel or separate NOCs
2. **Data corruption** → Check struct alignment and L1 buffer separation
3. **Address calculation errors** → Use `InterleavedAddrGenFast` instead of manual calculation
4. **Compilation errors** → Ensure struct definitions match between host and kernel

## 📜 License

This project follows the TT-Metal licensing terms. Check the main TT-Metal repository for details.

## 🤝 Contributing

This example serves as a foundation for arbitrary data processing on Tenstorrent hardware. Contributions for additional transformations, multi-core scaling, and HDL integration are welcome.

---

**Status**: ✅ **WORKING** - Full pipeline verification passed with 100% data integrity
