# AI Collaboration Log

## Prompts and Assistance

**1. Optimization Strategies & Ideas**
- I provided the base kernel code to the AI and asked for optimization strategies based on cache-locality, branchless programming, and SIMD principles.
- The AI suggested block-tiling for matrix multiplication, AoS-to-SoA data layout conversions for the filter gradient, and CSR formats for graph traversal.

**2. Debugging Precision Issues**
- During the implementation of the `matmul` and `sparse_spmm` optimizations, I encountered floating-point precision discrepancies against the baseline.
- The AI helped me understand that loop interchange strategies can trigger compiler-level floating point reduction changes or vector FMA optimizations that change the strict order of mathematical operations compared to the baseline loops. We tested caching blocks while retaining unrolled reduction loops to identify the exact cause.

**3. Reference Identification**
- The AI helped look up proper terminology for "SIMD Within A Register" (SWAR) and recommended strategies for algebraic math function approximation in single-precision (like `expf` instead of `exp`).

**4. Documentation Formatting**
- The AI helped format the final performance write-up and references into structured Markdown.
