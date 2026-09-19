# Vendored oMLX GLM runtime (glm5_next + glm_moe_dsa + deepseek_v4 deps)

Vendored from **oMLX** (`https://github.com/jonathan308/omlx`) at commit
`0cb0e58c73869c91696e0769472e80bffdde80ba` (2026-08-28), Apache License 2.0
(`LICENSE` in this directory). Vendored because oMLX is not published as an
installable package; the code here is the minimum surface needed to serve
GLM-5.3-Flash (`glm5_next`) artifacts inside MTPLX — model definitions,
KDA/DSA attention, HyperConnection plumbing, and the native-MTP runtime.

Layout:

- `glm5_next/` — GLM-5.3 model (VLM wrapper + text stack), from
  `omlx/patches/mlx_vlm_mtp/glm5_next*` plus the upstream `mlx_vlm` glm5_next
  module it extends.
- `glm_moe_dsa/` — GLM MoE-DSA model + kernel dispatch (`fast.py`,
  `kernels.py`) used by the glm5_next text stack.
- `deepseek_v4/` — shared helpers (`cache_extras` PoolingCache et al.).
- `common/csrc/` — the three shared kernel headers the glm_moe_dsa CMake
  project expects at `../../common/csrc` (`quantized_moe.h`,
  `steel_attention_block_token.h`, and an MLX `steel/attn/params.h`
  override), from `omlx/custom_kernels/common/csrc/`.
- `cache_rollback.py` — oMLX rollback utilities used by MTP verify.

## Native kernels

`glm_moe_dsa/csrc/` carries the Metal/C++ kernel sources from
`omlx/custom_kernels/glm_moe_dsa/csrc/` (CMake project; builds
`_ext.cpython-*.so`, `libomlx_glm_kernel_ops.dylib`,
`omlx_glm_kernels.metallib` into this directory). Prebuilt binaries are
NOT shipped — they are Python-version- and arch-locked (built for
cp313/arm64 in our dev environment). Without the extension the dispatch
falls back to `mx.fast` implementations, which is correct but slower.

Build (macOS arm64, requires Xcode CLT + CMake):

    cd mtplx/vendor/glm5_omlx/glm_moe_dsa/csrc
    cmake -B build && cmake --build build -j
    cp build/_ext.cpython-*.so build/libomlx_glm_kernel_ops.dylib \
       build/omlx_glm_kernels.metallib ..

Modifications vs upstream oMLX are marked `# MTPLX:` where touched; the
rest is verbatim. Fixes to the vendored code should be offered back
upstream.
