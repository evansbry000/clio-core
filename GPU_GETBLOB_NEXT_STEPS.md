# GPU CTE GetBlob Data Mismatch - Next Steps

## Status
Branch: `74-fix-context-transport-primitives-for-the-gpu`
Commit: `4af6ca1` - Fixed GpuMetadata destruction on pause/resume

## What was fixed
- `Resume()` no longer bumps `scratch_gen`, no longer memsets scratch headers, sets `skip_scratch_init = true`
- `EnsureMetaInit()` no longer invalidates `meta_` on generation change
- Removed unused `meta_gen_` field from `GpuRuntime`
- GPU-initiated PutBlob now succeeds (was rc=7 "no targets", now rc=0)

## Current test results (test_gpu_core)
- PASS: GPU PutBlob via LocalGpuBcast
- FAIL: GPU GetBlob via LocalGpuBcast (data_ok fails - output buffer doesn't match 0xCD pattern)
- PASS: GPU GetOrCreateTag via LocalGpuBcast
- FAIL: GPU-Initiated PutBlob+GetBlob (result=-4, data mismatch after successful put+get)
- PASS: AsyncPutBlob Local 4KB (CPU path)
- PASS: AsyncGetBlob Local 4KB (CPU path)

## Root cause analysis for remaining failures

Both failures are **data mismatch** - bdev Read returns rc=0 but output buffer has wrong data.

### Key observations
1. PutBlob succeeds (rc=0) in both GPU tests
2. GetBlob succeeds (rc=0) in both GPU tests
3. CPU-path Put+Get roundtrip works perfectly
4. Only GPU-path reads return wrong data

### Likely cause: blob_data_ ShmPtr resolution in bdev GPU Read

The data flow is:
1. CPU test creates pinned buffer via `cudaMallocHost`, encodes as `ShmPtr::FromRaw(ptr)` (null alloc_id + raw addr)
2. `SendToGpu` serializes `blob_data_` ShmPtr into copy_space via `ar.bulk(blob_data_, size_, BULK_EXPOSE)`
3. GPU bdev `Read()` calls `CHI_IPC->ToFullPtr(task->data_)` to resolve the output pointer
4. GPU bdev copies data from pinned storage into resolved pointer

### Files to investigate
- `context-runtime/modules/bdev/src/bdev_runtime_gpu.cc:194` - GPU Read implementation
- `context-transfer-engine/core/src/core_runtime_gpu.cc:545` - GPU GetBlob
- `context-transfer-engine/core/include/wrp_cte/core/core_tasks.h:1185` - GetBlobTask SerializeIn (`BULK_EXPOSE`)
- `context-transfer-engine/core/include/wrp_cte/core/core_tasks.h:1195` - GetBlobTask SerializeOut (`BULK_XFER`)
- `context-runtime/include/chimaera/ipc_manager.h:1803` - ToFullPtr GPU path

### Things to check
1. Is `blob_data_` correctly deserialized on GPU after `LocalSaveTaskArchive` serialization?
2. Does `BULK_EXPOSE` correctly pass through the null-alloc-id ShmPtr for GPU resolution?
3. After bdev Read copies data into the pinned buffer, is there a system-scope fence before FUTURE_COMPLETE is set? (CPU->GPU tasks use `SetBitsSystem` which should be correct)
4. Is the bdev Read writing to the correct destination pointer, or is ToFullPtr resolving to a different address?
5. Add debug prints in bdev GPU Read to verify src/dst pointers and copy sizes

### Build notes
- Container has clang-18 only; symlinks created at `/usr/bin/clang` and `/usr/bin/clang++`
- `libomp-dev` is NOT installed (needed for OpenMP with clang)
- Use `cmake .. -DCMAKE_C_COMPILER=/usr/bin/clang -DCMAKE_CXX_COMPILER=/usr/bin/clang++` with cuda-debug preset options
- Or install clang-20 in the rebuilt container (was originally used)
