cmake_minimum_required(VERSION 3.10)


# Create client library
add_chimod_client(
  SOURCES
    src/admin_client.cc
)

# Create runtime library
# Note: src/admin_runtime_gpu.cc is automatically detected by add_chimod_runtime
# and compiled as a GPU companion library when CUDA/ROCm is enabled
add_chimod_runtime(
  SOURCES
    src/admin_runtime.cc
    src/autogen/admin_lib_exec.cc
    src/admin_runtime_gpu.cc
)

# Add unit tests subdirectory
if(WRP_CORE_ENABLE_TESTS)
  add_subdirectory(test)
endif()