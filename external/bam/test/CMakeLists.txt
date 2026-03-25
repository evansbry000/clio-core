cmake_minimum_required(VERSION 3.20)

# Basic BaM test: exercises host-memory fallback path
add_executable(test_bam_basic test_bam_basic.cu)
target_link_libraries(test_bam_basic PRIVATE bam)
set_target_properties(test_bam_basic PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_STANDARD 17
  CXX_STANDARD 17
)
target_compile_options(test_bam_basic PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)
