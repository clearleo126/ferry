cmake_minimum_required(VERSION 3.18)
project(Ferry LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA 架构外部传入，不硬编码：
#   WSL 4060 Laptop: cmake -DCMAKE_CUDA_ARCHITECTURES=89 ..
#   集群 4x3090:     cmake -DCMAKE_CUDA_ARCHITECTURES=86 ..
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra)

find_package(CUDAToolkit REQUIRED)

# ------------------------------------------------------------------
# 模块库
# ------------------------------------------------------------------

# common: 错误检查宏、计时工具
add_library(ferry_common INTERFACE)
target_include_directories(ferry_common INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src)

# config: 环境变量/配置解析（纯 C++，无 CUDA 依赖）
add_library(ferry_config STATIC
  src/config/config.cpp
)
target_include_directories(ferry_config PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(ferry_config PUBLIC ferry_common)

# comm: host-staged 通信层（pinned ring + 双 stream 流水）
add_library(ferry_comm STATIC
  src/comm/host_staged_channel.cpp
)
target_include_directories(ferry_comm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(ferry_comm PUBLIC ferry_common CUDA::cudart)

# topo: 拓扑/设备探测（依赖 CUDA driver/runtime API）
add_library(ferry_topo STATIC
  src/topo/topo.cpp
)
target_include_directories(ferry_topo PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(ferry_topo PUBLIC ferry_config CUDA::cudart)

find_package(CUDAToolkit REQUIRED)

# ------------------------------------------------------------------
# 可执行程序
# ------------------------------------------------------------------

# probe: 拓扑探测 + host-staged 吞吐基准（P0 工具，WSL/集群通用）
add_executable(probe
  src/probe/main.cu
)
target_link_libraries(probe PRIVATE ferry_topo ferry_config ferry_common)

# bench_comm: 通信层验收（正确性 + 吞吐对比 P1 batched）
add_executable(bench_comm
  src/benches/bench_comm.cu
)
target_link_libraries(bench_comm PRIVATE ferry_comm ferry_topo ferry_config ferry_common)

# queue: GPU 本地有界环队列（header-only device code）
add_library(ferry_queue INTERFACE)
target_include_directories(ferry_queue INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(ferry_queue INTERFACE ferry_common)

# bench_queue: 本地队列验收（正确性 + per-thread vs wave-batched 吞吐）
add_executable(bench_queue
  src/benches/bench_queue.cu
)
target_link_libraries(bench_queue PRIVATE ferry_queue ferry_config ferry_common)

# runtime: 批处理跨卡重分配 + 批粒度自适应（header-only）
add_library(ferry_runtime INTERFACE)
target_include_directories(ferry_runtime INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(ferry_runtime INTERFACE ferry_comm ferry_config ferry_common)

# bench_redistribute: 重分配验收（H1 批处理收益 / H2 批粒度自适应）
add_executable(bench_redistribute
  src/benches/bench_redistribute.cu
)
target_link_libraries(bench_redistribute PRIVATE ferry_runtime ferry_comm ferry_config ferry_common)

# bench_overlap: 计算通信重叠验收（H3）
add_executable(bench_overlap
  src/benches/bench_overlap.cu
)
target_link_libraries(bench_overlap PRIVATE ferry_runtime ferry_comm ferry_config ferry_common)

# workloads: 合成任务流 + baseline A/B/C 执行器（header-only）
add_library(ferry_workloads INTERFACE)
target_include_directories(ferry_workloads INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(ferry_workloads INTERFACE ferry_comm ferry_runtime ferry_config ferry_common)

# run_synthetic: Workload B 端到端实验（吞吐 + p50/p99 + 通信量）
add_executable(run_synthetic
  src/benches/run_synthetic.cu
)
target_link_libraries(run_synthetic PRIVATE ferry_workloads ferry_comm ferry_runtime ferry_config ferry_common)

# run_missstream: Workload A 端到端实验（QSA-style miss 流 × A/B/C）
add_executable(run_missstream
  src/benches/run_missstream.cu
)
target_link_libraries(run_missstream PRIVATE ferry_workloads ferry_comm ferry_runtime ferry_config ferry_common)

# run_ablation: R4 消融矩阵（M1×M2×M3 开关全组合 + CSV）
add_executable(run_ablation
  src/benches/run_ablation.cu
)
target_link_libraries(run_ablation PRIVATE ferry_workloads ferry_comm ferry_runtime ferry_config ferry_common)

# run_multigpu: N 卡控制面（A/B/C × 到达偏斜，水位批窃取）
add_executable(run_multigpu
  src/benches/run_multigpu.cu
)
target_link_libraries(run_multigpu PRIVATE ferry_workloads ferry_comm ferry_runtime ferry_config ferry_common)

# ------------------------------------------------------------------
# 后续模块（P2 中逐步添加，先占位说明）
#   workloads/: Workload A/B 生成器
#   benches/: baseline A/B/C 与微基准
# ------------------------------------------------------------------