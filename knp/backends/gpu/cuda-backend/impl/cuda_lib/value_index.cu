//
// Created by vartenkov on 15.05.26.
//
#include "value_index.cuh"

namespace knp::backends::gpu::cuda::device_lib
{
// TODO maybe parallelize

/**
 * @brief Calculates the total number of values for all impulses.
 * @param index The built value index
 * @param inputs the impulses to be use
 * @return
 */
__host__ unsigned long long count_values_by_indexes(const ValueIndex &index,
                                                    const CUDAVectorView<cuda::SpikeIndex> inputs)
{
    SPDLOG_DEBUG("Count values by indexes");
    auto [num_blocks, num_threads] = get_blocks_config(inputs.size_);
    unsigned long long *result;
    call_and_check(cudaMalloc(&result, sizeof(unsigned long long)));
    call_and_check(cudaMemset(result, 0, sizeof(unsigned long long)));
    summarize_index_kernel<<<num_blocks, num_threads>>>(index.view(), inputs, result);
    unsigned long long out_result;
    call_and_check(cudaMemcpy(&out_result, result, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    cudaFree(result);
    return out_result;
}


__global__ void summarize_index_kernel(IndexView index, device_lib::CUDAVectorView<cuda::SpikeIndex> senders,
                                       unsigned long long *result)
{
    unsigned long long thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long value = 0;
    if (index.offsets_size_ != 0 && thread_id < index.offsets_size_ - 1)
    {
        value = index.offsets_ptr_[thread_id + 1] - index.offsets_ptr_[thread_id];
    }
    else return;
    atomicAdd(result, value);
}
} // namespace knp::backends::gpu::cuda::device_lib
