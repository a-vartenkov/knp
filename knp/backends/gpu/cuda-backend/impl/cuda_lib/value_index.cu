//
// Created by vartenkov on 15.05.26.
//
#include "value_index.cuh"

#include <thrust/transform.h>

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


// For each neuron index we find the number of connected synapses and write it to the result.
// Output of the same size as number of neurons
__global__ void gather_index_neuron_kernel(const IndexView index, const CUDAVectorView<cuda::SpikeIndex> inputs,
                                      unsigned long long *result)
{
    auto spike_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (spike_index > inputs.size_) return;
    SpikeIndex neuron = inputs.data_[spike_index];
    if (neuron > index.offsets_size_)
        result[spike_index] = 0; // Neuron out of index bounds: no connected synapses
    result[spike_index] = index.offsets_ptr_[neuron + 1] - index.offsets_ptr_[neuron];
}


// For each neuron id in "inputs" find the total number of synapses for "previous" neurons in "inputs".
// Output size() equals to index.size_
__host__ CUDAVector<unsigned long long> calculate_neuron_scan(const ValueIndex &index,
                                                              const CUDAVectorView<cuda::SpikeIndex> inputs)
{
    CuMallocAllocator<unsigned long long> allocator;
    unsigned long long *buffer = allocator.allocate(inputs.size_);
    auto [num_blocks, num_threads] = get_blocks_config(inputs.size_);
    gather_index_neuron_kernel<<<num_blocks, num_threads>>>(index.view(), inputs, buffer);
    // in-place prefix sum, for each neuron the value is the number of synapses before this, starts with 0.
    thrust::exclusive_scan(thrust::device, buffer, buffer + inputs.size_, buffer);
    return CUDAVector<unsigned long long>{buffer, inputs.size_}; // The vector would take care of releasing
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
