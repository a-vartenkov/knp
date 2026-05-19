//
// Created by vartenkov on 29.04.26.
//

#pragma once
#include <knp/core/projection.h>
#include <algorithm>
#include <map>
#include <stdexcept>
#include <vector>

#include "vector.cuh"
#include "../cuda_bus/messaging.cuh"


namespace knp::backends::gpu::cuda::device_lib
{

struct IndexView
{
    const unsigned long long *indices_ptr_;
    const unsigned long long indices_size_;
    const unsigned long long *offsets_ptr_;
    const unsigned long long offsets_size_;
};


struct ValueIndex
{
    device_lib::CUDAVector<unsigned long long> indices_;
    device_lib::CUDAVector<unsigned long long> offsets_;

    __host__ __device__ IndexView view() const
    {
        return {indices_.data(), indices_.size(), offsets_.data(), offsets_.size()};
    }

    void insert(unsigned long long sender, unsigned long long index)
    {
        throw std::logic_error("Not implemented");
        // TODO Add insert
    }

    void remove(unsigned long long sender, unsigned long long index)
    {
        throw std::logic_error("Not implemented");
        // TODO Add remove
    }

    __host__ __device__ void actualize()
    {
        indices_.actualize();
        offsets_.actualize();
    }
};


// TODO maybe parallelize
template <class SynapseType>
__host__ ValueIndex build_index(const knp::core::Projection<SynapseType> &cpu_projection);

__global__ void summarize_index_kernel(IndexView index, device_lib::CUDAVectorView<cuda::SpikeIndex> senders,
                                       unsigned long long *result);

/**
 * @brief Calculates the total number of values for all impulses.
 * @param index The built value index
 * @param inputs the impulses to be use
 * @return
 */
__host__ unsigned long long count_values_by_indexes(const ValueIndex &index,
                                                    const CUDAVectorView<cuda::SpikeIndex> inputs);


template <class SynapseType>
__host__ ValueIndex build_index(const knp::core::Projection<SynapseType> &cpu_projection)
{
    // Build map-based index
    std::map<unsigned long long, std::vector<unsigned long long>> buffer;
    for (size_t i = 0; i < cpu_projection.size(); ++i)
    {
        const auto &synapse = cpu_projection[i];
        auto neuron_id = std::get<core::source_neuron_id>(synapse);
        auto map_iter = buffer.find(neuron_id);
        if (map_iter == buffer.end())
        {
            map_iter = buffer.insert(std::make_pair(neuron_id, std::vector<unsigned long long>{})).first;
        }
        map_iter->second.push_back(i);
    }
    unsigned long long current_offset = 0;
    unsigned long long last_neuron = buffer.rbegin()->first;

    std::vector<unsigned long long> indices(cpu_projection.size());
    std::vector<unsigned long long> offsets;
    offsets.reserve(last_neuron + 2);  // Neuron number + 1 is last_index + 2.
    offsets.push_back(0);
    unsigned long long first_neuron = 0;
    for (auto iter = buffer.begin(); iter != buffer.end(); ++iter)
    {
        unsigned long long current_neuron = (*iter).first;
        // Filling offsets for skipped neurons: if first three are missing that would be (0, 0, 0, 0, 5...
        for (auto i = first_neuron; i < current_neuron; ++i)
        {
            offsets.push_back(current_offset);
        }
        first_neuron = current_neuron;
        std::copy(iter->second.begin(), iter->second.end(), indices.begin() + current_offset);
        current_offset += iter->second.size();
        offsets.push_back(current_offset);
    }
    ValueIndex index;
    index.indices_ = indices;
    index.offsets_ = offsets;
    return index;
}

}  // namespace knp::backends::gpu::cuda::device_lib
