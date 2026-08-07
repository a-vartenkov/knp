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

using LongIndex = unsigned long long;

struct IndexView
{
    const LongIndex *indices_ptr_;
    const LongIndex indices_size_;
    const LongIndex *offsets_ptr_;
    const LongIndex offsets_size_;
};


struct ValueIndex
{
    device_lib::CUDAVector<LongIndex> indices_;
    device_lib::CUDAVector<LongIndex> offsets_;

    __host__ __device__ IndexView view() const
    {
        return {indices_.data(), indices_.size(), offsets_.data(), offsets_.size()};
    }

    void insert(LongIndex sender, LongIndex index)
    {
        throw std::logic_error("Not implemented");
        // TODO Add insert
    }

    void remove(LongIndex sender, LongIndex index)
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
                                       LongIndex *result);

/**
 * @brief Calculates the total number of values for all impulses.
 * @param index The built value index
 * @param inputs the impulses to be use
 * @return
 */
__host__ LongIndex count_values_by_indexes(const ValueIndex &index, const CUDAVectorView<cuda::SpikeIndex> inputs);

/**
 * @brief Build exclusive prefix sum for number of synapses per neuron.
 * @param index the built value index.
 * @param inputs spiked neuron indices.
 * @return a vector of "output" offsets for each neuron.
 */
__host__ CUDAVector<LongIndex> calculate_neuron_scan(const ValueIndex &index,
                                                     const CUDAVectorView<cuda::SpikeIndex> inputs);


template <class SynapseType>
__host__ ValueIndex build_index(const knp::core::Projection<SynapseType> &cpu_projection)
{
    // Build map-based index
    std::map<LongIndex, std::vector<LongIndex>> buffer;
    for (size_t i = 0; i < cpu_projection.size(); ++i)
    {
        const auto &synapse = cpu_projection[i];
        auto neuron_id = std::get<core::source_neuron_id>(synapse);
        auto map_iter = buffer.find(neuron_id);
        if (map_iter == buffer.end())
        {
            map_iter = buffer.insert(std::make_pair(neuron_id, std::vector<LongIndex>{})).first;
        }
        map_iter->second.push_back(i);
    }
    LongIndex current_offset = 0;
    LongIndex last_neuron = buffer.rbegin()->first;

    std::vector<LongIndex> indices(cpu_projection.size());
    std::vector<LongIndex> offsets;
    offsets.reserve(last_neuron + 2);  // Neuron number + 1 is last_index + 2.
    offsets.push_back(0);
    LongIndex first_neuron = 0;
    for (auto iter = buffer.begin(); iter != buffer.end(); ++iter)
    {
        LongIndex current_neuron = (*iter).first;
        // Filling offsets for skipped neurons: if first three are missing that would be (0, 0, 0, 0, 5...
        for (auto i = first_neuron; i < current_neuron; ++i)
        {
            offsets.push_back(current_offset);
        }
        first_neuron = current_neuron + 1;
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
