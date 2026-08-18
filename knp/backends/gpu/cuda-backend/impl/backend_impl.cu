/**
 * @file backend_impl.cu
 * @brief CUDABackendImpl backend class implementation.
 * @kaspersky_support Artiom N.
 * @date 24.02.2025
 * @license Apache 2.0
 * @copyright © 2025 AO Kaspersky Lab
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#include <knp/devices/gpu_cuda.h>
#include <knp/meta/assert_helpers.h>
#include <knp/meta/stringify.h>
#include <knp/meta/variant_helpers.h>

#include <spdlog/spdlog.h>

#include <limits>
#include <vector>

#include <boost/mp11.hpp>

#include <algorithm>

#include "backend_impl.cuh"
#include "projection.cuh"
#include "population.cuh"

#include "cuda_lib/fast_error_check.cuh"
#include "cuda_lib/get_blocks_config.cuh"
#include "cuda_lib/printf.cuh"
#include "cuda_lib/register_all.cuh"
#include "cuda_lib/vector.cuh"
#include "cuda_bus/messaging.cuh"
#include <thrust/binary_search.h>
#include <thrust/sort.h>


namespace knp::backends::gpu::cuda
{
// helper type for the visitor.
template<class... Ts>
struct overloaded : Ts ...
{
    using Ts::operator()...;
};
// explicit deduction guide.
template<class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;


template <>
CUDABackendImpl::PopulationVariants gpu_extract<CUDABackendImpl::PopulationVariants>(
        const CUDABackendImpl::PopulationVariants *);

template <>
void gpu_insert<CUDABackendImpl::PopulationVariants>(const CUDABackendImpl::PopulationVariants &,
                                                     CUDABackendImpl::PopulationVariants *);

template <>
ProjectionVariants gpu_extract<ProjectionVariants>(const ProjectionVariants *);

template <>
void gpu_insert<ProjectionVariants>(const ProjectionVariants &, ProjectionVariants *);

namespace detail
{
    template <class Variant, class Instance>
    __global__ void make_variant_kernel(Variant *result, Instance *source)
    {
        new (result) Variant(*source);
    }
}


template<class TypeVariant, size_t index>
TypeVariant extract_by_index(const void *type_ptr)
{
    return gpu_extract<boost::mp11::mp_at_c<TypeVariant, index>>(
            reinterpret_cast<const boost::mp11::mp_at_c<TypeVariant, index> *>(type_ptr));
}


// TODO: Make a template, it's also used for messages.

template<typename T>
__host__ __device__ void get_kernel(const T *var, int *type, const void **val)
{
    int type_val = var->index();
    static_assert(::cuda::std::variant_size<T>() == 1, "Incorrect variant size!");
    switch (type_val)
    {
        case 0:
            *val = ::cuda::std::get_if<0>(var);
            break;
        default:
            *val = nullptr;
    }
    *type = type_val;
}


__global__ void get_population_kernel(const CUDABackendImpl::PopulationVariants *var, int *type, const void **pop)
{
    get_kernel(var, type, pop);
}


__global__ void get_projection_kernel(const ProjectionVariants *var, int *type, const void **proj)
{
    get_kernel(var, type, proj);
}


template<>
void gpu_insert<CUDABackendImpl::PopulationVariants>(const CUDABackendImpl::PopulationVariants &cpu_source,
                                                     CUDABackendImpl::PopulationVariants *gpu_target)
{
    ::cuda::std::visit([gpu_target](const auto &val)
                       {
                           using ValueType = std::decay_t<decltype(val)>;
                           ValueType *buffer;
                           call_and_check(cudaMalloc(&buffer, sizeof(ValueType)));
                           gpu_insert(val, buffer);
                           device_lib::make_variant_kernel<<<1, 1>>>(gpu_target, buffer);
                           call_and_check(cudaFree(buffer));
                       }, cpu_source);
}





template<class T>
__global__ void get_uids_kernel(const T *data, size_t size, cuda::UID *result)
{
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= size) return;
    result[index] = ::cuda::std::visit([](auto &v) { return v.uid_; }, data[index]);
}


template<class VectorData>
device_lib::CUDAVector<cuda::UID> get_uids(const device_lib::CUDAVector<VectorData> &entities)
{
    device_lib::CUDAVector<cuda::UID> result(entities.size());
    if (entities.size() != 0)
    {
        auto [num_blocks, num_threads] = device_lib::get_blocks_config(entities.size());
        get_uids_kernel<<<num_blocks, num_threads>>>(entities.data(), entities.size(), result.data());
    }
    return result;
}


template<class VectorData>
device_lib::CUDAVector<cuda::UID> get_uids_std(const std::vector<VectorData> &entities)
{
    device_lib::CUDAVector<cuda::UID> result;
    result.reserve(entities.size());
    for (size_t i = 0; i < entities.size(); ++i)
    {
        ::cuda::std::visit([&result](const auto &entity)
                   {
                       result.push_back(entity.uid_);
                   }, entities[i]);
    }
    return result;
}


void CUDABackendImpl::calculate_projections(StepIndex step)
{
    // Calculate projections.
    device_lib::CUDAVector<cuda::UID> projection_uids = get_uids_std(device_projections_);

    if (!device_projections_.size()) return;

    std::vector<std::vector<device_lib::LongIndex>> projection_messages;
    projection_messages.reserve(device_projections_.size());
    for (size_t i = 0; i < device_projections_.size(); ++i)
    {
        const std::vector<device_lib::LongIndex> message_ids
            = device_message_bus_.unload_messages<cuda::SpikeMessage>(projection_uids.copy_at(i));
        projection_messages.push_back(message_ids);
    }
    assert(device_projections_.size() == projection_messages.size());
    for (size_t i = 0; i < device_projections_.size(); ++i)
    {
        ::cuda::std::visit([this, &projection_messages, step, i](auto &projection)
        {
            calculate_projection(projection, device_message_bus_, projection_messages[i], step);
        }, device_projections_[i]);
    }
    cudaDeviceSynchronize();
}


void CUDABackendImpl::load_populations(const knp::backends::gpu::CUDABackend::PopulationContainer &populations)
{
    SPDLOG_DEBUG("Loading populations [{}]...", populations.size());

    device_populations_.clear();
    device_populations_.reserve(populations.size());
    for (const auto &population : populations)
    {
        ::std::visit([this](auto &arg)
                     {
                         using CPUPopulationType = std::decay_t<decltype(arg)>;
                         auto pop = CUDAPopulation<typename CPUPopulationType::PopulationNeuronType>(arg);
                         device_populations_.push_back(pop);
                     }, population);
    }

    SPDLOG_DEBUG("All populations loaded.");
}


void CUDABackendImpl::load_projections(const knp::backends::gpu::CUDABackend::ProjectionContainer &projections)
{
    SPDLOG_DEBUG("Loading projections [{}]...", projections.size());
    CUDA_FAST_ERROR_CHECK("Starting to load, already an error: {}");
    device_projections_.clear();
    CUDA_FAST_ERROR_CHECK("Cleared device projections: {}");
    device_projections_.reserve(projections.size());
    CUDA_FAST_ERROR_CHECK("Reserving device projections: {}");

    for (const auto &projection : projections)
    {
        ::std::visit([this](auto &arg)
                     {
                         using CPUProjectionType = std::decay_t<decltype(arg)>;

                         auto proj = CUDAProjection<typename CPUProjectionType::ProjectionSynapseType>{arg};
            SPDLOG_DEBUG("Pushing back a projection, size before: {}, pointer before: {}, capacity {}",
                         device_projections_.size(),
                         reinterpret_cast<void *>(device_projections_.data()),
                         device_projections_.capacity());
                         device_projections_.push_back(proj);
            CUDA_FAST_ERROR_CHECK("Pushed back {}");
            SPDLOG_DEBUG("Pushed back: size after: {}, pointer after: {}, capacity {}", device_projections_.size(),
                             reinterpret_cast<void *>(device_projections_.data()), device_projections_.capacity());

                     }, projection);
    }

    SPDLOG_DEBUG("All projections loaded.");
}


void CUDABackendImpl::init()
{
    SPDLOG_DEBUG("Initializing CUDABackendImpl...");

    // knp::backends::cpu::init(projections_, get_message_endpoint());
    for (size_t i = 0; i < device_projections_.size(); ++i)
    {
        const auto [pre_uid, post_uid, this_uid] = ::cuda::std::visit([](auto &proj)
            {
                return std::make_tuple(proj.presynaptic_uid_, proj.postsynaptic_uid_, proj.uid_);
            }, device_projections_[i]);
        if (!cuda::empty_uid(pre_uid)) this->device_message_bus_.subscribe_gpu<cuda::SpikeMessage>(this_uid, {pre_uid});
        if (!cuda::empty_uid(post_uid))
        {
            this->device_message_bus_.subscribe_gpu<cuda::SynapticImpactMessage>(post_uid, {this_uid});
        }
    }

    SPDLOG_DEBUG("Initialization finished.");
}


using BlifatParams = ::knp::neuron_traits::neuron_parameters<::knp::neuron_traits::BLIFATNeuron>;


__global__ void calculate_neurons_pre_impact(device_lib::CUDAVectorMutableView<BlifatParams> neurons,
                                             StepIndex current_step)
{
    size_t neuron_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron_index >= neurons.size_) return;

    BlifatParams &neuron = neurons.data_[neuron_index];
    ++neuron.n_time_steps_since_last_firing_;
    neuron.dynamic_threshold_ *= neuron.threshold_decay_;
    neuron.postsynaptic_trace_ *= neuron.postsynaptic_trace_decay_;
    neuron.inhibitory_conductance_ *= neuron.inhibitory_conductance_decay_;

    /*
    if constexpr (has_dopamine_plasticity<BlifatLikeNeuron>())
    {
        neuron.dopamine_value_ = 0.0;
        neuron.is_being_forced_ = false;
    }
    */

    if (neuron.bursting_phase_ && !--neuron.bursting_phase_)
    {
        neuron.potential_ = neuron.potential_ * neuron.potential_decay_ + neuron.reflexive_weight_;
    }
    else
    {
        neuron.potential_ *= neuron.potential_decay_;
    }
    neuron.pre_impact_potential_ = neuron.potential_;
}


__global__ void calculate_neurons_impacts(device_lib::CUDAVectorMutableView<BlifatParams> neurons,
                                          device_lib::CUDAVectorView<SynapticImpact> impacts)
{
    size_t impact_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (impact_index >= impacts.size_) return;
    const SynapticImpact &impact = impacts.data_[impact_index];
    if (impact.postsynaptic_neuron_index_ >= neurons.size_) return;
    auto &neuron = neurons.data_[impact.postsynaptic_neuron_index_];
    switch (impact.synapse_type_)
    {
        case knp::synapse_traits::OutputType::EXCITATORY:
            atomicAdd(&neuron.potential_, impact.impact_value_);
            break;
        case knp::synapse_traits::OutputType::INHIBITORY_CURRENT:
            atomicAdd(&neuron.potential_, -impact.impact_value_);
            break;
        case knp::synapse_traits::OutputType::INHIBITORY_CONDUCTANCE:
            atomicAdd(&neuron.inhibitory_conductance_, impact.impact_value_);
            break;
        case knp::synapse_traits::OutputType::DOPAMINE:
            atomicAdd(&neuron.dopamine_value_, impact.impact_value_);
            break;
        case knp::synapse_traits::OutputType::BLOCKING:
            neuron.total_blocking_period_ = static_cast<unsigned int>(impact.impact_value_);
            break;
    }
}


__host__ void calculate_neurons_impacts_all(device_lib::CUDAVectorMutableView<BlifatParams> neurons,
                                             const std::vector<SynapticImpactMessage> &messages_all,
                                             std::vector<device_lib::LongIndex> message_ids)
{
    for (const auto &msg_id : message_ids)
    {
        const auto &msg = messages_all[msg_id];
        auto [num_blocks, num_threads] = device_lib::get_blocks_config(msg.impacts_.size());
        calculate_neurons_impacts<<<num_blocks, num_threads>>>(neurons, msg.impacts_.view());
    }
    cudaDeviceSynchronize();
}


__global__ void calculate_neurons_post_impact(device_lib::CUDAVectorMutableView<BlifatParams> neurons,
                                              SpikeIndex *spike_buffer, SpikeIndex *size_counter)
{
    size_t neuron_index = blockIdx.x * blockDim.x + threadIdx.x;
    bool spike = false;
    neuron_traits::neuron_parameters <neuron_traits::BLIFATNeuron> &neuron = neurons.data_[neuron_index];
    if (neuron.total_blocking_period_ <= 0)
    {
        // TODO: Make it more readable, don't be afraid to use if operators.
        // Restore potential that the neuron had before impacts.
        neuron.potential_ = neuron.pre_impact_potential_;
        bool was_negative = neuron.total_blocking_period_ < 0;
        // If it is negative, increase by 1.
        neuron.total_blocking_period_ += was_negative;
        // If it is now zero, but was negative before, increase it to max, else leave it as is.
        neuron.total_blocking_period_ +=
                std::numeric_limits<int64_t>::max() * ((neuron.total_blocking_period_ == 0) && was_negative);
    }
    else
    {
        neuron.total_blocking_period_ -= 1;
    }

    if (neuron.inhibitory_conductance_ < 1.0)
    {
        neuron.potential_ -=
                (neuron.potential_ - neuron.reversal_inhibitory_potential_) *
                neuron.inhibitory_conductance_;
    }
    else
    {
        neuron.potential_ = neuron.reversal_inhibitory_potential_;
    }

    if ((neuron.n_time_steps_since_last_firing_ > neuron.absolute_refractory_period_) &&
        (neuron.potential_ >= neuron.activation_threshold_ + neuron.dynamic_threshold_))
    {
        // Spike.
        neuron.dynamic_threshold_ += neuron.threshold_increment_;
        neuron.postsynaptic_trace_ += neuron.postsynaptic_trace_increment_;

        neuron.potential_ = neuron.potential_reset_value_;
        neuron.bursting_phase_ = neuron.bursting_period_;
        neuron.n_time_steps_since_last_firing_ = 0;
        spike = true;
    }

    if (neuron.potential_ < neuron.min_potential_)
    {
        neuron.potential_ = neuron.min_potential_;
    }
    if (spike)
    {
        SpikeIndex counter = atomicAdd(size_counter, 1);
        spike_buffer[counter] = neuron_index;
    }
}


device_lib::CUDAVector<SpikeIndex> CUDABackendImpl::calculate_population(
        CUDAPopulation<knp::neuron_traits::BLIFATNeuron> &population, StepIndex step)
{
    auto [num_blocks_neuro, num_threads_neuro] = device_lib::get_blocks_config(population.neurons_.size());

    calculate_neurons_pre_impact<<<num_blocks_neuro, num_threads_neuro>>>(population.neurons_.mut_view(), step);
    std::vector<device_lib::LongIndex> message_ids
            = device_message_bus_.unload_messages<cuda::SynapticImpactMessage>(population.uid_);

    if (!message_ids.empty())
    {
        SPDLOG_DEBUG("Running calculate impacts on {} messages", message_ids.size());
        calculate_neurons_impacts_all(population.neurons_.mut_view(),
                                      device_message_bus_.all_messages<SynapticImpactMessage>(),
                                      message_ids);
    }
    SpikeIndex *output;
    cudaMalloc(&output, sizeof(SpikeIndex) * population.neurons_.size());
    SpikeIndex *counter;
    cudaMalloc(&counter, sizeof(SpikeIndex));
    cudaMemset(counter, 0, sizeof(SpikeIndex));
    calculate_neurons_post_impact<<<num_blocks_neuro, num_threads_neuro>>>(population.neurons_.mut_view(), output,
                                                                           counter);

    SpikeIndex out_size = 0;
    cudaMemcpy(&out_size, counter, sizeof(SpikeIndex), cudaMemcpyDeviceToHost);
    // Capacity would be "out_size" while the ptr is larger, but that doesn't matter as the pointer is freed as a whole.
    device_lib::CUDAVector<SpikeIndex> result{output, out_size};
    cudaFree(counter);
    return result;
}


void CUDABackendImpl::calculate_populations(StepIndex step)
{
    // device_message_bus_.clear<SpikeMessage>();
    for (auto &population : device_populations_)
    {
        ::cuda::std::visit([this, step](auto &pop)
        {
            auto spikes = calculate_population(pop, step);
            if (!spikes.empty())
            {
                cuda::SpikeMessage message{MessageHeader{pop.uid_, step, false}, std::move(spikes)};
                SPDLOG_DEBUG("Population {} created a message with {} spikes", std::string(to_cpu_uid(pop.uid_)),
                             message.neuron_indexes_.size());
                device_message_bus_.send_message(std::move(message));
            }
            else
            {
                SPDLOG_DEBUG("No spike messages were sent from population {}", ::std::string(to_cpu_uid(pop.uid_)));
            }
        }, population);
    }
}


__host__ uint64_t CUDABackendImpl::route_projection_messages(StepIndex step)
{
    using MessageVector = device_lib::CUDAVector<cuda::MessageVariant>;
    device_lib::LongIndex sent_message_counter = 0;
    device_message_bus_.clear();

    for (size_t i = 0; i < device_projections_.size(); ++i)
    {
        ::cuda::std::visit([this, &sent_message_counter](auto &proj)
               {
                   if (proj.message_buf_.impacts_.size())
                   {
                       device_message_bus_.send_message(std::move(proj.message_buf_));
                       // cudaDeviceSynchronize();
                       proj.message_buf_.impacts_.clear();
                       ++sent_message_counter;
                   }
               }, device_projections_[i]);
    }
    cudaDeviceSynchronize();
    SPDLOG_DEBUG("Projections sent {} messages", sent_message_counter);
    return sent_message_counter;
}





__host__ CUDABackendImpl::PopulationIterator CUDABackendImpl::begin_populations()
{
    return PopulationIterator{device_populations_.begin()};
}


__host__ CUDABackendImpl::PopulationConstIterator CUDABackendImpl::begin_populations() const
{
    return {device_populations_.cbegin()};
}


__host__ CUDABackendImpl::PopulationIterator CUDABackendImpl::end_populations()
{
    return PopulationIterator{device_populations_.end()};
}


__host__ CUDABackendImpl::PopulationConstIterator CUDABackendImpl::end_populations() const
{
    return device_populations_.cend();
}


__host__ CUDABackendImpl::ProjectionIterator CUDABackendImpl::begin_projections()
{
    return ProjectionIterator{device_projections_.begin()};
}


__host__ CUDABackendImpl::ProjectionConstIterator CUDABackendImpl::begin_projections() const
{
    return device_projections_.cbegin();
}


__host__ CUDABackendImpl::ProjectionIterator CUDABackendImpl::end_projections()
{
    return ProjectionIterator{device_projections_.end()};
}


__host__ CUDABackendImpl::ProjectionConstIterator CUDABackendImpl::end_projections() const
{
    return device_projections_.cend();
}


__global__ void get_spike_message_data(device_lib::CUDAVectorView<cuda::MessageVariant> all_messages,
               device_lib::LongIndex msg_index, device_lib::LongIndex *size, const SpikeIndex **data_pointer)
{
    constexpr size_t spike_message_index = boost::mp11::mp_find<cuda::MessageVariant, cuda::SpikeMessage>();
    auto &message_var = all_messages.data_[msg_index];
    if (message_var.index() != spike_message_index)
    {
        *size = 0;
        *data_pointer = nullptr;
        PRINTF_DEBUG("Pointer (no msg): %p, size: %lu\n", *data_pointer, *size);
        return;
    }
    *data_pointer = ::cuda::std::get<cuda::SpikeMessage>(message_var).neuron_indexes_.data();
    *size = ::cuda::std::get<cuda::SpikeMessage>(message_var).neuron_indexes_.size();
#ifdef DEBUG
    printf("Pointer: %p, size: %lu\n", *data_pointer, *size);
    for (size_t i = 0; i < *size; ++i)
    {
        printf("%u ",  ::cuda::std::get<cuda::SpikeMessage>(message_var).neuron_indexes_.data()[i]);
    }
    printf("\n");
#endif // DEBUG
}

}   // namespace knp::backends::gpu::cuda

REGISTER_CUDA_VECTOR_TYPE(knp::backends::gpu::cuda::CUDABackendImpl::PopulationVariants);
REGISTER_CUDA_VECTOR_TYPE(knp::backends::gpu::cuda::ProjectionVariants);
REGISTER_CUDA_VECTOR_TYPE(knp::backends::gpu::cuda::SynapticImpact);
REGISTER_CUDA_VECTOR_TYPE(knp::backends::gpu::cuda::CUDAPopulation<knp::neuron_traits::BLIFATNeuron>);
REGISTER_CUDA_VECTOR_TYPE(knp::backends::gpu::cuda::CUDAProjection<knp::synapse_traits::DeltaSynapse>);
REGISTER_CUDA_VECTOR_TYPE(knp::backends::gpu::cuda::CUDAProjection<knp::synapse_traits::DeltaSynapse>::Synapse);
REGISTER_CUDA_VECTOR_TYPE(knp::backends::gpu::cuda::CUDAPopulation<knp::neuron_traits::BLIFATNeuron>::NeuronParameters);
REGISTER_CUDA_VECTOR_TYPE(knp::backends::gpu::cuda::device_lib::CUDAVector<unsigned long long>);
REGISTER_CUDA_VECTOR_TYPE(unsigned int);
REGISTER_CUDA_VECTOR_TYPE(unsigned long long);
REGISTER_CUDA_VECTOR_TYPE(knp::backends::gpu::cuda::Subscription);
REGISTER_CUDA_VECTOR_TYPE(knp::backends::gpu::cuda::MessageVariant);
REGISTER_CUDA_VECTOR_TYPE(knp::backends::gpu::cuda::UID);
