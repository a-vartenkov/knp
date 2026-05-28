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

#include "backend_impl_alt.cuh"
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


template<class ProjectionType>
__host__ __device__ inline bool is_forcing()
{
    return false;
}


template <>
CUDABackendImpl::PopulationVariants gpu_extract<CUDABackendImpl::PopulationVariants>(
        const CUDABackendImpl::PopulationVariants *);

template <>
void gpu_insert<CUDABackendImpl::PopulationVariants>(const CUDABackendImpl::PopulationVariants &,
                                                     CUDABackendImpl::PopulationVariants *);

template <>
CUDABackendImpl::ProjectionVariants gpu_extract<CUDABackendImpl::ProjectionVariants>(
        const CUDABackendImpl::ProjectionVariants *);

template <>
void gpu_insert<CUDABackendImpl::ProjectionVariants>(const CUDABackendImpl::ProjectionVariants &,
                                                     CUDABackendImpl::ProjectionVariants *);

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


__global__ void get_projection_kernel(const CUDABackendImpl::ProjectionVariants *var, int *type, const void **proj)
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


template<>
void gpu_insert<CUDABackendImpl::ProjectionVariants>(const CUDABackendImpl::ProjectionVariants &cpu_source,
                                                     CUDABackendImpl::ProjectionVariants *gpu_target)
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


template<>
__host__ __device__ inline bool is_forcing<cuda::CUDAProjection<synapse_traits::DeltaSynapse>>()
{
    return true;
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


/// @note: out_messages_data should be of size to contain at least num_populations messages.
__global__ void calculate_populations_kernel(CUDABackendImpl::PopulationVariants *populations, size_t num_populations,
                                             const cuda::MessageVariant *messages, size_t messages_size,
                                             const cuda::device_lib::CUDAVector<unsigned long long> *indices,
                                             size_t indices_size,
                                             cuda::MessageVariant *out_messages_data, unsigned long long step)
{
    // Calculate populations. This is the same as inference.
    size_t thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_index >= num_populations) return;

    CUDABackendImpl::PopulationVariants &population = populations[thread_index];
    knp::backends::gpu::cuda::device_lib::CUDAVector<cuda::MessageVariant> new_messages;
    PRINTF_TRACE("Population index: %lu\n", population.index());

    size_t num_messages = indices[thread_index].size();
    PRINTF_TRACE("Num messages: %lu\n", num_messages);
    for (size_t n = 0; n < num_messages; ++n)
    {
        unsigned long long message_index = indices[thread_index][n];
        if (message_index >= messages_size) continue;
        PRINTF_TRACE("Population messages size: %lu, message index: %lu\n", messages_size,
                     message_index);
        new_messages.push_back(messages[message_index]);
    }

    auto message = ::cuda::std::visit([&new_messages, step](auto &pop)
                                      {
                                          PRINTF_TRACE("Population messages size: %lu\n", new_messages.size());
                                          return CUDABackendImpl::calculate_population(pop, new_messages, step);
                                      }, population);

    if (message) out_messages_data[thread_index] = cuda::MessageVariant{message.value()};
}


void CUDABackendImpl::calculate_populations(unsigned long long step)
{
    // Calculate populations. This is the same as inference.
    using MessageVector = device_lib::CUDAVector<cuda::MessageVariant>;
    if (!device_populations_.size()) return;

    device_lib::CUDAVector<cuda::UID> population_uids = get_uids(device_populations_);
    auto [num_blocks, num_threads] = device_lib::get_blocks_config(device_populations_.size());

    device_lib::CUDAVector<device_lib::CUDAVector<unsigned long long>> population_messages(device_populations_.size());

    for (size_t i = 0; i < device_populations_.size(); ++i)
    {
        const device_lib::CUDAVector<unsigned long long> message_ids =
                device_message_bus_.unload_messages<SynapticImpactMessage>(
                    population_uids.copy_at(i));
        gpu_insert(message_ids, population_messages.data() + i);
    }

    MessageVector out_messages(device_populations_.size());
    assert(device_populations_.size() == population_messages.size());
    calculate_populations_kernel<<<num_blocks, num_threads>>>(device_populations_.data(), device_populations_.size(),
                                                              device_message_bus_.all_messages().data(),
                                                              device_message_bus_.all_messages().size(),
                                                              population_messages.data(), population_messages.size(),
                                                              out_messages.data(), step);
    cudaDeviceSynchronize();
    SPDLOG_DEBUG("Sending {} spike messages.", out_messages.size());
    device_message_bus_.send_message_gpu_batch(out_messages);
}


void CUDABackendImpl::calculate_projections(unsigned long long step)
{
    // Calculate projections.
    device_lib::CUDAVector<cuda::UID> projection_uids = get_uids_std(device_projections_);

    if (!device_projections_.size()) return;

    std::vector<device_lib::CUDAVector<unsigned long long>> projection_messages;
    projection_messages.reserve(device_projections_.size());
    for (size_t i = 0; i < device_projections_.size(); ++i)
    {
        const device_lib::CUDAVector<unsigned long long> message_ids
            = device_message_bus_.unload_messages<cuda::SpikeMessage>(projection_uids.copy_at(i));
        projection_messages.push_back(message_ids);
    }
    assert(device_projections_.size() == projection_messages.size());
    for (size_t i = 0; i < device_projections_.size(); ++i)
    {
        ::cuda::std::visit([this, &projection_messages, step, i](auto &projection)
        {
            calculate_projection(projection, projection_messages[i], step);
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
    FAST_ERROR_CHECK("Starting to load, already an error: {}");
    device_projections_.clear();
    FAST_ERROR_CHECK("Cleared device projections: {}");
    device_projections_.reserve(projections.size());
    FAST_ERROR_CHECK("Reserving device projections: {}");

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
            FAST_ERROR_CHECK("Pushed back {}");
            SPDLOG_DEBUG("Pushed back: size after: {}, pointer after: {}, capacity {}", device_projections_.size(),
                             reinterpret_cast<void *>(device_projections_.data()), device_projections_.capacity());

                     }, projection);
    }

    SPDLOG_DEBUG("All projections loaded.");
}


__global__ void get_projection_uids_kernel(const CUDABackendImpl::ProjectionVariants *projection,
                                           cuda::UID *pre_uid,
                                           cuda::UID *post_uid,
                                           cuda::UID *self_uid)
{
    ::cuda::std::visit([pre_uid, post_uid, self_uid](const auto &proj)
                       {
                           *pre_uid = proj.presynaptic_uid_;
                           *post_uid = proj.postsynaptic_uid_;
                           *self_uid = proj.uid_;
                       }, *projection);
}


auto get_projection_uids(const CUDABackendImpl::ProjectionVariants *proj)
{
    cuda::UID *pre_uid_gpu;
    cuda::UID *post_uid_gpu;
    cuda::UID *self_uid_gpu;
    cuda::UID pre_uid, post_uid, self_uid;
    call_and_check(cudaMalloc(&pre_uid_gpu, sizeof(cuda::UID)));
    call_and_check(cudaMalloc(&post_uid_gpu, sizeof(cuda::UID)));
    call_and_check(cudaMalloc(&self_uid_gpu, sizeof(cuda::UID)));
    get_projection_uids_kernel<<<1, 1>>>(proj, pre_uid_gpu, post_uid_gpu, self_uid_gpu);
    call_and_check(cudaMemcpy(&pre_uid, pre_uid_gpu, sizeof(cuda::UID), cudaMemcpyDeviceToHost));
    call_and_check(cudaMemcpy(&post_uid, post_uid_gpu, sizeof(cuda::UID), cudaMemcpyDeviceToHost));
    call_and_check(cudaMemcpy(&self_uid, self_uid_gpu, sizeof(cuda::UID), cudaMemcpyDeviceToHost));
    call_and_check(cudaFree(pre_uid_gpu));
    call_and_check(cudaFree(post_uid_gpu));
    call_and_check(cudaFree(self_uid_gpu));
    return std::make_tuple(pre_uid, post_uid, self_uid);
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


__device__ ::cuda::std::optional<knp::backends::gpu::cuda::SpikeMessage> CUDABackendImpl::calculate_population(
        CUDAPopulation<knp::neuron_traits::BLIFATNeuron> &population,
        const knp::backends::gpu::cuda::device_lib::CUDAVector<cuda::MessageVariant> &messages,
        unsigned long long step_n)
{
    constexpr size_t spike_message_index =
            boost::mp11::mp_find<cuda::MessageVariant, cuda::SynapticImpactMessage>();

    // TODO rework
    for (size_t i = 0; i < population.neurons_.size(); ++i)
    {
        neuron_traits::neuron_parameters <neuron_traits::BLIFATNeuron> neuron = population.neurons_[i];
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

        population.neurons_[i] = neuron;
    }

    // process_inputs(population, messages);
    PRINTF_TRACE("Processing synaptic impacts for population\n");
    for (const knp::backends::gpu::cuda::MessageVariant &message_var : messages)
    {
        if (message_var.index() != spike_message_index) continue;
        const SynapticImpactMessage &message = ::cuda::std::get<SynapticImpactMessage>(message_var);
        PRINTF_TRACE("Message size is %lu\n", message.impacts_.size());
        for (size_t i = 0; i < message.impacts_.size(); ++i)
        {
            const auto &impact = message.impacts_[i];

            neuron_traits::neuron_parameters <neuron_traits::BLIFATNeuron> neuron =
                    population.neurons_[impact.postsynaptic_neuron_index_];

            // impact_neuron<BlifatLikeNeuron>(neuron, impact.synapse_type_, impact.impact_value_);
            switch (impact.synapse_type_)
            {
                case knp::synapse_traits::OutputType::EXCITATORY:
                    neuron.potential_ += impact.impact_value_;
                    break;
                case knp::synapse_traits::OutputType::INHIBITORY_CURRENT:
                    neuron.potential_ -= impact.impact_value_;
                    break;
                case knp::synapse_traits::OutputType::INHIBITORY_CONDUCTANCE:
                    neuron.inhibitory_conductance_ += impact.impact_value_;
                    break;
                case knp::synapse_traits::OutputType::DOPAMINE:
                    neuron.dopamine_value_ += impact.impact_value_;
                    break;
                case knp::synapse_traits::OutputType::BLOCKING:
                    neuron.total_blocking_period_ = static_cast<unsigned int>(impact.impact_value_);
                    break;
            }

            /*if constexpr (has_dopamine_plasticity<BlifatLikeNeuron>())
            {
                if (impact.synapse_type_ == synapse_traits::OutputType::EXCITATORY)
                {
                    neuron.is_being_forced_ |= message.is_forcing_;
                }
            }*/
            population.neurons_[impact.postsynaptic_neuron_index_] = neuron;
        }
    }

    device_lib::CUDAVector<uint32_t> neuron_indexes;

    // calculate_neurons_post_input_state(population, neuron_indexes);
    for (size_t index = 0; index < population.neurons_.size(); ++index)
    {
        bool spike = false;
        neuron_traits::neuron_parameters <neuron_traits::BLIFATNeuron> neuron = population.neurons_[index];
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
            neuron_indexes.push_back(index);
        }

        population.neurons_[index] = neuron;
    }
    if (!neuron_indexes.empty())
    {
        PRINTF_TRACE("Spike vector is not empty: size %lu\n", neuron_indexes.size());
        cuda::SpikeMessage res_message
                {
                        .header_ = {.sender_uid_ = population.uid_, step_n},
                        .neuron_indexes_ = neuron_indexes
                };

        return res_message;
    }
    PRINTF_TRACE("Spike vector empty\n");
    return {};
}


__host__ unsigned long long CUDABackendImpl::route_projection_messages(unsigned long long step)
{
    using MessageVector = device_lib::CUDAVector<cuda::MessageVariant>;
    unsigned long long sent_message_counter = 0;
    device_message_bus_.clear();

    for (size_t i = 0; i < device_projections_.size(); ++i)
    {
        ::cuda::std::visit([this, &sent_message_counter](auto &proj)
               {
                   if (proj.message_buf_.impacts_.size())
                   {
                       device_message_bus_.send_message(proj.message_buf_);
                       proj.message_buf_.impacts_.clear();
                       ++sent_message_counter;
                   }
               }, device_projections_[i]);
    }
    SPDLOG_DEBUG("Projections sent {} messages", sent_message_counter);
    return sent_message_counter;
}


__host__ void CUDABackendImpl::calculate_projection(
        CUDAProjection<knp::synapse_traits::AdditiveSTDPDeltaSynapse> &projection,
        const knp::backends::gpu::cuda::device_lib::CUDAVector<unsigned long long> &message_ids,
        unsigned long long step_n)
{
    //SPDLOG_TRACE("Calculate AdditiveSTDPDelta synapse projection {}.", std::string(projection.get_uid()));
}


__host__ void CUDABackendImpl::calculate_projection(
        CUDAProjection<knp::synapse_traits::SynapticResourceSTDPDeltaSynapse> &projection,
        const knp::backends::gpu::cuda::device_lib::CUDAVector<unsigned long long> &message_ids,
        unsigned long long step_n)
{
}


__host__ __device__ CUDABackendImpl::PopulationIterator CUDABackendImpl::begin_populations()
{
    return PopulationIterator{device_populations_.begin()};
}


__host__ __device__ CUDABackendImpl::PopulationConstIterator CUDABackendImpl::begin_populations() const
{
    return {device_populations_.cbegin()};
}


__host__ __device__ CUDABackendImpl::PopulationIterator CUDABackendImpl::end_populations()
{
    return PopulationIterator{device_populations_.end()};
}


__host__ __device__ CUDABackendImpl::PopulationConstIterator CUDABackendImpl::end_populations() const
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


// TODO : Maybe move to delta synapse implementation:
__global__ void calculate_synaptic_impact(
        const device_lib::CUDAVectorView<CUDAProjection<knp::synapse_traits::DeltaSynapse>::Synapse> synapses,
        const unsigned long long *synapse_indices, size_t size, unsigned long long current_step,
        unsigned long long *results, unsigned long long *send_steps)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    unsigned long long synapse_id = synapse_indices[i];
    if (synapse_id >= synapses.size_) return;
    results[i] = synapse_id;
    send_steps[i] = ::cuda::std::get<0>(synapses.data_[synapse_id]).delay_ + current_step - 1;
}


__global__ void calculate_impacts_per_spike(
        const device_lib::CUDAVectorView<CUDAProjection<knp::synapse_traits::DeltaSynapse>::Synapse> synapses,
        device_lib::CUDAVectorView<SpikeIndex> spike_ids, device_lib::IndexView index,
        unsigned long long current_step, unsigned long long *results, unsigned long long *send_steps)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= spike_ids.size_) return;
    SpikeIndex neuron_id = spike_ids.data_[i];

    unsigned long long start = index.offsets_ptr_[neuron_id];
    unsigned long long size = index.offsets_ptr_[neuron_id + 1] - index.offsets_ptr_[neuron_id];
    auto [num_blocks, num_threads] = device_lib::get_blocks_config(size);
    calculate_synaptic_impact<<<num_blocks, num_threads>>>(synapses, index.indices_ptr_ + start, size, current_step,
                                                           results + start, send_steps + start);
}


__global__ void delta_indices_to_impacts_kernel(unsigned long long *indices_begin, unsigned long long *indices_end,
                cuda::device_lib::CUDAVectorView<CUDAProjection<knp::synapse_traits::DeltaSynapse>::Synapse> synapses,
                SynapticImpact *impacts_out)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long *index = indices_begin + i;
    if (index >= indices_end) return;
    unsigned long long synapse_id = *index;
    if (synapse_id > synapses.size_) return;
    auto &synapse = ::cuda::std::get<0>(synapses.data_[synapse_id]);
    SynapticImpact impact_out;
    impact_out.connection_index_ = synapse_id;
    impact_out.impact_value_ = synapse.weight_;
    impact_out.synapse_type_ = synapse.output_type_;
    impact_out.presynaptic_neuron_index_ = ::cuda::std::get<1>(synapses.data_[synapse_id]);
    impact_out.postsynaptic_neuron_index_ = ::cuda::std::get<2>(synapses.data_[synapse_id]);
    impacts_out[synapse_id] = impact_out;
}


template<>
void CUDAProjection<knp::synapse_traits::DeltaSynapse>::form_message(
        unsigned long long current_step)
{
    auto iter = thrust::upper_bound(thrust::device, sending_steps_.begin(), sending_steps_.end(), current_step);
    if (iter == sending_steps_.begin())
    {
        message_buf_.impacts_.clear();
        return;
    }
    unsigned long long num_impacts = iter - sending_steps_.begin();
    SynapticImpact *impacts;
    auto [num_blocks, num_threads] = device_lib::get_blocks_config(num_impacts);
    call_and_check(cudaMalloc(&impacts, sizeof(SynapticImpact) * num_impacts));
    delta_indices_to_impacts_kernel<<<num_blocks, num_threads>>>(impact_indexes_.data(),
                                                                impact_indexes_.data() + num_impacts, synapses_.view(),
                                                                impacts);
    MessageHeader header{uid_, current_step};
    message_buf_.header_ = header;
    message_buf_.presynaptic_population_uid_ = presynaptic_uid_;
    message_buf_.postsynaptic_population_uid_ = postsynaptic_uid_;
    message_buf_.impacts_ = device_lib::CUDAVector<SynapticImpact>{impacts, num_impacts};
    message_buf_.is_forcing_ = true;
    sending_steps_.erase(sending_steps_.begin(), sending_steps_.begin() + num_impacts);
    impact_indexes_.erase(impact_indexes_.begin(), impact_indexes_.begin() + num_impacts);
}


__global__ void get_spike_message_data(device_lib::CUDAVectorView<cuda::MessageVariant> all_messages,
                    unsigned long long msg_index, unsigned long long *size, const SpikeIndex **data_pointer)
{
    if (msg_index >= all_messages.size_ || all_messages.data_[msg_index].index())
    {
        *data_pointer = nullptr;
        *size = 0;
        return;
    }
    constexpr size_t spike_message_index = boost::mp11::mp_find<cuda::MessageVariant, cuda::SpikeMessage>();
    auto &message_var = all_messages.data_[msg_index];
    if (message_var.index() != spike_message_index)
    {
        *size = 0;
        *data_pointer = nullptr;
        return;
    }
    *data_pointer = ::cuda::std::get<cuda::SpikeMessage>(message_var).neuron_indexes_.data();
    *size = ::cuda::std::get<cuda::SpikeMessage>(message_var).neuron_indexes_.size();
}


/**
 * @brief Calculate delta projection, host gpu-using variant.
*/
__host__ void CUDABackendImpl::calculate_projection(
        CUDAProjection<knp::synapse_traits::DeltaSynapse> &projection,
        const device_lib::CUDAVector<unsigned long long> &message_ids,
        unsigned long long step_n)
{
    for (size_t i = 0; i < message_ids.size(); ++i)
    {
        unsigned long long msg_index = message_ids.copy_at(i);

        // 1. For each neuron count the corresponding synapses and allocate a buffer of the required size,
        // possibly on host.
        // Extracting message data pointer and size.
        const SpikeIndex **msg_data_pointer;
        unsigned long long *msg_data_size;
        call_and_check(cudaMalloc(&msg_data_pointer, sizeof(void *)));
        call_and_check(cudaMalloc(&msg_data_size, sizeof(unsigned long long)));

        get_spike_message_data<<<1, 1>>>(device_message_bus_.all_messages().view(), msg_index, msg_data_size,
                                         msg_data_pointer);
        unsigned long long data_size;
        const SpikeIndex *msg_data_pointer_cpu; // Non-owning GPU pointer to message data
        cudaMemcpy(&data_size, msg_data_size, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        cudaFree(msg_data_size);
        cudaMemcpy(&msg_data_pointer_cpu, msg_data_pointer, sizeof(SpikeIndex *), cudaMemcpyDeviceToHost);

        if (data_size)
        {
            unsigned long long impacts_count = count_values_by_indexes(projection.index_,
                                                                       device_lib::CUDAVectorView<SpikeIndex>{
                                                                               msg_data_pointer_cpu, data_size});

            unsigned long long *impacts_buffer;
            unsigned long long *delay_buffer;
            call_and_check(cudaMalloc(&impacts_buffer, sizeof(unsigned long long) * impacts_count));
            call_and_check(cudaMalloc(&delay_buffer, sizeof(unsigned long long) * impacts_count));

            // 2. For each active synapse calculate its impact.
            auto [num_blocks, num_threads] = device_lib::get_blocks_config(data_size);
            calculate_impacts_per_spike<<<num_blocks, num_threads>>>(projection.synapses_.view(),
                                                                     device_lib::CUDAVectorView<SpikeIndex>{
                                                                             msg_data_pointer_cpu, data_size},
                                                                     projection.index_.view(), step_n, impacts_buffer,
                                                                     delay_buffer);
            cudaDeviceSynchronize();
            // 3. Sort impacts by time
            thrust::sort_by_key(thrust::device, delay_buffer, delay_buffer + impacts_count, impacts_buffer);
            projection.add_impacts(device_lib::CUDAVector<unsigned long long>{impacts_buffer, impacts_count},
                                   device_lib::CUDAVector<unsigned long long>{delay_buffer, impacts_count});
        }
        cudaFree(msg_data_pointer);
        // Make messages
        projection.form_message(step_n);
    }
}

}   // namespace knp::backends::gpu::cuda

REGISTER_CUDA_VECTOR_TYPE(knp::backends::gpu::cuda::CUDABackendImpl::PopulationVariants);
REGISTER_CUDA_VECTOR_TYPE(knp::backends::gpu::cuda::CUDABackendImpl::ProjectionVariants);
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
