/**
 * @file populations_impl.cuh
 * @brief Contains functions for population calculation.
 * @kaspersky_support A. Vartenkov
 * @date 17.08.2026
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

#include "backend_impl.cuh"
#include "population.cuh"
#include "projection.cuh"
#include "projection_rstdp.cuh"
#include "populations_impl.cuh"
#include <knp/core/population.h>

#include <vector>


/**
 * @brief Namespace for CUDA backend.
 */
namespace knp::backends::gpu::cuda
{
    using BlifatParams = knp::neuron_traits::neuron_parameters<knp::neuron_traits::BLIFATNeuron>;

// TODO: BLIFAT implementation
//__global__ void calculate_neurons_pre_impact(device_lib::CUDAVectorMutableView<BlifatParams> neurons,
//                                             StepIndex current_step)
//{
//    const size_t neuron_index = blockIdx.x * blockDim.x + threadIdx.x;
//    if (neuron_index >= neurons.size_) return;
//
//    BlifatParams &neuron = neurons.data_[neuron_index];
//    ++neuron.n_time_steps_since_last_firing_;
//    neuron.dynamic_threshold_ *= neuron.threshold_decay_;
//    neuron.postsynaptic_trace_ *= neuron.postsynaptic_trace_decay_;
//    neuron.inhibitory_conductance_ *= neuron.inhibitory_conductance_decay_;
//
//    /*
//    if constexpr (has_dopamine_plasticity<BlifatLikeNeuron>())
//    {
//        neuron.dopamine_value_ = 0.0;
//        neuron.is_being_forced_ = false;
//    }
//    */
//
//    if (neuron.bursting_phase_ && !--neuron.bursting_phase_)
//    {
//        neuron.potential_ = neuron.potential_ * neuron.potential_decay_ + neuron.reflexive_weight_;
//    }
//    else
//    {
//        neuron.potential_ *= neuron.potential_decay_;
//    }
//    neuron.pre_impact_potential_ = neuron.potential_;
//}
//
//
//__global__ void calculate_neurons_impacts(device_lib::CUDAVectorMutableView<BlifatParams> neurons,
//                                          device_lib::CUDAVectorView<SynapticImpact> impacts)
//{
//    const size_t impact_index = blockIdx.x * blockDim.x + threadIdx.x;
//    if (impact_index >= impacts.size_) return;
//    const SynapticImpact &impact = impacts.data_[impact_index];
//    if (impact.postsynaptic_neuron_index_ >= neurons.size_) return;
//    auto &neuron = neurons.data_[impact.postsynaptic_neuron_index_];
//    switch (impact.synapse_type_)
//    {
//        case knp::synapse_traits::OutputType::EXCITATORY:
//            atomicAdd(&neuron.potential_, impact.impact_value_);
//            break;
//        case knp::synapse_traits::OutputType::INHIBITORY_CURRENT:
//            atomicAdd(&neuron.potential_, -impact.impact_value_);
//            break;
//        case knp::synapse_traits::OutputType::INHIBITORY_CONDUCTANCE:
//            atomicAdd(&neuron.inhibitory_conductance_, impact.impact_value_);
//            break;
//        case knp::synapse_traits::OutputType::DOPAMINE:
//            atomicAdd(&neuron.dopamine_value_, impact.impact_value_);
//            break;
//        case knp::synapse_traits::OutputType::BLOCKING:
//            neuron.total_blocking_period_ = static_cast<unsigned int>(impact.impact_value_);
//            break;
//    }
//}
//
//

//
//
//__global__ void calculate_neurons_post_impact(device_lib::CUDAVectorMutableView<BlifatParams> neurons,
//                                              SpikeIndex *spike_buffer, SpikeIndex *size_counter)
//{
//    const size_t neuron_index = blockIdx.x * blockDim.x + threadIdx.x;
//    bool spike = false;
//    neuron_traits::neuron_parameters <neuron_traits::BLIFATNeuron> &neuron = neurons.data_[neuron_index];
//    if (neuron.total_blocking_period_ <= 0)
//    {
//        // TODO: Make it more readable, don't be afraid to use if operators.
//        // Restore potential that the neuron had before impacts.
//        neuron.potential_ = neuron.pre_impact_potential_;
//        bool was_negative = neuron.total_blocking_period_ < 0;
//        // If it is negative, increase by 1.
//        neuron.total_blocking_period_ += was_negative;
//        // If it is now zero, but was negative before, increase it to max, else leave it as is.
//        neuron.total_blocking_period_ +=
//                std::numeric_limits<int64_t>::max() * ((neuron.total_blocking_period_ == 0) && was_negative);
//    }
//    else
//    {
//        neuron.total_blocking_period_ -= 1;
//    }
//
//    if (neuron.inhibitory_conductance_ < 1.0)
//    {
//        neuron.potential_ -=
//                (neuron.potential_ - neuron.reversal_inhibitory_potential_) *
//                neuron.inhibitory_conductance_;
//    }
//    else
//    {
//        neuron.potential_ = neuron.reversal_inhibitory_potential_;
//    }
//
//    if ((neuron.n_time_steps_since_last_firing_ > neuron.absolute_refractory_period_) &&
//        (neuron.potential_ >= neuron.activation_threshold_ + neuron.dynamic_threshold_))
//    {
//        // Spike.
//        neuron.dynamic_threshold_ += neuron.threshold_increment_;
//        neuron.postsynaptic_trace_ += neuron.postsynaptic_trace_increment_;
//
//        neuron.potential_ = neuron.potential_reset_value_;
//        neuron.bursting_phase_ = neuron.bursting_period_;
//        neuron.n_time_steps_since_last_firing_ = 0;
//        spike = true;
//    }
//
//    if (neuron.potential_ < neuron.min_potential_)
//    {
//        neuron.potential_ = neuron.min_potential_;
//    }
//    if (spike)
//    {
//        SpikeIndex counter = atomicAdd(size_counter, 1);
//        spike_buffer[counter] = neuron_index;
//    }
//}


//    device_lib::CUDAVector<SpikeIndex> calculate_population(
//            CUDAPopulation<knp::neuron_traits::BLIFATNeuron> &population, const CUDAMessageBus& device_message_bus,
//            StepIndex step)
//    {
//        auto [num_blocks_neuro, num_threads_neuro] = device_lib::get_blocks_config(population.neurons_.size());
//
//        calculate_neurons_pre_impact<<<num_blocks_neuro, num_threads_neuro>>>(population.neurons_.mut_view(), step);
//        std::vector<device_lib::LongIndex> message_ids
//                = device_message_bus.unload_messages<cuda::SynapticImpactMessage>(population.uid_);
//
//        if (!message_ids.empty())
//        {
//            SPDLOG_DEBUG("Running calculate impacts on {} messages", message_ids.size());
//            calculate_neurons_impacts_all(population.neurons_.mut_view(),
//                                          device_message_bus.all_messages<SynapticImpactMessage>(),
//                                          message_ids);
//        }
//        SpikeIndex *output;
//        cudaMalloc(&output, sizeof(SpikeIndex) * population.neurons_.size());
//        SpikeIndex *counter;
//        cudaMalloc(&counter, sizeof(SpikeIndex));
//        cudaMemset(counter, 0, sizeof(SpikeIndex));
//        calculate_neurons_post_impact<<<num_blocks_neuro, num_threads_neuro>>>(population.neurons_.mut_view(), output,
//                                                                               counter);
//
//        SpikeIndex out_size = 0;
//        cudaMemcpy(&out_size, counter, sizeof(SpikeIndex), cudaMemcpyDeviceToHost);
//        // Capacity would be "out_size" while the ptr is larger, but that doesn't matter as the pointer is freed as a whole.
//        device_lib::CUDAVector<SpikeIndex> result{output, out_size};
//        cudaFree(counter);
//        return result;
//    }

    template<class BlifatLikeNeuron, class BaseSynapseType, class ProjectionContainer>
    std::optional <core::messaging::SpikeMessage> calculate_resource_stdp_population(
            knp::core::Population<neuron_traits::SynapticResourceSTDPNeuron < BlifatLikeNeuron>>

    &pop,
    ProjectionContainer &container, knp::core::MessageEndpoint
    &endpoint,
    size_t step_n
    )
{
    std::vector <knp::core::messaging::SynapticImpactMessage> messages =
            endpoint.unload_messages<knp::core::messaging::SynapticImpactMessage>(pop.get_uid());
    knp::core::messaging::SpikeMessage message_out{{pop.get_uid(), step_n},
                                                   {}};
    populations::calculate_pre_impact_population_state(pop,
    0, pop.

    size()

    );
    populations::impact_population(pop, messages
    );
    populations::calculate_post_impact_population_state(pop, message_out,
    0, pop.

    size()

    );

    auto working_projections = find_projection_by_type_and_postsynaptic<
            knp::synapse_traits::SynapticResourceSTDPDeltaSynapse, ProjectionContainer>(container, pop.get_uid(), true);
    cpu::populations::train_population(pop, working_projections, message_out, step_n
    );

    if (!message_out.neuron_indexes_.

    empty()

    )
{
    endpoint.
    send_message(message_out);
}

return
message_out;
}

// TODO: device
// Pre-impact
inline void calculate_pre_impact_single_neuron_state_impl(
        knp::neuron_traits::neuron_parameters<knp::neuron_traits::SynapticResourceSTDPBLIFATNeuron> &neuron)
{
    ++neuron.n_time_steps_since_last_firing_;
    neuron.dynamic_threshold_ *= neuron.threshold_decay_;
    neuron.postsynaptic_trace_ *= neuron.postsynaptic_trace_decay_;
    neuron.inhibitory_conductance_ *= neuron.inhibitory_conductance_decay_;

    neuron.dopamine_value_ = 0.0;
    neuron.is_being_forced_ = false;

    neuron.potential_ *= neuron.potential_decay_;
    if (1 == neuron.bursting_phase_) neuron.potential_ += neuron.reflexive_weight_;

    neuron.pre_impact_potential_ = neuron.potential_;
}


template<class Neuron>
void calculate_pre_impact_population_state(knp::core::Population<Neuron> &population, size_t start, size_t end)
{
    SPDLOG_TRACE("Calculate pre impact state of [{},{}] neurons.", start, end);
    for (size_t i = start; i < end; ++i)
    {
        impl::calculate_pre_impact_single_neuron_state_dispatch(population[i]);
    }
}

// Impact
inline void impact_neuron_impl(
        knp::neuron_traits::neuron_parameters<knp::neuron_traits::BLIFATNeuron> &neuron,
        const knp::core::messaging::SynapticImpact &impact, bool is_forcing)
{
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
            neuron.total_blocking_period_ = static_cast<decltype(neuron.total_blocking_period_) >
                                                        (impact.impact_value_);
            break;
        default:
            SPDLOG_ERROR("Unhandled synapse type.");
            throw std::runtime_error("Unhandled synapse type.");
    }
}


inline void impact_neuron_impl(
        knp::neuron_traits::neuron_parameters<knp::neuron_traits::SynapticResourceSTDPBLIFATNeuron> &neuron,
        const knp::core::messaging::SynapticImpact &impact, bool is_forcing)
{
    impact_neuron_impl(
            static_cast<knp::neuron_traits::neuron_parameters<knp::neuron_traits::BLIFATNeuron> &>(neuron), impact,
            is_forcing);
    if (synapse_traits::OutputType::EXCITATORY == impact.synapse_type_)
    {
        neuron.is_being_forced_ |= is_forcing;
    }
}

template<typename Synapse>
inline void process_spiking_neurons_impl(
        const core::messaging::SpikeMessage &msg,
        std::vector <std::reference_wrapper<knp::core::Projection<Synapse>>> &working_projections,
        knp::core::Population<knp::neuron_traits::SynapticResourceSTDPBLIFATNeuron> &population, uint64_t step)
{
    // It's very important that during this function no projection invalidates iterators.
    // Loop over neurons.
    for (const auto &spiked_neuron_index: msg.neuron_indexes_)
    {
        auto synapse_params =
                training::stdp::get_all_connected_synapses<Synapse>(working_projections, spiked_neuron_index);
        auto &neuron = population[spiked_neuron_index];
        neuron.last_spike_step_ = step;
        // Calculate neuron ISI status.
        training::stdp::update_isi<knp::neuron_traits::BLIFATNeuron>(neuron, step);
        if (neuron_traits::ISIPeriodType::period_started == neuron.isi_status_)
            neuron.stability_ -= neuron.stability_change_at_isi_;
        neuron.additional_threshold_ = 0.0;
        // Mark contributed synapses
        for (auto &synapse: synapse_params)
        {
            neuron.additional_threshold_ += synapse.get().weight_ * (synapse.get().weight_ > 0);
            const bool had_spike = training::stdp::is_point_in_interval(
                    step - synapse.get().rule_.dopamine_plasticity_period_, step,
                    synapse.get().rule_.last_spike_step_ + synapse.get().delay_ - 1);
            // While period continues we don't change has_contributed from true to false.
            if (neuron_traits::ISIPeriodType::period_continued != neuron.isi_status_ || had_spike)
            {
                synapse.get().rule_.has_contributed_ = had_spike;
            }
        }
        neuron.additional_threshold_ *= neuron.synapse_sum_threshold_coefficient_;

        // This is a new spiking sequence, we can update synapses now.
        if (neuron.isi_status_ != neuron_traits::ISIPeriodType::period_continued)
        {
            for (auto &synapse: synapse_params)
            {
                synapse.get().rule_.had_hebbian_update_ = false;
            }
        }

        // Update synapse-only data.
        if (neuron.isi_status_ != neuron_traits::ISIPeriodType::is_forced)
        {
            for (auto &synapse: synapse_params)
            {
                // Unconditional decreasing synaptic resource.
                // TODO: NOT HERE. This shouldn't matter now as d_u_ is zero for our task, but the logic is wrong.
                synapse.get().rule_.synaptic_resource_ -= synapse.get().rule_.d_u_;
                neuron.free_synaptic_resource_ += synapse.get().rule_.d_u_;
                // Hebbian plasticity.
                // 1. Check if synapse ever got a spike in the current ISI period.
                if (synapse.get().rule_.has_contributed_ && !synapse.get().rule_.had_hebbian_update_)
                {
                    // 2. If it did, then update synaptic resource value.
                    const float d_h = neuron.d_h_ * std::min(static_cast<float>(std::pow(2, -neuron.stability_)), 1.F);

                    synapse.get().rule_.synaptic_resource_ += d_h;
                    neuron.free_synaptic_resource_ -= d_h;
                    synapse.get().rule_.had_hebbian_update_ = true;
                }
            }
        }
        // Recalculating synapse weights. Sometimes it probably doesn't need to happen, check it later.
        training::stdp::recalculate_synapse_weights<knp::synapse_traits::DeltaSynapse>(synapse_params);
    }
}


template<typename Synapse>
inline void do_dopamine_plasticity_impl(
        std::vector <std::reference_wrapper<knp::core::Projection<Synapse>>> &working_projections,
        knp::core::Population<knp::neuron_traits::SynapticResourceSTDPBLIFATNeuron> &population, uint64_t step)
{
    using SynapseType = knp::synapse_traits::SynapticResourceSTDPDeltaSynapse;
    using SynapseParamType = knp::synapse_traits::synapse_parameters<SynapseType>;
    for (size_t neuron_index = 0; neuron_index < population.size(); ++neuron_index)
    {
        auto &neuron = population[neuron_index];
        // Dopamine processing. Dopamine punishment if forced does nothing.
        if (neuron.dopamine_value_ > 0.0 ||
            (neuron.dopamine_value_ < 0.0 && neuron.isi_status_ != neuron_traits::ISIPeriodType::is_forced))
        {
            std::vector <std::reference_wrapper<SynapseParamType>> synapse_params =
                    training::stdp::get_all_connected_synapses<SynapseType>(working_projections, neuron_index);
            // Change synapse values for both `D > 0` and `D < 0`.
            for (auto &synapse: synapse_params)
            {
                // if ((step - synapse.get().rule_.last_spike_step_ < synapse.get().rule_.dopamine_plasticity_period_)
                if (step - neuron.last_spike_step_ <= neuron.dopamine_plasticity_time_ &&
                    synapse.get().rule_.has_contributed_)
                {
                    // Change synapse resource.
                    float resource_change =
                            neuron.dopamine_value_ * std::min(static_cast<float>(std::pow(2, -neuron.stability_)), 1.F);

                    synapse.get().rule_.synaptic_resource_ += resource_change;
                    neuron.free_synaptic_resource_ -= resource_change;
                }
            }
            // Stability changes.
            if (neuron.is_being_forced_ || neuron.dopamine_value_ < 0)
            {
                // A dopamine reward when forced or a dopamine punishment reduce stability by `r * D`.
                neuron.stability_ -= neuron.dopamine_value_ * neuron.stability_change_parameter_;
                neuron.stability_ = std::max(neuron.stability_, 0.0F);
            }
            else
            {
                // A dopamine reward when non-forced changes stability by `D max(2 - |t(TSS) - ISImax| / ISImax, -1)`.
                const double dopamine_constant = 2.0;
                const double difference = std::fabs(step - neuron.first_isi_spike_ - neuron.isi_max_);
                neuron.stability_ += neuron.stability_change_parameter_ * neuron.dopamine_value_ *
                                     std::max(dopamine_constant - difference / neuron.isi_max_, -1.0);
            }
            training::stdp::recalculate_synapse_weights(synapse_params);
        }
    }
}


inline void train_population_impl(
        knp::core::Population<knp::neuron_traits::SynapticResourceSTDPBLIFATNeuron> &population,
        std::vector <std::reference_wrapper<knp::core::Projection<knp::synapse_traits::SynapticResourceSTDPDeltaSynapse>>>
        &projections,
        const knp::core::messaging::SpikeMessage &message, knp::core::Step step)
{
    if (message.neuron_indexes_.size())
    {
        process_spiking_neurons_impl(message, projections, population, step);
    }

    do_dopamine_plasticity_impl(projections, population, step);

    training::stdp::renormalize_resource(projections, population, step);
}


template<class Neuron, class Synapse>
inline void renormalize_resource(
        std::vector <std::reference_wrapper<knp::core::Projection<Synapse>>> &working_projections,
        knp::core::Population<Neuron> &population, uint64_t step)
{
    for (size_t neuron_index = 0; neuron_index < population.size(); ++neuron_index)
    {
        auto &neuron = population[neuron_index];
        if (step - neuron.last_step_ <= neuron.isi_max_ &&
            neuron.isi_status_ != neuron_traits::ISIPeriodType::is_forced)
        {
            // Neuron is still in ISI period, skip it.
            continue;
        }

        if (std::fabs(neuron.free_synaptic_resource_) < neuron.synaptic_resource_threshold_)
        {
            continue;
        }

        auto synapse_params = get_all_connected_synapses<Synapse>(working_projections, neuron_index);

        // Divide free resource between all synapses.
        auto add_resource_value =
                neuron.free_synaptic_resource_ / (synapse_params.size() + neuron.resource_drain_coefficient_);

        for (auto &synapse: synapse_params)
        {
            synapse.get().rule_.synaptic_resource_ += add_resource_value;
        }

        neuron.free_synaptic_resource_ = 0.0F;
        recalculate_synapse_weights(synapse_params);
    }
}

template<class Synapse>
void recalculate_synapse_weights(
        std::vector <std::reference_wrapper<knp::synapse_traits::synapse_parameters<
                knp::synapse_traits::STDP<knp::synapse_traits::STDPSynapticResourceRule, Synapse>>>> &synapse_params)
{
    // Synapse weight recalculation.
    for (auto &synapse: synapse_params)
    {
        const auto &rule = synapse.get().rule_;
        const auto syn_w = std::max(rule.synaptic_resource_, 0.F);
        const auto weight_diff = rule.w_max_ - rule.w_min_;
        synapse.get().weight_ = rule.w_min_ + weight_diff * syn_w / (weight_diff + syn_w);
    }
}


// So let's see:
// 1. Устанавливаем все нейроны в начальную позицию. Дальше, предположительно, ожидание? Цикл по нейронам.
// 2. Раскидываем воздействия по нейронам, ожидание. Цикл по воздействиям.
// 3. финализируем нейроны.
// 3.1 Получаем синапсы по нейронам, апдейтим синапсы. Ожидание не нужно?
using ResourceBlifatParams = knp::neuron_traits::neuron_parameters<
        knp::neuron_traits::SynapticResourceSTDPBLIFATNeuron>;

using ResourceSynapseParams = knp::synapse_traits::synapse_parameters<knp::synapse_traits::SynapticResource


__global__ void calculate_neurons_pre_impact(device_lib::CUDAVectorMutableView <ResourceBlifatParams> neurons,
                                             StepIndex current_step)
{
    const size_t neuron_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron_index >= neurons.size_) return;

    BlifatParams &neuron = neurons.data_[neuron_index];
    ++neuron.n_time_steps_since_last_firing_;
    neuron.dynamic_threshold_ *= neuron.threshold_decay_;
    neuron.postsynaptic_trace_ *= neuron.postsynaptic_trace_decay_;
    neuron.inhibitory_conductance_ *= neuron.inhibitory_conductance_decay_;

    if constexpr(has_dopamine_plasticity<BlifatLikeNeuron>())
    {
        neuron.dopamine_value_ = 0.0;
        neuron.is_being_forced_ = false;
    }

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


template <class Synapse>
device_lib::CUDAVector<synapse_traits::synapse_parameters <Synapse> *>
get_all_connected_synapses(CUDABackendImpl::ProjectionContainer &device_projections,
                           const CUDAVector<LongIndex> &projection_indices, size_t neuron_index)
{
    device_lib::CUDAVector<synapse_traits::synapse_parameters < Synapse> *> result;
    for (auto &projection: projections)
    {
        // we need synapses, for each neuron and projection there's a VectorView with synapse ids.
        auto synapses =
                projection.get().find_synapses(neuron_index, core::Projection<Synapse>::Search::by_postsynaptic);
        std::transform(synapses.begin(), synapses.end(), std::back_inserter(result),
        [&projection](auto const &index)
        {
            return std::reference_wrapper(std::get<core::synapse_data>(projection.get()[index]));
        });
    }
    return result;
}


__global__ void calculate_neurons_impacts(device_lib::CUDAVectorMutableView <ResourceBlifatParams> neurons,
                                          device_lib::CUDAVectorView <SynapticImpact> impacts, bool is_forcing)
{
    const size_t impact_index = blockIdx.x * blockDim.x + threadIdx.x;
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
    neuron.is_being_forced_ |= is_forcing;
}


// TODO: template this, it's a complete copy from populations_impl.cu
__host__ void calculate_neurons_impacts_all(device_lib::CUDAVectorMutableView <ResourceBlifatParams> neurons,
                                            const std::vector <SynapticImpactMessage> &messages_all,
                                            std::vector <device_lib::LongIndex> message_ids)
{
    for (const auto &msg_id: message_ids)
    {
        const auto &msg = messages_all[msg_id];
        auto [num_blocks, num_threads] = device_lib::get_blocks_config(msg.impacts_.size());
        calculate_neurons_impacts<<<num_blocks, num_threads>>>(neurons, msg.impacts_.view());
    }
    cudaDeviceSynchronize();
}


__global__ void calculate_neurons_post_impact(device_lib::CUDAVectorMutableView <ResourceBlifatParams> neurons,
                                              SpikeIndex *spike_buffer, SpikeIndex *size_counter)
{
    const size_t neuron_index = blockIdx.x * blockDim.x + threadIdx.x;
    bool spike = false;
    neuron_traits::neuron_parameters <neuron_traits::BLIFATNeuron> &neuron = neurons.data_[neuron_index];
    if (neuron.total_blocking_period_ <= 0)
    {
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


//inline void do_dopamine_plasticity_impl(
//        ProjectionContainer &device_projections,
//        device_lib::CUDAVector<LongIndex> &working_projections_indices,
//        CUDAPopulation<knp::neuron_traits::SynapticResourceSTDPBLIFATNeuron> &population, StepIndex step)
//{
//    using SynapseType = knp::synapse_traits::SynapticResourceSTDPDeltaSynapse;
//    using SynapseParamType = knp::synapse_traits::synapse_parameters<SynapseType>;
//    for (size_t neuron_index = 0; neuron_index < population.size(); ++neuron_index)
//    {
//        auto &neuron = population[neuron_index];
//        // Dopamine processing. Dopamine punishment if forced does nothing.
//        if (neuron.dopamine_value_ > 0.0 ||
//            (neuron.dopamine_value_ < 0.0 && neuron.isi_status_ != neuron_traits::ISIPeriodType::is_forced))
//        {
//            std::vector<std::reference_wrapper<SynapseParamType>> synapse_params =
//                    training::stdp::get_all_connected_synapses<SynapseType>(working_projections, neuron_index);
//            // Change synapse values for both `D > 0` and `D < 0`.
//            for (auto &synapse : synapse_params)
//            {
//                // if ((step - synapse.get().rule_.last_spike_step_ < synapse.get().rule_.dopamine_plasticity_period_)
//                if (step - neuron.last_spike_step_ <= neuron.dopamine_plasticity_time_ &&
//                    synapse.get().rule_.has_contributed_)
//                {
//                    // Change synapse resource.
//                    float resource_change =
//                            neuron.dopamine_value_ * std::min(static_cast<float>(std::pow(2, -neuron.stability_)), 1.F);
//
//                    synapse.get().rule_.synaptic_resource_ += resource_change;
//                    neuron.free_synaptic_resource_ -= resource_change;
//                }
//            }
//            // Stability changes.
//            if (neuron.is_being_forced_ || neuron.dopamine_value_ < 0)
//            {
//                // A dopamine reward when forced or a dopamine punishment reduce stability by `r * D`.
//                neuron.stability_ -= neuron.dopamine_value_ * neuron.stability_change_parameter_;
//                neuron.stability_ = std::max(neuron.stability_, 0.0F);
//            }
//            else
//            {
//                // A dopamine reward when non-forced changes stability by `D max(2 - |t(TSS) - ISImax| / ISImax, -1)`.
//                const double dopamine_constant = 2.0;
//                const double difference = std::fabs(step - neuron.first_isi_spike_ - neuron.isi_max_);
//                neuron.stability_ += neuron.stability_change_parameter_ * neuron.dopamine_value_ *
//                                     std::max(dopamine_constant - difference / neuron.isi_max_, -1.0);
//            }
//            training::stdp::recalculate_synapse_weights(synapse_params);
//        }
//    }
//}

template<class Synapse>
struct SynapsesPerNeurons
{
    LongIndex offsets_size_;
    LongIndex *offsets_;

    knp::synapse_traits::synapse_parameters<Synapse> *synapses_;
    LongIndex synapses_size_;
};


template<class Synapse>
__global__ index_to_pointer(device_lib::IndexView
synapse_index,
knp::synapse_traits::synapse_parameters<Synapse> *start,
        knp::synapse_traits::synapse_parameters<Synapse>
**output)
{
const device_lib::LongIndex synapse_index = blockIdx.x * blockDim.x + threadIdx.x;
if (synapse_index >= synapse_index.indices_size_) return;
output[neuron_index] = start +
synapse_index;
}


template<class Synapse>
__host__ SynapsesPerNeurons initialize_synapses_per_neurons(const device_lib::IndexView &synapse_index,
                                                            const knp::synapse_traits::synapse_parameters<Synapse> *start)
{
    SynapsesPerNeurons<Synapse> result;
    call_and_check(cudaMalloc(&result.offsets_, sizeof(LongIndex) * synapse_index.offsets_size_));
    call_and_check(cudaMalloc(&result.synapses_, sizeof(LongIndex) * synapse_index.offsets_size_));
    // TODO static_assert(is_same_type(SynapsesPerNeurons::synapses_, ValueIndexView::offsets_))
    cudaMemcpy(result.offsets_, synapse_index.offsets_ptr_, sizeof(LongIndex) * synapse_index.offsets_size_,
               cudaMemcpyDeviceToDevice);
    result.offsets_size_ = synapse_index.offsets_size_;
    result.synapses_size_ = synapse_index.indices_size_;
    cudaMalloc(result.synapses_, sizeof(void *) * synapse_index.indices_size_);
    auto [num_blocks, num_threads] = get_blocks_config(synapse_index.indices_size_);
    index_to_pointer<<<num_blocks, num_threads>>>(synapse_index, start, &result.synapses_);
}


// TODO: Merge synapses pointers
__host__ SynapsesPerNeurons merge_synapses_per_neurons(const knp::SynapsesPerNeurons **synapses_array)
{
    // result.offsets[i] = sum(array.offsets[i])
    // result.synapses[offsets[i]] = concat(array.synapses[array[i].offsets[i] : array[i].offsets[i + 1]])
}


// Ядро для проверки
// Как должно работать:
// Вначале мы находим для всех нейронов связанные с ними синапсы. Это входной параметр, который SynapsesPerNeurons
__global__ void do_dopamine_plasticity_kernel(SynapsesPerNeurons synapse_pointer_index,
                                              device_lib::CUDAVectorMutableView <ResourceBlifatParams> neurons)
{
    using SynapseType = knp::synapse_traits::SynapticResourceSTDPDeltaSynapse;
    using SynapseParamType = knp::synapse_traits::synapse_parameters<SynapseType>;
    // Find the current neuron.
    const device_lib::LongIndex neuron_id = blockIdx.x * blockDim.x + threadIdx.x;
    // Check that it's a correct kernel.
    if (neuron_id >= neurons.size_) return;
    // Check that it's not an unconnected "extra" neuron.
    if (synapse_pointer_index.offsets_size_ == 0 || neuron_id >= synapse_pointer_index.offsets_size_ - 1) return;
    // These are the beginning index of synapse pointer array section and the size of that section.
    device_lib::LongIndex offset = synapse_pointer_index.offsets_[neuron_id];
    device_lib::LongIndex size = synapse_pointer_index.offsets_[neuron_id + 1] - offset;
    // Now we make a VectorMutableView on synapses and a neuron index. It's a device function

}


template<Synapse>
__device__ void do_dopamine_plasticity_device(
        device_lib::CUDAVectorMutableView <knp::synapse_traits::synapse_parameters<Synapse>> synapses,
        ResourceBlifatParams &neuron)
{

    do_dopamine_plasticity_synapse_kernel(...);

    if (neuron.is_being_forced_ || neuron.dopamine_value_ < 0)
    {
        // A dopamine reward when forced or a dopamine punishment reduce stability by `r * D`.
        neuron.stability_ -= neuron.dopamine_value_ * neuron.stability_change_parameter_;
        neuron.stability_ = std::max(neuron.stability_, 0.0F);
    }
    else
    {
        // A dopamine reward when non-forced changes stability by `D max(2 - |t(TSS) - ISImax| / ISImax, -1)`.
        const double dopamine_constant = 2.0;
        const double difference = std::fabs(step - neuron.first_isi_spike_ - neuron.isi_max_);
        neuron.stability_ += neuron.stability_change_parameter_ * neuron.dopamine_value_ *
                             std::max(dopamine_constant - difference / neuron.isi_max_, -1.0);
    }

}


__device__ void recalculate_synapse_weight(device_lib::CUDAVectorMutableView<> synapse_params)
{
    const device_lib::LongIndex synapse_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (synapse_id >= synapse_params.size_) return;
    auto
    const auto &rule = synapse.get().rule_;
    const auto syn_w = std::max(rule.synaptic_resource_, 0.F);
    const auto weight_diff = rule.w_max_ - rule.w_min_;
    synapse.get().weight_ = rule.w_min_ + weight_diff * syn_w / (weight_diff + syn_w);
}


template <Synapse>
__global__ void do_dopamine_plasticity_synapse_kernel(
        device_lib::CUDAVectorMutableView<knp::synapse_traits::synapse_parameters<Synapse>> synapses,
        ResourceBlifatParams *neuron)
{
    const device_lib::LongIndex synapse_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (synapse_id >= synapses.size_) return;

    auto &synapse = synapses.data_[synapse_id];
    if (step - neuron->last_spike_step_ <= neuron.dopamine_plasticity_time_ &&
        synapse.get().rule_.has_contributed_)
    {
        // Change synapse resource.
        float resource_change =
                neuron.dopamine_value_ * std::min(static_cast<float>(std::pow(2, -neuron.stability_)), 1.F);

        synapse.get().rule_.synaptic_resource_ += resource_change;
        atomicAdd(&neuron->free_synaptic_resource_, -resource_change);
    }
    recalculate_synapse_weights(synapse);
}


// для каждой проекции находим по постсинаптическому индексу связанные с нужным нам нейроном синапсы. Кернелом
// вызываем кернелы в цикле, синхронизация потом.
// заранее выделить память на нужное число View, которое равно числу рабочих проекций.
__global__ void do_dopamine_plasticity_impl(
        ProjectionContainer &device_projections,
        device_lib::CUDAVectorView<LongIndex> working_projections_indices,
        device_lib::CUDAVectorMutableView<ResourceBlifatParams> neurons, StepIndex step)
{

    using SynapseType = knp::synapse_traits::SynapticResourceSTDPDeltaSynapse;
    using SynapseParamType = knp::synapse_traits::synapse_parameters<SynapseType>;
    const size_t neuron_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron_index >= neurons.size_) return;

    auto &neuron = population[neuron_index];
    // Dopamine punishment if forced does nothing.
    if (neuron.dopamine_value_ <= 0.0 && neuron.isi_status_ == neuron_traits::ISIPeriodType::is_forced)) return;

    // Dopamine processing.
    std::vector<std::reference_wrapper<SynapseParamType>> synapse_params =
            training::stdp::get_all_connected_synapses<SynapseType>(working_projections, neuron_index);
    // Change synapse values for both `D > 0` and `D < 0`.
    for (auto &synapse : synapse_params)
    {
        // if ((step - synapse.get().rule_.last_spike_step_ < synapse.get().rule_.dopamine_plasticity_period_)
        if (step - neuron.last_spike_step_ <= neuron.dopamine_plasticity_time_ &&
            synapse.get().rule_.has_contributed_)
        {
            // Change synapse resource.
            float resource_change =
                    neuron.dopamine_value_ * std::min(static_cast<float>(std::pow(2, -neuron.stability_)), 1.F);

            synapse.get().rule_.synaptic_resource_ += resource_change;
            neuron.free_synaptic_resource_ -= resource_change;
        }
    }
    // Stability changes.
    if (neuron.is_being_forced_ || neuron.dopamine_value_ < 0)
    {
        // A dopamine reward when forced or a dopamine punishment reduce stability by `r * D`.
        neuron.stability_ -= neuron.dopamine_value_ * neuron.stability_change_parameter_;
        neuron.stability_ = std::max(neuron.stability_, 0.0F);
    }
    else
    {
        // A dopamine reward when non-forced changes stability by `D max(2 - |t(TSS) - ISImax| / ISImax, -1)`.
        const double dopamine_constant = 2.0;
        const double difference = std::fabs(step - neuron.first_isi_spike_ - neuron.isi_max_);
        neuron.stability_ += neuron.stability_change_parameter_ * neuron.dopamine_value_ *
                             std::max(dopamine_constant - difference / neuron.isi_max_, -1.0);
    }
    training::stdp::recalculate_synapse_weights(synapse_params);
}


device_lib::CUDAVector<SpikeIndex> calculate_population(
        CUDAPopulation<knp::neuron_traits::SynapticResourceSTDPNeuron> &population, CUDABackendImpl *this_backend,
        StepIndex step)
{
    auto [num_blocks_neuro, num_threads_neuro] = device_lib::get_blocks_config(population.neurons_.size());

    calculate_neurons_pre_impact<<<num_blocks_neuro, num_threads_neuro>>>(population.neurons_.mut_view(), step);
    auto &device_message_bus = this_backend->get_message_bus();
    std::vector<device_lib::LongIndex> message_ids
            = device_message_bus.unload_messages<cuda::SynapticImpactMessage>(population.uid_);

    if (!message_ids.empty())
    {
        SPDLOG_DEBUG("Running calculate impacts on {} messages", message_ids.size());
        calculate_neurons_impacts_all(population.neurons_.mut_view(),
                                      device_message_bus.all_messages<SynapticImpactMessage>(),
                                      message_ids);
    }
    SpikeIndex *output;
    cudaMalloc(&output, sizeof(SpikeIndex) * population.neurons_.size());
    SpikeIndex *counter;
    cudaMalloc(&counter, sizeof(SpikeIndex));
    cudaMemset(counter, 0, sizeof(SpikeIndex));
    calculate_neurons_post_impact<<<num_blocks_neuro, num_threads_neuro>>>(population.neurons_.mut_view(), output,
                                                                           counter);

    std::vector<device_lib::LongIndex> working_projection_indices = this_backend->find_projections_by_postsynaptic<
            synapse_traits::SynapticResourceSTDPDeltaSynapse>(population.get_uid());
    using ResourceProjection = CUDAProjection<synapse_traits::SynapticResourceSTDPDeltaSynapse>;
    // We have a number of projections, let's take them and for each we have a VectorView with synapses per neuron.
    // So it's a VectorView<VectorView<LongIndex>>? No. We need a CUDAVector of synapse pointers. Per neuron? No.
    // We start a kernel for each working projection + population, that gives us a set(?) of vectors views per
    // projection. Number of vecviews is equal to the number of neurons, size of set is the number of working
    // projections. So. We create a vector of views... This is a really complex structure, actually!
    // Let's try to simplify at least somehow. What we need is a VectorView of synapse pointers per neuron. Or not,
    // maybe more like a multivector?
    // Let's start with the result. The result is: a long list of synapse pointer and a shorter list of offsets per
    // neuron. We can make one of those per projection, easily. 

    do_dopamine_plasticity_impl(projections, population, step);

    training::stdp::renormalize_resource(projections, population, step);
}


} // namespace knp::backends::gpu::cuda