/**
 * @file synapse_slicing_generators.h
 * @brief Parameters generators that can be used for projection upcasting and simplification.
 * @kaspersky_support A. Vartenkov
 * @date 17.02.2026
 * @license Apache 2.0
 * @copyright © 2026 AO Kaspersky Lab
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
#pragma once
#include <knp/core/projection.h>
#include <knp/synapse-traits/all_traits.h>


namespace knp::framework::projection::synapse_generators
{
template <class BaseSynapse, class DerivedSynapse>
constexpr synapse_traits::synapse_parameters<BaseSynapse> slice_synapse(
        const synapse_traits::synapse_parameters<DerivedSynapse> &input_synapse_params)
{
    static_assert(
            std::is_base_of_v<
                    synapse_traits::synapse_parameters<BaseSynapse>,
                    synapse_traits::synapse_parameters<DerivedSynapse>
                    >);
    return static_cast<synapse_traits::synapse_parameters<BaseSynapse>>(input_synapse_params);
}


template <class BaseSynapse, class DerivedSynapse>
typename core::Projection<BaseSynapse>::SynapseGenerator make_slicing_synapse_generator(
        const core::Projection<DerivedSynapse> &projection)
{
    // Getting projection by value as we don't know if the generator is used before original projection is destructed
    typename core::Projection<BaseSynapse>::SynapseGenerator generator = [projection](size_t step)
            -> std::optional<typename core::Projection<BaseSynapse>::Synapse>
    {
        if (step >= projection.size()) return {};
        const auto &element = projection[step];
        return std::make_tuple(slice_synapse<BaseSynapse, DerivedSynapse>(
                std::get<0>(element)), std::get<1>(element), std::get<2>(element));
    };
    return generator;
}


template <class BaseSynapse, class DerivedSynapse>
core::Projection<BaseSynapse> upcast_projection(const core::Projection<DerivedSynapse> &derived_projection)
{
    return core::Projection<BaseSynapse>{derived_projection.get_uid(), derived_projection.get_presynaptic(),
             derived_projection.get_postsynaptic(),
             make_slicing_synapse_generator<BaseSynapse, DerivedSynapse>(derived_projection),
             derived_projection.size()};
}
} // namespace knp::framework::projection::synapse_generators
