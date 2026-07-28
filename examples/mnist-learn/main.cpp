/**
 * @file main.cpp
 * @brief Example of training a MNIST network.
 * @kaspersky_support D. Postnikov
 * @date 03.02.2026
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

#include <iostream>

#include "dataset.h"
#include "evaluate_results.h"
#include "inference.h"
#include "network_validation.h"
#include "parse_arguments.h"
#include "save_network.h"
#include "training.h"


/**
 * @brief Execute complete model pipeline for specified neuron type.
 *
 * @details This template function orchestrates the entire machine learning pipeline for neural networks, including
 * dataset processing, network construction, training, inference, and evaluation. It serves as the core execution
 * engine for both AltAI and BLIFAT neuron models.
 *
 * @tparam Neuron neuron type for neuron model specification.
 *
 * @param model_desc model description containing configuration parameters and paths.
 */
template <typename Neuron>
void run_model(const ModelDescription& model_desc)
{
    Dataset dataset = process_dataset(model_desc);
    AnnotatedNetwork network;
    knp::framework::BackendLoader backend_loader;
    if (!model_desc.inference_only_)
    {
        // Online Help link: https://click.kaspersky.com/?hl=en-US&version=2.0&pid=KNP&link=online_help&helpid=243548
        network = construct_network<Neuron>(model_desc);
        train_model<Neuron>(model_desc, dataset, network, backend_loader);
        // Сonvert model to the most base neuron and synapse classes.
        if (model_desc.type_ == SupportedModelType::BLIFAT)
            network.network_.upcast_populations<knp::neuron_traits::BLIFATNeuron>();
        else if (model_desc.type_ == SupportedModelType::AltAI)
            network.network_.upcast_populations<knp::neuron_traits::AltAILIF>();
        else throw std::runtime_error("Unsupported model type");

        network.network_.upcast_projections<knp::synapse_traits::DeltaSynapse>();
    }
    if (!model_desc.model_saving_path_.empty())
    {
        if (!model_desc.inference_only_)
        {
            save_network(model_desc, network);
        }
        knp::framework::Network new_network = knp::framework::sonata::load_network(model_desc.model_saving_path_);
        if (new_network.populations_count() != network.network_.populations_count()
        || new_network.projections_count() != network.network_.projections_count())
        {
            std::cout << "Populations " << new_network.populations_count() << " vs. "
                      << network.network_.populations_count() << std::endl;
            std::cout << "Projections: " << new_network.projections_count() << " vs. "
                      << network.network_.projections_count() << std::endl;
        }
        network.network_ = new_network;
    }
    else if (model_desc.inference_only_) // inference without model saving.
    {
        throw std::runtime_error("Model leading path is not defined for inference-only mode");
    };


    // Execute inference on test data.
    auto inference_spikes = infer_model<Neuron>(model_desc, dataset, network, backend_loader);

    // Evaluate and report inference results.
    evaluate_results(inference_spikes, dataset);
}


/**
 * @brief Main application entry point.
 *
 * @details This function serves as the primary execution point for the MNIST neural network learning application.
 * It handles command-line argument parsing, configuration validation, user interaction, and routes execution to
 * the appropriate neuron model.
 *
 * @param argc argument count.
 * @param argv arguments values.
 * 
 * @return exit code.
 */
int main(int argc, char** argv)
{
    // Parse command-line arguments and validate configuration.
    std::optional<ModelDescription> model_desc_opt = parse_arguments(argc, argv);
    if (!model_desc_opt.has_value()) return EXIT_FAILURE;
    const ModelDescription& model_desc = model_desc_opt.value();

    // Display configuration to user for confirmation.
    std::cout << "Model description:\n"
              << model_desc << "\nPress ENTER to accept parameters and start model." << std::endl;
    std::cin.get();
    std::cout << "Starting model." << std::endl;

    // Execute model according to selected neuron type.
    switch (model_desc.type_)
    {
        case SupportedModelType::BLIFAT:
        {
            // cppcheck-suppress throwInEntryPoint
            run_model<knp::neuron_traits::BLIFATNeuron>(model_desc);
            break;
        }
        case SupportedModelType::AltAI:
        {
            run_model<knp::neuron_traits::AltAILIF>(model_desc);
            break;
        }
        default:
            throw std::runtime_error("Unknown model type.");
    }

    return EXIT_SUCCESS;
}
