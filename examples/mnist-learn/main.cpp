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
 * @brief Run model with just inference.
 * @tparam Neuron
 * @param model_desc
 * @param dataset
 * @return
 */
template <typename Neuron>
std::vector<knp::core::messaging::SpikeMessage> run_model_inference_only(const ModelDescription& model_desc,
                                                                         const Dataset& dataset)
{
    AnnotatedNetwork network;
    knp::framework::BackendLoader backend_loader;
    if (model_desc.model_saving_path_.empty())
        throw std::runtime_error("Model loading path is not defined for inference-only mode");

    // Execute inference on test data.
    return infer_model<Neuron>(model_desc, dataset, network, backend_loader);
}


/**
 * @brief Run model with training.
 * @tparam Neuron neuron base type.
 * @param model_desc model description containing configuration parameters and paths.
 * @param dataset dataset.
 * @return vector of output spike messages.
 */
template <typename Neuron>
std::vector<knp::core::messaging::SpikeMessage> run_model_training(const ModelDescription& model_desc,
                                                                   const Dataset& dataset)
{
    AnnotatedNetwork network;
    knp::framework::BackendLoader backend_loader;
    // Online Help link: https://click.kaspersky.com/?hl=en-US&version=2.0&pid=KNP&link=online_help&helpid=243548
    network = construct_network<Neuron>(model_desc);
    train_model<Neuron>(model_desc, dataset, network, backend_loader);
    // Сonvert model to the most base neuron and synapse classes.
    if (model_desc.type_ == SupportedModelType::BLIFAT)
        knp::framework::upcast_populations<knp::neuron_traits::BLIFATNeuron>(network.network_);
    else if (model_desc.type_ == SupportedModelType::AltAI)
        knp::framework::upcast_populations<knp::neuron_traits::AltAILIF>(network.network_);
    else throw std::runtime_error("Unsupported model type");

    knp::framework::upcast_projections<knp::synapse_traits::DeltaSynapse>(network.network_);

    return infer_model<Neuron>(model_desc, dataset, network, backend_loader);
}


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
    std::vector<knp::core::messaging::SpikeMessage> results;
    if (model_desc.inference_only_) results = run_model_inference_only<Neuron>(model_desc, dataset);
    else results = run_model_training<Neuron>(model_desc, dataset);

    // Evaluate and report inference results.
    evaluate_results(results, dataset);
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
