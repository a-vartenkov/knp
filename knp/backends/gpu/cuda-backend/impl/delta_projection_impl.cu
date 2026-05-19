//
// Created by vartenkov on 07.05.26.
//

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
#include <thrust/sort.h>
#include <thrust/binary_search.h>


REGISTER_CUDA_VECTOR_TYPE(knp::backends::gpu::cuda::CUDABackendImpl::PopulationVariants);
REGISTER_CUDA_VECTOR_TYPE(knp::backends::gpu::cuda::CUDABackendImpl::ProjectionVariants);
REGISTER_CUDA_VECTOR_TYPE(knp::backends::gpu::cuda::CUDAPopulation<knp::neuron_traits::BLIFATNeuron>);
REGISTER_CUDA_VECTOR_TYPE(knp::backends::gpu::cuda::CUDAProjection<knp::synapse_traits::DeltaSynapse>);
REGISTER_CUDA_VECTOR_TYPE(knp::backends::gpu::cuda::CUDAProjection<knp::synapse_traits::DeltaSynapse>::Synapse);
REGISTER_CUDA_VECTOR_TYPE(knp::backends::gpu::cuda::CUDAPopulation<knp::neuron_traits::BLIFATNeuron>::NeuronParameters);
REGISTER_CUDA_VECTOR_TYPE(knp::backends::gpu::cuda::device_lib::CUDAVector<unsigned long long>);
REGISTER_CUDA_VECTOR_TYPE(unsigned long long);
REGISTER_CUDA_VECTOR_TYPE(unsigned int);
REGISTER_CUDA_VECTOR_TYPE(knp::backends::gpu::cuda::Subscription);
REGISTER_CUDA_VECTOR_TYPE(knp::backends::gpu::cuda::MessageVariant);
REGISTER_CUDA_VECTOR_TYPE(knp::backends::gpu::cuda::UID);