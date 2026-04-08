//
// Created by vartenkov on 06.04.26.
//

#pragma once
#include "register_type.cuh"

#define REGISTER_ALL_TYPES \
REGISTER_CUDA_VECTOR_TYPE(knp::backends::gpu::cuda::CUDABackendImpl::PopulationVariants); \
REGISTER_CUDA_VECTOR_TYPE(knp::backends::gpu::cuda::CUDABackendImpl::ProjectionVariants); \
REGISTER_CUDA_VECTOR_TYPE(knp::backends::gpu::cuda::CUDAPopulation<knp::neuron_traits::BLIFATNeuron>); \
REGISTER_CUDA_VECTOR_TYPE(knp::backends::gpu::cuda::CUDAProjection<knp::synapse_traits::DeltaSynapse>); \
REGISTER_CUDA_VECTOR_TYPE(knp::backends::gpu::cuda::CUDAProjection<knp::synapse_traits::DeltaSynapse>::Synapse); \
REGISTER_CUDA_VECTOR_TYPE(uint64_t); \
REGISTER_CUDA_VECTOR_TYPE(knp::backends::gpu::cuda::Subscription); \
REGISTER_CUDA_VECTOR_TYPE(knp::backends::gpu::cuda::MessageVariant); \
REGISTER_CUDA_VECTOR_TYPE(knp::backends::gpu::cuda::UID)
