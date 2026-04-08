//
// Created by vartenkov on 02.04.26.
//

#pragma once

#include <cuda_runtime.h>
#include <spdlog/spdlog.h>

#define FAST_ERROR_CHECK(error_message) \
{ auto error = cudaGetLastError(); if (error != cudaSuccess) SPDLOG_ERROR(error_message, cudaGetErrorString(error)); }
