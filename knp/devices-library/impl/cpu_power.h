/**
 * @file cpu_power.h
 * @brief CPU power consumption get via Intel PCM library.
 * @author Artiom N.
 * @date 20.02.2023
 */

#pragma once

#if defined(__clang__)
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wdocumentation"
#endif

#include <pcm/src/cpucounters.h>

#include <chrono>
#include <cinttypes>
#include <vector>

#if defined(__clang__)
#    pragma clang diagnostic pop
#endif


namespace knp::devices::cpu
{
/**
 * @brief Power getter via Intel PCM.
 */
class CpuPower
{
public:
    CpuPower();

public:
    float get_power();

private:
    std::chrono::time_point<std::chrono::steady_clock> time_start_;
    pcm::PCM *pcm_instance_ = nullptr;
    std::vector<pcm::SocketCounterState> sktstate1_, sktstate2_;
    std::vector<pcm::CoreCounterState> /* cstates1, */ cstates2_;
    pcm::SystemCounterState sstate1_, sstate2_;
};
}  // namespace knp::devices::cpu