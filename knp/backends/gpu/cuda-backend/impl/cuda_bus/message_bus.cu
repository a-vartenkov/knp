/**
 * @file message_bus.cu
 * @brief Message bus implementation.
 * @kaspersky_support Artiom N.
 * @date 21.02.2025
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

#include <algorithm>
#include <vector>
#include <boost/mp11/algorithm.hpp>
#include <boost/preprocessor.hpp>
#include <cuda/std/detail/libcxx/include/algorithm>
#include <thrust/device_ptr.h>
#include <knp/meta/macro.h>
#include "message_bus.cuh"

#include "../cuda_lib/vector.cuh"
#include "../cuda_lib/register_all.cuh"
#include "../cuda_lib/get_blocks_config.cuh"


namespace knp::backends::gpu::cuda
{
template <class T>
using DevVec = device_lib::CUDAVector<T>;


__global__ void find_subscription_by_receiver(const Subscription *subscriptions, size_t size, const UID receiver,
                                              size_t type, size_t *index_out)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    const Subscription &sub = subscriptions[i];
    if (sub.type() != type) return;
    if (sub.get_receiver_uid() == receiver) *index_out = i; // Should only work once, so no race condition problems.
}


template <typename MessageType>
__host__ size_t CUDAMessageBus::find_subscription(const cuda::UID &receiver)
{
    constexpr size_t type_index = boost::mp11::mp_find<MessageVariant, MessageType>::value;
    return find_subscription(receiver, type_index);
}


__host__ size_t CUDAMessageBus::find_subscription(const cuda::UID &receiver, size_t type_index)
{
    if (!subscriptions_.size()) return 0;
    auto [num_blocks, num_threads] = device_lib::get_blocks_config(subscriptions_.size());

    size_t *index_out;
    cudaMalloc(&index_out, sizeof(size_t));
    size_t subscriptions_size = subscriptions_.size();
    cudaMemcpy(index_out, &subscriptions_size, sizeof(size_t), cudaMemcpyHostToDevice);

    find_subscription_by_receiver<<<num_blocks, num_threads>>>(subscriptions_.data(), subscriptions_.size(), receiver,
                                                               type_index, index_out);

    size_t result;
    cudaMemcpy(&result, index_out, sizeof(size_t), cudaMemcpyDeviceToHost);
    cudaFree(index_out);
    return result;
}


template <typename MessageType>
__host__ bool CUDAMessageBus::unsubscribe(const cuda::UID &receiver)
{
    SPDLOG_DEBUG("Unsubscribing");
    size_t sub_index = find_subscription<MessageType>(receiver);
    if (sub_index >= subscriptions_.size())
    {
        SPDLOG_TRACE("No subscriptions found to unsubscribe from: returned {}", sub_index);
        return false;
    }
    SPDLOG_TRACE("Removing subscription #{}", sub_index);
    subscriptions_.erase(subscriptions_.begin() + sub_index, subscriptions_.begin() + sub_index + 1);
    SPDLOG_TRACE("Done unsubscribing");
    return true;
}


__host__ void CUDAMessageBus::remove_receiver(const cuda::UID &receiver)
{
    for (auto sub_iter = subscriptions_.begin(); sub_iter != subscriptions_.end(); ++sub_iter)
    {
        // TODO: Finish
    }
}


// This is not threadsafe, make sure it's not run in parallel.
__host__ __device__ void CUDAMessageBus::send_message(const cuda::MessageVariant &message)
{
    messages_to_route_.push_back(message);
}


__host__ void CUDAMessageBus::send_message_gpu_batch(const device_lib::CUDAVector<cuda::MessageVariant> &vec)
{
    SPDLOG_DEBUG("Sending {} GPU messages", vec.size());
    size_t msg_size = messages_to_route_.size();
    messages_to_route_.resize(msg_size + vec.size());
    if (!vec.size()) return;
    auto [num_blocks, num_threads] = device_lib::get_blocks_config(vec.size());
    device_lib::copy_kernel<<<num_blocks, num_threads>>>(messages_to_route_.data() + msg_size, vec.size(), vec.data());
    cudaDeviceSynchronize();
}


/**
 * @brief Find all message indices that correspond to the current subscription.
 * @param messages pointer to messages
 * @param messages_size number of messages
 * @param subscription the subscription used for searching
 * @param indexes
 * @param counter a zero-initialized counter, it would be equal to number of found messages after the function finishes
 * @return
 */
__global__ void find_messages_kernel(const MessageVariant *messages, size_t messages_size, Subscription *subscription,
                              unsigned long long *indices, unsigned long long *counter)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Find message kernel, i %lu\n", i);
    if (i >= messages_size) return;
    if (subscription->is_my_message(messages[i]))
    {
        unsigned long long index = atomicAdd(counter, 1ull);
        printf("Found message: index %lu, message_index %lu\n", i, index);
        indices[index] = i;
    }
    else
        printf("No message found!\n");
}


template <class MessageType>
device_lib::CUDAVector<unsigned long long> CUDAMessageBus::unload_messages(const cuda::UID &receiver_uid)
{
    SPDLOG_DEBUG("Unloading messages from GPU message bus for receiver {}.", std::string(to_cpu_uid(receiver_uid)));
    size_t sub_index = find_subscription<MessageType>(receiver_uid);
    if (sub_index >= subscriptions_.size()) return device_lib::CUDAVector<unsigned long long>{};
    if (!messages_to_route_.size()) return device_lib::CUDAVector<unsigned long long>{};
    Subscription subscription = subscriptions_.copy_at(sub_index);
    constexpr size_t message_type = boost::mp11::mp_find<MessageVariant, MessageType>();
    SPDLOG_TRACE("There is an associated type-{} subscription and the bus is non-empty.", message_type);
    if (subscription.type() != message_type) return device_lib::CUDAVector<unsigned long long>{};

    auto [num_blocks, num_threads] = device_lib::get_blocks_config(messages_to_route_.size());
    unsigned long long *counter;
    // cudaMallocManaged(&counter, 0);
    call_and_check(cudaMalloc(&counter, sizeof(unsigned long long)));
    cudaMemset(counter, 0, sizeof(unsigned long long));
    unsigned long long *indices;
    cudaMalloc(&indices, sizeof(unsigned long long) * messages_to_route_.size());
    SPDLOG_TRACE("Starting kernel: messages {}, {}, subscription {}, {}, i{}", (void *)(messages_to_route_.data()),
                 messages_to_route_.size(), (void *)(subscriptions_.data()), subscriptions_.size(), sub_index);
    find_messages_kernel<<<num_blocks, num_threads>>>(messages_to_route_.data(), messages_to_route_.size(),
               subscriptions_.data() + sub_index, indices, counter);
    // Here we have a set of message indexes. Is that enough? I think it is.
    size_t cpu_counter;
    call_and_check(cudaDeviceSynchronize());
    call_and_check(cudaMemcpy(&cpu_counter, counter, sizeof(cpu_counter), cudaMemcpyDeviceToHost));
    call_and_check(cudaFree(counter));
    SPDLOG_TRACE("Found {} incoming messages.", cpu_counter);
    device_lib::CUDAVector<unsigned long long> result(cpu_counter);
    if (cpu_counter != 0)
    {
        auto [num_blocks_copy, num_threads_copy] = device_lib::get_blocks_config(cpu_counter);
        device_lib::copy_kernel<<<num_blocks_copy, num_threads_copy>>>(result.data(), cpu_counter, indices);
    }
    cudaDeviceSynchronize();
    call_and_check(cudaFree(indices));
    return result;
}


__global__ void get_message_kernel(const MessageVariant *var, int *type, const void **msg) {
    int type_val = var->index();
    switch (type_val)
    {
        // TODO : Add more after adding new messages
        static_assert(::cuda::std::variant_size<cuda::MessageVariant>() == 2, "Add a case statement here!");
        case 0:
            *msg = ::cuda::std::get_if<0>(var);
            break;
        case 1:
            *msg = ::cuda::std::get_if<1>(var);
            break;
        default:
            *msg = nullptr;
    }
    *type = type_val;
}


__host__ void CUDAMessageBus::subscribe_host(const cuda::UID &receiver, const std::vector<cuda::UID> &senders,
                                             size_t type_id)
{
    knp::core::UID host_receiver = to_cpu_uid(receiver);
    std::vector <knp::core::UID> host_senders;
    host_senders.reserve(senders.size());
    for (const cuda::UID &cuda_uid : senders)
    {
        host_senders.push_back(to_cpu_uid(cuda_uid));
    }
    // TODO: Do it in a normal way maybe.
    static_assert(::cuda::std::variant_size<cuda::MessageVariant>() == 2, "Add a case statement here!");
    switch (type_id)
    {
        case 0:
            cpu_endpoint_.subscribe < boost::mp11::mp_at_c < knp::core::messaging::MessageVariant, 0 >> (
                    host_receiver, host_senders);
            break;
        case 1:
            cpu_endpoint_.subscribe < boost::mp11::mp_at_c < knp::core::messaging::MessageVariant, 1 >> (
                    host_receiver, host_senders);
            break;
    }
}


__host__ void CUDAMessageBus::send_messages_to_host(size_t step)
{
    for (size_t i = 0; i < messages_to_route_.size(); ++i)
    {
        cuda::MessageVariant msg = messages_to_route_.copy_at(i);
        cuda::MessageHeader header = ::cuda::std::visit(
                [](const auto &msg) -> cuda::MessageHeader
                { return msg.header_; },
            msg);
        if (header.send_time_ != step - 1 && header.send_time_ != step || header.is_external_) continue;
        auto host_message = make_host_message(msg);
        cpu_endpoint_.send_message(host_message);
    }
}


// We need to check that the messages we get from host were not previously sent there by GPU.
__global__ void same_sender_kernel(cuda::UID uid, cuda::Subscription *subscriptions, size_t sub_size, bool *result)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= sub_size) return;
    if (subscriptions[i].get_receiver_uid() == uid) *result = true;
}


__host__ bool same_sender(const knp::core::messaging::MessageVariant &message,
                          CUDAMessageBus::SubscriptionContainer &subs)
{
    if (subs.size() == 0) return false;

    knp::core::UID sender_uid = std::visit([](const auto &msg)
    {
        return msg.header_.sender_uid_;
    }, message);

    cuda::UID gpu_uid = to_gpu_uid(sender_uid);
    bool result = false;
    bool *result_ptr;

    call_and_check(cudaMalloc(&result_ptr, sizeof(bool)));
    call_and_check(cudaMemcpy(result_ptr, &result, sizeof(bool), cudaMemcpyHostToDevice));
    auto [num_blocks, num_threads] = device_lib::get_blocks_config(subs.size());
    same_sender_kernel<<<num_blocks, num_threads>>>(gpu_uid, subs.data(), subs.size(), result_ptr);
    call_and_check(cudaMemcpy(&result, result_ptr, sizeof(bool), cudaMemcpyDeviceToHost));
    call_and_check(cudaFree(result_ptr));
    return result;
}


/**
 * @brief Copy host subscriptions here.
 */
__host__ void CUDAMessageBus::sync_with_host()
{
    int device;
    cudaGetDevice(&device);
    SPDLOG_DEBUG("Current CUDA device: {}", device);
    for (const auto &cpu_subscription : cpu_endpoint_.get_endpoint_subscriptions())
    {
        cuda::Subscription gpu_sub{cpu_subscription.second};
        std::vector<cuda::UID> senders;
        const auto &gpu_senders = gpu_sub.get_senders();
        senders.resize(gpu_senders.size());
        cudaMemcpy(senders.data(), gpu_senders.data(), sizeof(cuda::UID) * gpu_senders.size(), cudaMemcpyDeviceToHost);
        subscribe_both(gpu_sub.get_receiver_uid(), senders, gpu_sub.type());
    }
}


// TODO: Maybe template this or something.
__host__ void CUDAMessageBus::receive_messages_from_host()
{
    cpu_endpoint_.receive_all_messages();
    SPDLOG_DEBUG("CPU subscriptions: {}", cpu_endpoint_.get_endpoint_subscriptions().size());
    for (size_t i = 0; i < subscriptions_.size(); ++i)
    {
        auto sub = subscriptions_.copy_at(i);
        auto receiver_uid = to_cpu_uid(sub.get_receiver_uid());
        size_t type = sub.type();
        // TODO: do it in a proper way or rather just rework the whole mechanic.
        if (type == 0)
        {
            using MessageType = boost::mp11::mp_at_c<knp::core::messaging::MessageVariant, 0>;
            std::vector <knp::core::messaging::SpikeMessage> message_buf
                    = cpu_endpoint_.unload_messages<knp::core::messaging::SpikeMessage>(receiver_uid);
            for (auto &msg : message_buf)
            {
                auto cpu_msg_var = knp::core::messaging::MessageVariant{msg};
                cuda::MessageVariant gpu_msg = make_gpu_message(cpu_msg_var);

                send_message(gpu_msg);
            }
        }
        else if (type == 1)
        {
            using MessageType = boost::mp11::mp_at_c<knp::core::messaging::MessageVariant, 1>;
            std::vector<MessageType> message_buf
                    = cpu_endpoint_.unload_messages<MessageType>(receiver_uid);
            for (auto &msg : message_buf)
            {
                auto cpu_msg_var = knp::core::messaging::MessageVariant{msg};
                cuda::MessageVariant gpu_msg = make_gpu_message(cpu_msg_var);
                send_message(gpu_msg);
            }
        }
    }
}


namespace cm = knp::backends::gpu::cuda;

template
__host__ bool cm::CUDAMessageBus::subscribe_gpu<SpikeMessage>(const cm::UID&, const std::vector<cuda::UID>&);
template
__host__ bool cm::CUDAMessageBus::subscribe_gpu<SynapticImpactMessage>(const cm::UID&, const std::vector<cuda::UID>&);

template
__host__ bool cm::CUDAMessageBus::subscribe_both<SpikeMessage>(const cm::UID&, const std::vector<cuda::UID>&);
template
__host__ bool cm::CUDAMessageBus::subscribe_both<SynapticImpactMessage>(const cm::UID&, const std::vector<cuda::UID>&);

template
__host__ cm::device_lib::CUDAVector<unsigned long long> cm::CUDAMessageBus::unload_messages<SpikeMessage>(
        const cm::UID &receiver_uid);

template
__host__ cm::device_lib::CUDAVector<unsigned long long> cm::CUDAMessageBus::unload_messages<SynapticImpactMessage>(
        const cm::UID &receiver_uid);


#define INSTANCE_MESSAGES_FUNCTIONS(n, template_for_instance, message_type)                \
    template bool CUDAMessageBus::unsubscribe<cm::message_type>(const cuda::UID &receiver);

BOOST_PP_SEQ_FOR_EACH(INSTANCE_MESSAGES_FUNCTIONS, "", BOOST_PP_VARIADIC_TO_SEQ(ALL_CUDA_MESSAGES))


}  // namespace knp::backends::gpu::cuda

REGISTER_CUDA_VECTOR_TYPE(unsigned long long);
REGISTER_CUDA_VECTOR_TYPE(unsigned int);
REGISTER_CUDA_VECTOR_TYPE(knp::backends::gpu::cuda::device_lib::CUDAVector<unsigned long long>);
REGISTER_CUDA_VECTOR_TYPE(knp::backends::gpu::cuda::Subscription);
REGISTER_CUDA_VECTOR_TYPE(knp::backends::gpu::cuda::MessageVariant);
REGISTER_CUDA_VECTOR_TYPE(knp::backends::gpu::cuda::SpikeMessage);
REGISTER_CUDA_VECTOR_TYPE(knp::backends::gpu::cuda::SynapticImpactMessage);
REGISTER_CUDA_VECTOR_TYPE(knp::backends::gpu::cuda::SynapticImpact);
REGISTER_CUDA_VECTOR_TYPE(knp::backends::gpu::cuda::UID);
