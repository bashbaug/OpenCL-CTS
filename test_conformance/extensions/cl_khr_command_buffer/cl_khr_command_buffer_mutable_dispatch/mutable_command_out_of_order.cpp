//
// Copyright (c) 2022 The Khronos Group Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#include <vector>

#include <extensionHelpers.h>
#include "typeWrappers.h"
#include "procs.h"
#include "testHarness.h"
#include "mutable_command_basic.h"

#include <CL/cl.h>
#include <CL/cl_ext.h>
////////////////////////////////////////////////////////////////////////////////
// mutable dispatch tests which handle following cases:
// - simultaneous use
// - cross-queue simultaneous-use

namespace {

template <bool simultaneous_request>
struct OutOfOrderTest : public BasicMutableCommandBufferTest
{
    OutOfOrderTest(cl_device_id device, cl_context context,
                   cl_command_queue queue)
        : BasicMutableCommandBufferTest(device, context, queue),
          out_of_order_queue(nullptr), out_of_order_command_buffer(this),
          user_event(nullptr), wait_pass_event(nullptr), kernel_fill(nullptr),
          program_fill(nullptr)
    {
        simultaneous_use_requested = simultaneous_request;
        if (simultaneous_request) buffer_size_multiplier = 2;
    }

    //--------------------------------------------------------------------------
    cl_int SetUpKernel() override
    {
        // if device doesn't support simultaneous use which was requested
        // we can skip creation of OCL resources
        if (simultaneous_use_requested && !simultaneous_use_support)
            return CL_SUCCESS;

        cl_int error = BasicMutableCommandBufferTest::SetUpKernel();
        test_error(error, "BasicMutableCommandBufferTest::SetUpKernel failed");

        // create additional kernel to properly prepare output buffer for test
        const char* kernel_str =
            R"(
          __kernel void fill(int pattern, __global int* out, __global int*
        offset)
          {
              size_t id = get_global_id(0);
              size_t ind = offset[0] + id ;
              out[ind] = pattern;
          })";

        error = create_single_kernel_helper_create_program(
            context, &program_fill, 1, &kernel_str);
        test_error(error, "Failed to create program with source");

        error =
            clBuildProgram(program_fill, 1, &device, nullptr, nullptr, nullptr);
        test_error(error, "Failed to build program");

        kernel_fill = clCreateKernel(program_fill, "fill", &error);
        test_error(error, "Failed to create copy kernel");

        return CL_SUCCESS;
    }

    //--------------------------------------------------------------------------
    cl_int SetUpKernelArgs() override
    {
        // if device doesn't support simultaneous use which was requested
        // we can skip creation of OCL resources
        if (simultaneous_use_requested && !simultaneous_use_support)
            return CL_SUCCESS;

        cl_int error = BasicMutableCommandBufferTest::SetUpKernelArgs();
        test_error(error,
                   "BasicMutableCommandBufferTest::SetUpKernelArgs failed");

        error = clSetKernelArg(kernel_fill, 0, sizeof(cl_int),
                               &overwritten_pattern);
        test_error(error, "clSetKernelArg failed");

        error = clSetKernelArg(kernel_fill, 1, sizeof(out_mem), &out_mem);
        test_error(error, "clSetKernelArg failed");

        error = clSetKernelArg(kernel_fill, 2, sizeof(off_mem), &off_mem);
        test_error(error, "clSetKernelArg failed");

        return CL_SUCCESS;
    }

    //--------------------------------------------------------------------------
    cl_int SetUp(int elements) override
    {
        cl_int error = BasicMutableCommandBufferTest::SetUp(elements);
        test_error(error, "BasicMutableCommandBufferTest::SetUp failed");

        cl_platform_id platform;
        error = clGetDeviceInfo(device, CL_DEVICE_PLATFORM,
                                sizeof(cl_platform_id), &platform, nullptr);
        test_error(error, "clGetDeviceInfo for CL_DEVICE_PLATFORM failed");

        GET_EXTENSION_ADDRESS(clUpdateMutableCommandsKHR);

        error = SetUpKernel();
        test_error(error, "SetUpKernel failed");

        error = SetUpKernelArgs();
        test_error(error, "SetUpKernelArgs failed");

        out_of_order_queue = clCreateCommandQueue(
            context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &error);
        test_error(error, "Unable to create command queue to test with");

        cl_command_buffer_properties_khr properties[3] = {
            CL_COMMAND_BUFFER_FLAGS_KHR, 0, 0
        };

        if (simultaneous_use_requested && simultaneous_use_support)
            properties[1] |= CL_COMMAND_BUFFER_SIMULTANEOUS_USE_KHR;

        properties[1] |= CL_COMMAND_BUFFER_MUTABLE_KHR;

        out_of_order_command_buffer = clCreateCommandBufferKHR(
            1, &out_of_order_queue, properties, &error);
        test_error(error, "clCreateCommandBufferKHR failed");

        return CL_SUCCESS;
    }

    //--------------------------------------------------------------------------
    bool Skip() override
    {
        if (BasicMutableCommandBufferTest::Skip()) return true;

        if (!out_of_order_support
            || (simultaneous_use_requested && !simultaneous_use_support))
            return true;

        return false;
    }

    //--------------------------------------------------------------------------
    cl_int Run() override
    {
        cl_int error = CL_SUCCESS;

        if (simultaneous_use_support)
        {
            // enqueue simultaneous command-buffers with out-of-order calls
            error = RunSimultaneous();
            test_error(error, "RunSimultaneous failed");
        }
        else
        {
            // enqueue single command-buffer with out-of-order calls
            error = RunSingle();
            test_error(error, "RunSingle failed");
        }

        return CL_SUCCESS;
    }

    //--------------------------------------------------------------------------
    cl_int RecordCommandBuffer()
    {
        cl_sync_point_khr sync_points[2];
        const cl_int pattern = pattern_pri;
        cl_int error =
            clCommandFillBufferKHR(out_of_order_command_buffer, nullptr, in_mem,
                                   &pattern, sizeof(cl_int), 0, data_size(), 0,
                                   nullptr, &sync_points[0], nullptr);
        test_error(error, "clCommandFillBufferKHR failed");

        error = clCommandFillBufferKHR(out_of_order_command_buffer, nullptr,
                                       out_mem, &overwritten_pattern,
                                       sizeof(cl_int), 0, data_size(), 0,
                                       nullptr, &sync_points[1], nullptr);
        test_error(error, "clCommandFillBufferKHR failed");

        error = clCommandNDRangeKernelKHR(
            out_of_order_command_buffer, nullptr, nullptr, kernel, 1, nullptr,
            &num_elements, nullptr, 2, sync_points, nullptr, &command);
        test_error(error, "clCommandNDRangeKernelKHR failed");

        error = clFinalizeCommandBufferKHR(out_of_order_command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");

        return CL_SUCCESS;
    }

    //--------------------------------------------------------------------------
    cl_int RunSingle()
    {
        cl_int error = RecordCommandBuffer();
        test_error(error, "RecordCommandBuffer failed");

        error = clEnqueueCommandBufferKHR(
            0, nullptr, out_of_order_command_buffer, 0, nullptr, &user_event);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        std::vector<cl_int> output_data(num_elements);
        error = clEnqueueReadBuffer(out_of_order_queue, out_mem, CL_TRUE, 0,
                                    data_size(), output_data.data(), 1,
                                    &user_event, nullptr);
        test_error(error, "clEnqueueReadBuffer failed");

        cl_mutable_dispatch_exec_info_khr exec_info_list{
            CL_KERNEL_EXEC_INFO_SVM_PTRS, sizeof(in_mem), in_mem
        };

        cl_mutable_dispatch_config_khr dispatch_config{
            CL_STRUCTURE_TYPE_MUTABLE_DISPATCH_CONFIG_KHR,
            nullptr,
            command,
            0 /* num_args */,
            0 /* num_svm_arg */,
            1 /* num_exec_infos */,
            0 /* work_dim - 0 means no change to dimensions */,
            nullptr /* arg_list */,
            nullptr /* arg_svm_list - nullptr means no change*/,
            &exec_info_list /* exec_info_list */,
            nullptr /* global_work_offset */,
            nullptr /* global_work_size */,
            nullptr /* local_work_size */
        };
        cl_mutable_base_config_khr mutable_config{
            CL_STRUCTURE_TYPE_MUTABLE_BASE_CONFIG_KHR, nullptr, 1,
            &dispatch_config
        };

        error = clUpdateMutableCommandsKHR(out_of_order_command_buffer,
                                           &mutable_config);
        test_error(error, "clUpdateMutableCommandsKHR failed");

        for (size_t i = 0; i < num_elements; i++)
        {
            CHECK_VERIFICATION_ERROR(pattern_pri, output_data[i], i);
        }

        return CL_SUCCESS;
    }

    //--------------------------------------------------------------------------
    cl_int RecordSimultaneousCommandBuffer()
    {
        cl_sync_point_khr sync_points[2];
        // for both simultaneous passes this call will fill entire in_mem buffer
        cl_int error = clCommandFillBufferKHR(
            out_of_order_command_buffer, nullptr, in_mem, &pattern_pri,
            sizeof(cl_int), 0, data_size() * buffer_size_multiplier, 0, nullptr,
            &sync_points[0], nullptr);
        test_error(error, "clCommandFillBufferKHR failed");

        // to avoid overwriting the entire result buffer instead of filling
        // only relevant part this additional kernel was introduced

        error = clCommandNDRangeKernelKHR(out_of_order_command_buffer, nullptr,
                                          nullptr, kernel_fill, 1, nullptr,
                                          &num_elements, nullptr, 0, nullptr,
                                          &sync_points[1], &command);
        test_error(error, "clCommandNDRangeKernelKHR failed");

        error = clCommandNDRangeKernelKHR(
            out_of_order_command_buffer, nullptr, nullptr, kernel, 1, nullptr,
            &num_elements, nullptr, 2, sync_points, nullptr, &command);
        test_error(error, "clCommandNDRangeKernelKHR failed");

        error = clFinalizeCommandBufferKHR(out_of_order_command_buffer);
        test_error(error, "clFinalizeCommandBufferKHR failed");

        cl_mutable_dispatch_exec_info_khr exec_info_list{
            CL_KERNEL_EXEC_INFO_SVM_PTRS, sizeof(in_mem), in_mem
        };

        cl_mutable_dispatch_config_khr dispatch_config{
            CL_STRUCTURE_TYPE_MUTABLE_DISPATCH_CONFIG_KHR,
            nullptr,
            command,
            0 /* num_args */,
            0 /* num_svm_arg */,
            1 /* num_exec_infos */,
            0 /* work_dim - 0 means no change to dimensions */,
            nullptr /* arg_list */,
            nullptr /* arg_svm_list - nullptr means no change*/,
            &exec_info_list /* exec_info_list */,
            nullptr /* global_work_offset */,
            nullptr /* global_work_size */,
            nullptr /* local_work_size */
        };
        cl_mutable_base_config_khr mutable_config{
            CL_STRUCTURE_TYPE_MUTABLE_BASE_CONFIG_KHR, nullptr, 1,
            &dispatch_config
        };

        error = clUpdateMutableCommandsKHR(out_of_order_command_buffer,
                                           &mutable_config);
        test_error(error, "clUpdateMutableCommandsKHR failed");

        return CL_SUCCESS;
    }

    //--------------------------------------------------------------------------
    struct SimulPassData
    {
        cl_int offset;
        std::vector<cl_int> output_buffer;
        // 0:user event, 1:offset-buffer fill event, 2:kernel done event
        clEventWrapper wait_events[3];
    };

    //--------------------------------------------------------------------------
    cl_int EnqueueSimultaneousPass(SimulPassData& pd)
    {
        cl_int error = CL_SUCCESS;
        if (!user_event)
        {
            user_event = clCreateUserEvent(context, &error);
            test_error(error, "clCreateUserEvent failed");
        }

        pd.wait_events[0] = user_event;

        // filling offset buffer must wait for previous pass completeness
        error = clEnqueueFillBuffer(
            out_of_order_queue, off_mem, &pd.offset, sizeof(cl_int), 0,
            sizeof(cl_int), (wait_pass_event != nullptr ? 1 : 0),
            (wait_pass_event != nullptr ? &wait_pass_event : nullptr),
            &pd.wait_events[1]);
        test_error(error, "clEnqueueFillBuffer failed");

        // command buffer execution must wait for two wait-events
        error = clEnqueueCommandBufferKHR(
            0, nullptr, out_of_order_command_buffer, 2, &pd.wait_events[0],
            &pd.wait_events[2]);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        error = clEnqueueReadBuffer(out_of_order_queue, out_mem, CL_FALSE,
                                    pd.offset * sizeof(cl_int), data_size(),
                                    pd.output_buffer.data(), 1,
                                    &pd.wait_events[2], nullptr);
        test_error(error, "clEnqueueReadBuffer failed");

        return CL_SUCCESS;
    }

    //--------------------------------------------------------------------------
    cl_int RunSimultaneous()
    {
        cl_int error = RecordSimultaneousCommandBuffer();
        test_error(error, "RecordSimultaneousCommandBuffer failed");

        cl_int offset = static_cast<cl_int>(num_elements);

        std::vector<SimulPassData> simul_passes = {
            { 0, std::vector<cl_int>(num_elements) },
            { offset, std::vector<cl_int>(num_elements) }
        };

        for (auto&& pass : simul_passes)
        {
            error = EnqueueSimultaneousPass(pass);
            test_error(error, "EnqueueSimultaneousPass failed");

            wait_pass_event = pass.wait_events[2];
        }

        error = clSetUserEventStatus(user_event, CL_COMPLETE);
        test_error(error, "clSetUserEventStatus failed");

        error = clFinish(out_of_order_queue);
        test_error(error, "clFinish failed");

        // verify the result buffers
        for (auto&& pass : simul_passes)
        {
            auto& res_data = pass.output_buffer;
            for (size_t i = 0; i < num_elements; i++)
            {
                CHECK_VERIFICATION_ERROR(pattern_pri, res_data[i], i);
            }
        }

        return CL_SUCCESS;
    }

    //--------------------------------------------------------------------------
    clCommandQueueWrapper out_of_order_queue;
    clCommandBufferWrapper out_of_order_command_buffer;

    clEventWrapper user_event;
    clEventWrapper wait_pass_event;

    clKernelWrapper kernel_fill;
    clProgramWrapper program_fill;

    const size_t test_global_work_size = 3 * sizeof(cl_int);
    cl_mutable_command_khr command = nullptr;
    clUpdateMutableCommandsKHR_fn clUpdateMutableCommandsKHR = nullptr;

    const cl_int overwritten_pattern = 0xACDC;
    const cl_int pattern_pri = 42;
};

} // anonymous namespace

int test_mutable_dispatch_out_of_order(cl_device_id device, cl_context context,
                                       cl_command_queue queue, int num_elements)
{
    return MakeAndRunTest<OutOfOrderTest<false>>(device, context, queue,
                                                 num_elements);
}

int test_mutable_dispatch_simultaneous_out_of_order(cl_device_id device,
                                                    cl_context context,
                                                    cl_command_queue queue,
                                                    int num_elements)
{
    return MakeAndRunTest<OutOfOrderTest<true>>(device, context, queue,
                                                num_elements);
}
