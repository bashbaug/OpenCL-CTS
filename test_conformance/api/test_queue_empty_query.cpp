//
// Copyright (c) 2025 The Khronos Group Inc.
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
#include "testBase.h"

// temporary!
cl_command_queue_info CL_QUEUE_EMPTY = 0x31337;

static const char* timesink_kernel = R"CLC(
kernel void TimeSink(global float* dst)
{
    float result = 0.0f;
    while (result < 1.0f) result += 1e-6f;
    dst[get_global_id(0)] = result;
}
)CLC";

static int test_queue_empty_helper(cl_command_queue_properties props,
                                   cl_context context, cl_device_id device,
                                   cl_kernel kernel, cl_mem buf)
{
    cl_bool isEmpty = CL_FALSE;
    cl_int error = CL_SUCCESS;

    // Check that the passed-in properties are supported by the device
    cl_command_queue_properties device_props = 0;
    error = clGetDeviceInfo(device, CL_DEVICE_QUEUE_PROPERTIES,
                            sizeof(device_props), &device_props, nullptr);
    test_error(error, "clGetDeviceInfo for CL_DEVICE_QUEUE_PROPERTIES failed");
    if ((props & device_props) != props)
    {
        return TEST_PASS;
    }

    // Check that a newly created command queue is empty
    clCommandQueueWrapper queue =
        clCreateCommandQueue(context, device, props, &error);
    test_error(error, "Unable to create command queue");

    error = clGetCommandQueueInfo(queue, CL_QUEUE_EMPTY, sizeof(isEmpty),
                                  &isEmpty, nullptr);
    test_error(error, "Querying command queue empty status failed");
    test_assert_error(isEmpty == CL_TRUE,
                      "Command queue is not empty after creation");

    // Try a test without a dependent user event.
    // This may not work in all cases, but this is the most likely real-world
    // usage for the queue empty query. If the host is too slow, or the device
    // is too fast, we won't be able to test the queue empty status, in which
    // case we will just pass the test.
    {
        const size_t globalWorkSize = 1;
        clEventWrapper event;
        error =
            clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalWorkSize,
                                   nullptr, 0, nullptr, &event);
        test_error(error, "Unable to enqueue kernel");

        error = clGetCommandQueueInfo(queue, CL_QUEUE_EMPTY, sizeof(isEmpty),
                                      &isEmpty, nullptr);
        test_error(error, "Querying command queue empty status failed");

        cl_int eventStatus = CL_COMPLETE;
        error = clGetEventInfo(event, CL_EVENT_COMMAND_EXECUTION_STATUS,
                               sizeof(eventStatus), &eventStatus, nullptr);
        test_error(error, "Querying event status failed");

        if (eventStatus == CL_COMPLETE)
        {
            log_info("Could not test queue empty status because kernel event "
                     "was complete");
        }
        else if (isEmpty == CL_TRUE)
        {
            log_error(
                "Command queue could not be empty before event is complete");
            return TEST_FAIL;
        }

        error = clFinish(queue);
        test_error(error, "clFinish failed");

        error = clGetCommandQueueInfo(queue, CL_QUEUE_EMPTY, sizeof(isEmpty),
                                      &isEmpty, nullptr);
        test_error(error, "Querying command queue empty status failed");
        test_assert_error(isEmpty == CL_TRUE,
                          "Command queue is not empty after clFinish");
    }

    // Enqueue a command dependent on a user event.  This is less of a
    // real-world case, but this is a case where the queue can reliably be
    // tested that it is not empty.
    {
        clEventWrapper event = clCreateUserEvent(context, &error);
        test_error(error, "Unable to create user event");

        cl_float value = 0.0f;
        error = clEnqueueWriteBuffer(queue, buf, CL_FALSE, 0, sizeof(value),
                                     &value, 1, &event, nullptr);
        test_error(error, "Unable to enqueue write buffer");

        error = clGetCommandQueueInfo(queue, CL_QUEUE_EMPTY, sizeof(isEmpty),
                                      &isEmpty, nullptr);
        test_error(error, "Querying command queue empty status failed");
        test_assert_error(isEmpty == CL_FALSE,
                          "Command queue is empty with dependent command");

        error = clSetUserEventStatus(event, CL_COMPLETE);
        test_error(error, "Unable to set user event status");

        error = clFinish(queue);
        test_error(error, "clFinish failed");

        error = clGetCommandQueueInfo(queue, CL_QUEUE_EMPTY, sizeof(isEmpty),
                                      &isEmpty, nullptr);
        test_error(error, "Querying command queue empty status failed");
        test_assert_error(isEmpty == CL_TRUE,
                          "Command queue is not empty after clFinish");
    }

    return TEST_PASS;
}

REGISTER_TEST(queue_empty_query)
{
    cl_int error = CL_SUCCESS;

    // common test setup

    clMemWrapper buf = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                      sizeof(cl_float), nullptr, &error);
    test_error(error, "Unable to create buffer");

    clProgramWrapper program;
    clKernelWrapper kernel;

    error = create_single_kernel_helper(context, &program, &kernel, 1,
                                        &timesink_kernel, "TimeSink");
    test_error(error, "Unable to create kernel");

    error = clSetKernelArg(kernel, 0, sizeof(buf), &buf);
    test_error(error, "Unable to set kernel arg");

    // test with different command queue properties

    int result = TEST_PASS;

    std::vector<cl_command_queue_properties> test_props = {
        0, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, CL_QUEUE_PROFILING_ENABLE,
        CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE
    };

    for (auto props : test_props)
    {
        result |= test_queue_empty_helper(props, context, device, kernel, buf);
    }

    return result;
}
