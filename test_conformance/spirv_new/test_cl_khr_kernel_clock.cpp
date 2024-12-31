//
// Copyright (c) 2024 The Khronos Group Inc.
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
#include "types.hpp"

#include "testBase.h"

#include <array>

#if !defined(cl_khr_kernel_clock)
typedef cl_bitfield cl_device_kernel_clock_capabilities_khr;
#define CL_DEVICE_KERNEL_CLOCK_CAPABILITIES_KHR 0x1076
#define CL_DEVICE_KERNEL_CLOCK_SCOPE_DEVICE_KHR (1 << 0)
#define CL_DEVICE_KERNEL_CLOCK_SCOPE_WORK_GROUP_KHR (1 << 1)
#define CL_DEVICE_KERNEL_CLOCK_SCOPE_SUB_GROUP_KHR (1 << 2)
#endif

static int test_kernel_clock(cl_device_id device, cl_context context,
                             cl_command_queue queue,
                             cl_device_kernel_clock_capabilities_khr clockType)
{
    cl_int error = CL_SUCCESS;

#if 0
    cl_device_kernel_clock_capabilities_khr clockCaps;
    error = clGetDeviceInfo(device, CL_DEVICE_KERNEL_CLOCK_CAPABILITIES_KHR,
                            sizeof(cl_device_kernel_clock_capabilities_khr),
                            &clockCaps, NULL);
    test_error(error,
               "Unable to query "
               "CL_DEVICE_KERNEL_CLOCK_CAPABILITIES_KHR");
    if (!(clockCaps & clockType))
    {
        return TEST_SKIPPED_ITSELF;
    }
#endif

    std::array<const char*, 2> testNames;
    switch (clockType)
    {
        case CL_DEVICE_KERNEL_CLOCK_SCOPE_DEVICE_KHR:
            testNames[0] = "clock_read_device";
            testNames[1] = "clock_read_device_hilo";
            break;
        case CL_DEVICE_KERNEL_CLOCK_SCOPE_WORK_GROUP_KHR:
            testNames[0] = "clock_read_work_group";
            testNames[1] = "clock_read_work_group_hilo";
            break;
        case CL_DEVICE_KERNEL_CLOCK_SCOPE_SUB_GROUP_KHR:
            testNames[0] = "clock_read_sub_group";
            testNames[1] = "clock_read_sub_group_hilo";
            break;
        default: return TEST_FAIL;
    }

    for (size_t i = 0; i < testNames.size(); i++)
    {
        if (i == 0 && !gHasLong)
        {
            continue;
        }

        std::array<cl_uint, 4> buf({ 0, 0, 1024, 0 });
        clMemWrapper dst =
            clCreateBuffer(context, CL_MEM_COPY_HOST_PTR,
                           buf.size() * sizeof(cl_uint), buf.data(), &error);
        test_error(error, "Unable to create destination buffer");

        clProgramWrapper prog;
        error = get_program_with_il(prog, device, context, testNames[i]);
        test_error(error, "Unable to build SPIR-V program");

        clKernelWrapper kernel = clCreateKernel(prog, testNames[i], &error);
        test_error(error, "Unable to create SPIR-V kernel");

        error |= clSetKernelArg(kernel, 0, sizeof(dst), &dst);
        test_error(error, "Unable to set kernel arguments");

        size_t global = 1;
        error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, NULL, 0,
                                       NULL, NULL);
        test_error(error, "Unable to enqueue kernel");

        error = clEnqueueReadBuffer(queue, dst, CL_TRUE, 0,
                                    buf.size() * sizeof(cl_uint), buf.data(), 0,
                                    NULL, NULL);
        test_error(error, "Unable to read destination buffer");

        if (buf[3] != 1)
        {
            log_error("The clock did not increase!\n");
            return TEST_FAIL;
        }
    }

    return TEST_PASS;
}

REGISTER_TEST(kernel_clock_sub_group)
{
    if (false && !is_extension_available(device, "cl_khr_kernel_clock"))
    {
        log_info("cl_khr_kernel_clock is not supported; skipping test.\n");
        return TEST_SKIPPED_ITSELF;
    }

    return test_kernel_clock(device, context, queue,
                             CL_DEVICE_KERNEL_CLOCK_SCOPE_SUB_GROUP_KHR);
}
