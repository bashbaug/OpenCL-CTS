// Copyright (c) 2026 The Khronos Group Inc.
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
#include "harness/typeWrappers.h"

#include <algorithm>
#include <vector>

// TODO: clean this up once support is in the upstream headers.
#if !defined(cl_intel_concurrent_dispatch)

#define cl_intel_concurrent_dispatch 1
#define CL_INTEL_CONCURRENT_DISPATCH_EXTENSION_NAME                            \
    "cl_intel_concurrent_dispatch"

#define CL_INTEL_CONCURRENT_DISPATCH_EXTENSION_VERSION CL_MAKE_VERSION(1, 0, 0)

/* cl_kernel_exec_info */
#define CL_KERNEL_EXEC_INFO_DISPATCH_TYPE_INTEL 0x4257

typedef cl_uint cl_kernel_exec_info_dispatch_type_intel;

/* cl_kernel_exec_info_dispatch_type_intel */
#define CL_KERNEL_EXEC_INFO_DISPATCH_TYPE_DEFAULT_INTEL 0
#define CL_KERNEL_EXEC_INFO_DISPATCH_TYPE_CONCURRENT_INTEL 1

typedef cl_int CL_API_CALL clGetKernelMaxConcurrentWorkGroupCountINTEL_t(
    cl_command_queue command_queue, cl_kernel kernel, cl_uint work_dim,
    const size_t* global_work_offset, const size_t* local_work_size,
    size_t* max_work_group_count);

typedef clGetKernelMaxConcurrentWorkGroupCountINTEL_t*
    clGetKernelMaxConcurrentWorkGroupCountINTEL_fn;

#if !defined(CL_NO_NON_ICD_DISPATCH_EXTENSION_PROTOTYPES)

extern CL_API_ENTRY cl_int CL_API_CALL
clGetKernelMaxConcurrentWorkGroupCountINTEL(cl_command_queue command_queue,
                                            cl_kernel kernel, cl_uint work_dim,
                                            const size_t* global_work_offset,
                                            const size_t* local_work_size,
                                            size_t* max_work_group_count);

#endif /* !defined(CL_NO_NON_ICD_DISPATCH_EXTENSION_PROTOTYPES) */

#endif // !defined(cl_intel_concurrent_dispatch)

// After running this kernel dst should contain:
// Indices 0 through global_size: either:
//      global_size + 1 (concurrent dispatch), or
//      a number between 2 and global_size + 1 (no concurrent dispatch)
// Index global_size + 0: global_size
// Index global_size + 1: either:
//      1 (concurrent dispatch), or
//      0 (no concurrent dispatch)
static const char* kernelString = R"CLC(
kernel void DeviceBarrierTest(global uint* dst)
{
    const size_t gws = get_global_size(0);
    atomic_add(&dst[gws + 0], 1);
    if (intel_is_device_barrier_valid()) {
        atomic_cmpxchg(&dst[gws + 1], 0, 1);        // record that the device barrier is valid
        intel_device_barrier(CLK_GLOBAL_MEM_FENCE); // device barrier with no scope
        intel_device_barrier(CLK_GLOBAL_MEM_FENCE, memory_scope_device);    // with scope
    }
    const uint id = get_global_id(0);
    dst[id] = dst[gws] + 1;
}
)CLC";

// Other tests we could do:
// Test with and without a global offset.
// Test 2D and 3D dispatches.

static int concurrent_dispatch_helper(cl_device_id device, cl_context context,
                                      cl_command_queue queue, int dispatch_type)
{
    cl_int error = CL_SUCCESS;

    clProgramWrapper program;
    clKernelWrapper kernel;

    error = create_single_kernel_helper(context, &program, &kernel, 1,
                                        &kernelString, "DeviceBarrierTest");
    test_error_fail(error, "Failed to create kernel");

    // Set the kernel argument to nullptr initially.
    error = clSetKernelArg(kernel, 0, sizeof(cl_mem), nullptr);
    test_error_fail(error, "Initial clSetKernelArg to nullptr failed");

    // Figure out the maximum local work-group size for this kernel.
    size_t lws = 0;
    error = clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE,
                                     sizeof(lws), &lws, nullptr);
    test_error_fail(error,
                    "Unable to query maximum local work-group size for kernel");

    // Clamp to 256 work-items.
    lws = std::min<size_t>(lws, 256);

    size_t gws = lws * 16;

    // For concurrent dispatch, figure out the number of work-groups that can
    // execute concurrently.
    if (dispatch_type == CL_KERNEL_EXEC_INFO_DISPATCH_TYPE_CONCURRENT_INTEL)
    {
        cl_platform_id platform = nullptr;
        error = clGetDeviceInfo(device, CL_DEVICE_PLATFORM, sizeof(platform),
                                &platform, nullptr);
        test_error_fail(error, "Unable to query platform from device");

        auto clGetKernelMaxConcurrentWorkGroupCountINTEL =
            (clGetKernelMaxConcurrentWorkGroupCountINTEL_fn)
                clGetExtensionFunctionAddressForPlatform(
                    platform, "clGetKernelMaxConcurrentWorkGroupCountINTEL");
        test_assert_error(clGetKernelMaxConcurrentWorkGroupCountINTEL
                              != nullptr,
                          "Couldn't get function pointer for "
                          "clGetKernelMaxConcurrentWorkGroupCountINTEL");

        size_t wgCount = 0;
        error = clGetKernelMaxConcurrentWorkGroupCountINTEL(
            queue, kernel, 1, nullptr, &lws, &wgCount);
        test_error_fail(error,
                        "clGetKernelMaxConcurrentWorkGroupCountINTEL failed");

        // Clamp to 1024 work-groups.
        wgCount = std::min<size_t>(wgCount, 1024);

        gws = lws * wgCount;

        cl_uint type = dispatch_type;
        error =
            clSetKernelExecInfo(kernel, CL_KERNEL_EXEC_INFO_DISPATCH_TYPE_INTEL,
                                sizeof(type), &type);
        test_error_fail(
            error,
            "clSetKernelExecInfo to set "
            "CL_KERNEL_EXEC_INFO_DISPATCH_TYPE_CONCURRENT_INTEL failed");
    }
    else if (dispatch_type == CL_KERNEL_EXEC_INFO_DISPATCH_TYPE_DEFAULT_INTEL)
    {
        cl_uint type = dispatch_type;
        error =
            clSetKernelExecInfo(kernel, CL_KERNEL_EXEC_INFO_DISPATCH_TYPE_INTEL,
                                sizeof(type), &type);
        test_error_fail(
            error,
            "clSetKernelExecInfo to set "
            "CL_KERNEL_EXEC_INFO_DISPATCH_TYPE_DEFAULT_INTEL failed");
    }

    clMemWrapper buf =
        clCreateBuffer(context, CL_MEM_READ_WRITE, (gws + 2) * sizeof(cl_uint),
                       nullptr, &error);
    test_error_fail(error, "clCreateBuffer for dst buffer failed");

    const cl_uint zero = 0;
    error =
        clEnqueueFillBuffer(queue, buf, &zero, sizeof(zero), 0,
                            (gws + 2) * sizeof(cl_uint), 0, nullptr, nullptr);
    test_error_fail(error,
                    "clEnqueueFillBuffer to initialize dst buffer failed");

    error = clSetKernelArg(kernel, 0, sizeof(buf), &buf);
    test_error_fail(error, "clSetKernelArg with dst buffer failed");

    error = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &gws, &lws, 0,
                                   nullptr, nullptr);
    test_error_fail(error, "clEnqueueNDRangeKernel failed");

    std::vector<cl_uint> check(gws + 2);
    error = clEnqueueReadBuffer(queue, buf, CL_BLOCKING, 0,
                                check.size() * sizeof(cl_uint), check.data(), 0,
                                nullptr, nullptr);
    test_error_fail(error, "clEnqueueReadBuffer failed");

    // Check total:
    if (check[gws] != gws)
    {
        test_fail("Unexpected total: expected %zu, got %u\n", gws, check[gws]);
    }

    if (dispatch_type == CL_KERNEL_EXEC_INFO_DISPATCH_TYPE_CONCURRENT_INTEL)
    {
        // Check per-work-item results:
        for (size_t i = 0; i < gws; i++)
        {
            if (check[i] != gws + 1)
            {
                test_fail(
                    "Unexpected result at index %zu: expected %zu, got %u\n", i,
                    gws + 1, check[i]);
            }
        }

        // Check device barrier valid:
        if (check[gws + 1] != 1)
        {
            test_fail("Unexpected result for device barrier valid: got %u\n",
                      check[gws]);
        }
    }
    else
    {
        // Check per-work-item results:
        for (size_t i = 0; i < gws; i++)
        {
            if (check[i] > gws + 1)
            {
                test_fail(
                    "Unexpected result at index %zu: expected %zu, got %u\n", i,
                    gws + 1, check[i]);
            }
        }

        // Check device barrier valid:
        if (check[gws + 1] != 0)
        {
            test_fail("Unexpected result for device barrier valid: got %u\n",
                      check[gws]);
        }
    }

    return TEST_PASS;
};

REGISTER_TEST(dispatch_unset)
{
    REQUIRE_EXTENSION(CL_INTEL_CONCURRENT_DISPATCH_EXTENSION_NAME);
    return concurrent_dispatch_helper(device, context, queue, -1);
}

REGISTER_TEST(dispatch_default)
{
    REQUIRE_EXTENSION(CL_INTEL_CONCURRENT_DISPATCH_EXTENSION_NAME);
    return concurrent_dispatch_helper(
        device, context, queue,
        CL_KERNEL_EXEC_INFO_DISPATCH_TYPE_DEFAULT_INTEL);
}

REGISTER_TEST(dispatch_concurrent)
{
    REQUIRE_EXTENSION(CL_INTEL_CONCURRENT_DISPATCH_EXTENSION_NAME);
    return concurrent_dispatch_helper(
        device, context, queue,
        CL_KERNEL_EXEC_INFO_DISPATCH_TYPE_CONCURRENT_INTEL);
}
