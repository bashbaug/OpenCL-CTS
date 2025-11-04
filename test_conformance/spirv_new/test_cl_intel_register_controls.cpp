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
#include "spirvInfo.hpp"

// TODO: Update this if/when the extension is added to the upstream headers.
#if !defined(CL_KERNEL_REGISTER_COUNT_INTEL)
#define CL_KERNEL_REGISTER_COUNT_INTEL 0x425B
#endif

static int test_build_option(cl_device_id device, cl_context context)
{
    const char* kernel_string = "kernel void test(global int* dst) {}";

    std::vector<const char*> build_options;
    build_options.push_back("-cl-intel-256-GRF-per-thread");
    build_options.push_back("-cl-intel-enable-auto-large-GRF-mode");

    log_info("  testing building from source with a build option\n");

    for (auto build_option : build_options)
    {
        log_info("    testing build option: %s\n", build_option);

        clProgramWrapper program;
        clKernelWrapper kernel;
        cl_int error =
            create_single_kernel_helper(context, &program, &kernel, 1,
                                        &kernel_string, "test", build_option);
        test_error(error, "unable to create test kernel");

        cl_uint count = 0;
        error = clGetKernelWorkGroupInfo(kernel, device,
                                         CL_KERNEL_REGISTER_COUNT_INTEL,
                                         sizeof(count), &count, nullptr);
        test_error(error, "unable to query register count");

        log_info("      register count is: %u\n", count);
    }

    return TEST_PASS;
}

static int run_spirv_case(cl_device_id device, cl_context context,
                          const char* spv_file_name,
                          const spec_const& sc = spec_const())
{
    clProgramWrapper prog;
    cl_int error =
        get_program_with_il(prog, device, context, spv_file_name, sc);
    test_error(error, "unable to create program from SPIR-V");

    clKernelWrapper kernel = clCreateKernel(prog, "test", &error);
    test_error(error, "unable to create test kernel");

    cl_uint count = 0;
    error =
        clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_REGISTER_COUNT_INTEL,
                                 sizeof(count), &count, nullptr);
    test_error(error, "unable to query register count");

    log_info("      register count is: %u\n", count);
    return TEST_PASS;
}

static int test_spirv(cl_device_id device, cl_context context)
{
    cl_int result = TEST_PASS;

    log_info("  testing building from SPIR-V\n");

    log_info("    testing SPIR-V file with literal max registers\n");
    result |= run_spirv_case(device, context,
                             "spv1.2/ext_cl_intel_register_controls_literal");

    log_info("    testing SPIR-V file with named auto max registers\n");
    result |= run_spirv_case(
        device, context, "spv1.2/ext_cl_intel_register_controls_named_auto");

    for (cl_uint regs = 32; regs <= 1024; regs *= 2)
    {
        const spec_const regs_spec_const = spec_const(101, sizeof(regs), &regs);

        log_info(
            "    testing SPIR-V file with spec constant max registers: %u\n",
            regs);
        result |= run_spirv_case(
            device, context, "spv1.2/ext_cl_intel_register_controls_specconst",
            regs_spec_const);
    }

    return TEST_PASS;
}

REGISTER_TEST(intel_register_controls)
{
#if 0
    REQUIRE_EXTENSION("cl_intel_register_controls");
#else
    cl_device_type type = 0;
    clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(type), &type, NULL);
    cl_uint vendor_id = 0;
    clGetDeviceInfo(device, CL_DEVICE_VENDOR_ID, sizeof(vendor_id), &vendor_id,
                    NULL);
    if ((type & CL_DEVICE_TYPE_GPU) == 0 || vendor_id != 0x8086)
    {
        return TEST_SKIPPED_ITSELF;
    }
#endif

    if (!is_spirv_version_supported(device, "SPIR-V_1.2"))
    {
        log_info("SPIR-V 1.2 not supported; skipping tests.\n");
        return TEST_SKIPPED_ITSELF;
    }

    int result = TEST_PASS;

    result |= test_build_option(device, context);
    result |= test_spirv(device, context);

    return result;
}
