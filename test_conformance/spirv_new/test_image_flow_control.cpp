//
// Copyright (c) 2016-2023 The Khronos Group Inc.
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

#include <sstream>
#include <string>

REGISTER_TEST(image_flow_control)
{
    PASSIVE_REQUIRE_IMAGE_SUPPORT(device);

    cl_image_format image_format = {};
    image_format.image_channel_order = CL_RGBA;
    image_format.image_channel_data_type = CL_SIGNED_INT8;

    cl_int error = CL_SUCCESS;

    std::vector<cl_uchar> imgData({ 0x11, 0x22, 0x33, 0x44 });

    clProgramWrapper prog;
    error = get_program_with_il(prog, device, context, "image_flow_control");
    SPIRV_CHECK_ERROR(error, "Failed to compile spv program");

    clKernelWrapper kernel = clCreateKernel(prog, "image_flow_control_test", &error);
    SPIRV_CHECK_ERROR(error, "Failed to create spv kernel");

    cl_uint h_dst = 0;
    clMemWrapper dst =
        clCreateBuffer(context, CL_MEM_COPY_HOST_PTR,
                       sizeof(cl_uint), &h_dst, &error);
    SPIRV_CHECK_ERROR(error, "Failed to create dst buffer");

    clMemWrapper src =
        clCreateImage2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        &image_format, 1, 1, 0, imgData.data(), &error);
    SPIRV_CHECK_ERROR(error, "Failed to create src image");

    error |= clSetKernelArg(kernel, 0, sizeof(dst), &dst);
    error |= clSetKernelArg(kernel, 1, sizeof(src), &src);
    SPIRV_CHECK_ERROR(error, "Failed to set kernel args");

    size_t global = 1;
    error = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, NULL, 0,
                                   NULL, NULL);
    SPIRV_CHECK_ERROR(error, "Failed to enqueue kernel");

    error = clEnqueueReadBuffer(queue, dst, CL_TRUE, 0,
                                sizeof(cl_uint), &h_dst, 0,
                                NULL, NULL);
    SPIRV_CHECK_ERROR(error, "Unable to read destination buffer");

    const cl_uint expected = 47;
    if (h_dst != expected)
    {
        log_error("Mismatch! Got: %u, Wanted: %u\n",
                  h_dst, expected);
        return TEST_FAIL;
    }

    return TEST_PASS;
}
