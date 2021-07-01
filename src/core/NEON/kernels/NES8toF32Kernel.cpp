/*
 * Copyright (c) 2017-2019 ARM Limited.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include "arm_compute/core/NEON/kernels/NES8toF32Kernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"

#include <arm_neon.h>
#include <cstddef>
#include <cstdint>

using namespace arm_compute;
using namespace misc::shape_calculator;

namespace
{
// Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output)
// {
//     // //Note: ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(input) is not needed here as this kernel doesn't use NEON FP16 instructions.
//     // ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8, DataType::S8, DataType::QASYMM8,
//     //                                                      DataType::U16, DataType::S16,
//     //                                                      DataType::U32, DataType::S32,
//     //                                                      DataType::F16, DataType::F32);

//     // // Validate configured output
//     // if(output->total_size() != 0)
//     // {
//     //     ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(output->tensor_shape(), compute_col2im_shape(*input, convolved_dims, false));
//     //     ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
//     //     ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_QUANTIZATION_INFO(input, output);
//     // }

//     return Status{};
// }

std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *output)
{
    // Output auto inizialitation if not yet initialized
        // TensorShape shape=input->tensor_shape();
        // TensorInfo info =(shape,1,DataType::F32);
        // _output->allocator
        output->set_data_type(DataType::F32);
        output->set_num_channels(input->num_channels());
        output->set_tensor_shape(input->tensor_shape());
        output->set_quantization_info(input->quantization_info());
        output->set_data_layout(input->data_layout());
    //auto_init_if_empty(*output, input->clone()->set_tensor_shape(compute_col2im_shape(*input, convolved_dims, false)));

    // Configure kernel window
    Window win = calculate_max_window(*input, Steps());

    // The NECol2ImKernel doesn't need padding so update_window_and_padding() can be skipped
    win.set(Window::DimX, Window::Dimension(0, input->dimension(0),input->dimension(0)));   
    win.set(Window::DimY, Window::Dimension(0, input->dimension(1),input->dimension(1)));   
    win.set(Window::DimZ, Window::Dimension(0, input->dimension(2),1));   
    
    
    return std::make_pair(Status{}, win);
}
} // namespace


NES8toF32Kernel::NES8toF32Kernel()
    :  _input(nullptr), _output(nullptr)
{
}

void NES8toF32Kernel::configure(const ITensor *input, ITensor *output)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    //ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), output->info()));

    _input          = input;
    _output         = output;
   
    auto win_config = validate_and_configure_window(input->info(), output->info());
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    INEKernel::configure(win_config.second);
}

Status NES8toF32Kernel::validate(const ITensorInfo *input, const ITensorInfo *output)
{
    // ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output, convolved_dims));
    // ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(input->clone().get(), output->clone().get(), convolved_dims).first);
    return Status{};
}

void NES8toF32Kernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    Iterator in(_input, window);
    Iterator out(_output, window);

    execute_window_loop(window, [&](const Coordinates & id)
    {
        uint8_t *in_ptr=in.ptr();
        uint8_t *out_ptr=out.ptr();
        int8_t *in_addr;float *out_addr;
        unsigned int x_end=_input->info()->dimension(0);
        unsigned int y_end=_input->info()->dimension(1);
        for(unsigned int i=0;i<y_end;i++)
        {
            in_addr=reinterpret_cast<int8_t*>(in_ptr+i*x_end);
            out_addr=reinterpret_cast<float*>(out_ptr+i*x_end*4);
            for(unsigned int j=0;j<x_end;j++)
            {
                (*out_addr)=(float)(*in_addr);
                out_addr++;
                in_addr++;
            }
        }
       // std::cout<<"1"<<std::endl;
    },
    in, out);
}

