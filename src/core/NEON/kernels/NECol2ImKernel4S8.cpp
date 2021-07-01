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
#include "arm_compute/core/NEON/kernels/NECol2ImKernel4S8.h"

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
inline Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output, const Size2D &convolved_dims)
{
    // //Note: ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(input) is not needed here as this kernel doesn't use NEON FP16 instructions.
    // ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8, DataType::S8, DataType::QASYMM8,
    //                                                      DataType::U16, DataType::S16,
    //                                                      DataType::U32, DataType::S32,
    //                                                      DataType::F16, DataType::F32);

    // // Validate configured output
    // if(output->total_size() != 0)
    // {
    //     ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(output->tensor_shape(), compute_col2im_shape(*input, convolved_dims, false));
    //     ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    //     ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_QUANTIZATION_INFO(input, output);
    // }

    return Status{};
}

inline std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *output, const Size2D &convolved_dims)
{
    const DataLayout data_layout = input->data_layout();
    const int        width_idx   = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const int        height_idx  = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);
    const int        channel_idx = get_data_layout_dimension_index(data_layout, DataLayoutDimension::CHANNEL);
    TensorShape col2im_shape{ input->tensor_shape() };
    col2im_shape.set(width_idx, convolved_dims.width);
    col2im_shape.set(height_idx, convolved_dims.height);
    col2im_shape.set(channel_idx, input->dimension(1) );
    // Output auto inizialitation if not yet initialized
    auto_init_if_empty(*output, input->clone()->set_tensor_shape(col2im_shape));

    // Configure kernel window
    Window win = calculate_max_window(*input, Steps());
    win.set(Window::DimX, Window::Dimension(0, input->dimension(0),input->dimension(0)));   
    win.set(Window::DimY, Window::Dimension(0, input->dimension(1),1));   
    win.set(Window::DimZ, Window::Dimension(0, input->dimension(2),input->dimension(2)));   

    // The NECol2ImKernel doesn't need padding so update_window_and_padding() can be skipped
    Coordinates coord;
    coord.set_num_dimensions(output->num_dimensions());
    output->set_valid_region(ValidRegion(coord, output->tensor_shape()));

    return std::make_pair(Status{}, win);
}
} // namespace


NECol2ImKernel4S8::NECol2ImKernel4S8()
    : _input(nullptr), _output(nullptr), _convolved_dims()
{
}

void NECol2ImKernel4S8::configure(const ITensor *input, ITensor *output, const Size2D &convolved_dims)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), output->info(), convolved_dims));

    _input          = input;
    _output         = output;
    _convolved_dims = convolved_dims;

    // Configure kernel window
    auto win_config = validate_and_configure_window(input->info(), output->info(), convolved_dims);
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    INEKernel::configure(win_config.second);
}

Status NECol2ImKernel4S8::validate(const ITensorInfo *input, const ITensorInfo *output, const Size2D &convolved_dims)
{
    //ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output, convolved_dims));
    //ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(input->clone().get(), output->clone().get(), convolved_dims).first);
    return Status{};
}

void NECol2ImKernel4S8::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    //  const int output_stride_x = _output->info()->strides_in_bytes().x();
    // const int output_stride_y = _output->info()->strides_in_bytes().y();
    // const int output_stride_z = _output->info()->strides_in_bytes().z();

    Window window_out(window);
    window_out.set(Window::DimX, Window::Dimension(0, 0, 0));
    window_out.set(Window::DimY, Window::Dimension(0, 0, 0));
    window_out.set(Window::DimZ, Window::Dimension(0, 0, 0));

    // Create iterators
    Iterator in(_input, window);
    Iterator out(_output, window_out);

    execute_window_loop(window, [&](const Coordinates & id)
    {
        size_t y_dimension=id[1];
        int8_t *out_addr;size_t out_offset=y_dimension*_output->info()->strides_in_bytes().z();
        int8_t *in_addr;size_t in_offset=_input->info()->offset_element_in_bytes(Coordinates(0,y_dimension,0));
        unsigned int nums=_input->info()->dimension(0)/16;
        for(unsigned int i=0;i<16*nums;i+=16)
        {
            
            in_addr=reinterpret_cast<int8_t*>(_input->buffer()+(unsigned int)in_offset+i);
            out_addr=reinterpret_cast<int8_t*>( _output->buffer()+(unsigned int)out_offset+i);
            vst1q_s8(out_addr,vld1q_s8(in_addr));
        }
        for(unsigned int i=nums*16;i<_input->info()->dimension(0);i++)
        {
            in_addr=reinterpret_cast<int8_t*>(_input->buffer()+(unsigned int)in_offset+i);
            out_addr=reinterpret_cast<int8_t*>( _output->buffer()+(unsigned int)out_offset+i);
            (*out_addr)=(*in_addr);
        }
        // const int widx = id.x();
        // //以byte为单位记录索引
        // const int idx  = id.y() * output_stride_z + (widx / _convolved_dims.width) * output_stride_y + (widx % _convolved_dims.width) * output_stride_x;

        // *(reinterpret_cast<int8_t *>(out.ptr() + idx)) = *(reinterpret_cast<const int8_t *>(in.ptr()));
    },
    in, out);
   
}

