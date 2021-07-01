#include "arm_compute/core/NEON/kernels/NEABMInputTransposeKernel.h"

#include "arm_compute/core/CPP/Validate.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/Size2D.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"

#include "arm_compute/core/utils/misc/ShapeCalculator.h"

#include <arm_neon.h>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <tuple>

using namespace arm_compute;
using namespace misc::shape_calculator;

namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output, const Size2D &kernel_dims, const PadStrideInfo &conv_info,
                          bool has_bias, const Size2D &dilation, unsigned int num_groups)
{
    // ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(input);
    // ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QASYMM8, DataType::F16, DataType::F32);
    // ARM_COMPUTE_RETURN_ERROR_ON(input->data_type() == DataType::QASYMM8 && has_bias);
    // ARM_COMPUTE_RETURN_ERROR_ON((dilation.x() < 1) || (dilation.y() < 1));
    // ARM_COMPUTE_RETURN_ERROR_ON_MSG(num_groups > 1, "Number of groups greater than one are not supported on NEON");

    // if(output->total_size() > 0)
    // {
    //     TensorInfo expected_output = output->clone()->set_tensor_shape(compute_im2col_conv_shape(input, kernel_dims, conv_info, has_bias, dilation, false));
    //     ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(&expected_output, output);
    //     ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    //     ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_QUANTIZATION_INFO(input, output);
    // }

    return Status{};
}

std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *output, const Size2D &kernel_dims, const PadStrideInfo &conv_info,
                                                        bool has_bias, const Size2D &dilation)
{
    const unsigned int width_idx   = get_data_layout_dimension_index(input->data_layout(), DataLayoutDimension::WIDTH);
    const unsigned int height_idx  = get_data_layout_dimension_index(input->data_layout(), DataLayoutDimension::HEIGHT);
    const unsigned int channel_idx = get_data_layout_dimension_index(input->data_layout(), DataLayoutDimension::CHANNEL);

    std::pair<unsigned int, unsigned int> convolved_dims = scaled_dimensions(input->dimension(width_idx), input->dimension(height_idx),
                                                                             kernel_dims.width, kernel_dims.height,
                                                                             conv_info, dilation);

    // Output tensor auto initialization if not yet initialized
    TensorShape output_shape{ input->tensor_shape() };



    std::pair<unsigned int, unsigned int> out_dims = scaled_dimensions(output_shape[width_idx], output_shape[height_idx], kernel_dims.width, kernel_dims.height, conv_info, dilation);
    output_shape.set(0,(out_dims.first * out_dims.second));                  //dimension 0  : the total count of dimension steps
    output_shape.set(1,  (output_shape[channel_idx]  * kernel_dims.area() + (has_bias ? 1 : 0)));                      //one convolution 
    output_shape.set(2, 1); 
    auto_init_if_empty(*output, input->clone()->set_tensor_shape(output_shape));
    //end initialization output tensor
    Window win = calculate_max_window(*input, Steps());
    win.set(width_idx, Window::Dimension(0, convolved_dims.first, 1));
    win.set(height_idx, Window::Dimension(0, convolved_dims.second, 1));
    win.set(channel_idx, Window::Dimension(0, 1, 1));

    // The NEIm2ColKernel doesn't need padding so update_window_and_padding() can be skipped
    output->set_valid_region(ValidRegion(Coordinates(), output->tensor_shape()));

    return std::make_pair(Status{}, win);
}

inline void linearize_volume_nchw(const uint8_t *const in_ptr,
                                  float                   *out_ptr,
                                  bool                 has_bias,
                                  int                  top_left_x,
                                  int                  top_left_y,
                                  int                  kernel_width,
                                  int                  kernel_height,
                                  int                  kernel_depth,
                                  int                  input_w,
                                  int                  input_h,
                                  int                  input_stride_x,
                                  int                  input_stride_y,
                                  int                  input_stride_z,
                                  int                  pad_value,
                                  int                  dilation_x,
                                  int                  dilation_y,
                                  bool              has_pads,
                                  int                  output_dimension_x)
{
    const int kernel_size2 = kernel_width * kernel_height;
    const int x_e          = top_left_x + kernel_width * dilation_x;
    const int y_e          = top_left_y + kernel_height * dilation_y;
    // Linearize volume
    int d = 0;
    // This for loop linearize a volume with 3 slices. This allows:
    // 1) to reduce the iterations of the outer for loop "d"
    // 2) to have an optimized im2col for the first convolution layer where usually we have 3 IFMs
    for(; d <= (kernel_depth - 3); d += 3)
    {
        for(int y = top_left_y; y < y_e; y += dilation_y)
        {
            if((y < 0 || y >= input_h) && has_pads)
            {
                // All the values will be the offset (will be zeros when not quantized)
                for(int x = top_left_x; x < x_e; x += dilation_x, out_ptr+=output_dimension_x)
                {
                    *(out_ptr + 0 * kernel_size2*output_dimension_x) = pad_value;
                    *(out_ptr + 1 * kernel_size2*output_dimension_x) = pad_value;
                    *(out_ptr + 2 * kernel_size2*output_dimension_x) = pad_value;
                }
            }
            else
            {
                for(int x = top_left_x; x < x_e; x += dilation_x, out_ptr+=output_dimension_x)
                {
                    if((x < 0 || x >= input_w) && has_pads)
                    {
                        *(out_ptr + 0 * kernel_size2*output_dimension_x) = pad_value;
                        *(out_ptr + 1 * kernel_size2*output_dimension_x) = pad_value;
                        *(out_ptr + 2 * kernel_size2*output_dimension_x) = pad_value;
                    }
                    else
                    {
                        *(out_ptr + 0 * kernel_size2*output_dimension_x) = *(reinterpret_cast<const float *>(in_ptr + ((d + 0) * input_stride_z + y * input_stride_y + x * input_stride_x)));
                        *(out_ptr + 1 * kernel_size2*output_dimension_x) = *(reinterpret_cast<const float *>(in_ptr + ((d + 1) * input_stride_z + y * input_stride_y + x * input_stride_x)));
                        *(out_ptr + 2 * kernel_size2*output_dimension_x) = *(reinterpret_cast<const float *>(in_ptr + ((d + 2) * input_stride_z + y * input_stride_y + x * input_stride_x)));
                    }
                }
            }
        }
        out_ptr +=( 2 * kernel_size2*output_dimension_x);
    }

    // Left over
    for(; d < kernel_depth; d++)
    {
        for(int y = top_left_y; y < y_e; y += dilation_y)
        {
            if((y < 0 || y >= input_h) && has_pads)
            {
                // All the values will be the offset (will be zeros when not quantized)
               for(int x = top_left_x; x < x_e; x += dilation_x, out_ptr+=output_dimension_x)
                {
                    *(out_ptr) = pad_value;
                }
            }
            else
            {
                for(int x = top_left_x; x < x_e; x += dilation_x, out_ptr+=output_dimension_x)
                {
                    if((x < 0 || x >= input_w) && has_pads)
                    {
                        *out_ptr = pad_value;
                    }
                    else
                    {
                        *out_ptr = *(reinterpret_cast<const float *>(in_ptr + (d * input_stride_z + y * input_stride_y + x * input_stride_x)));
                    }
                }
            }
        }
    }
    // Append 1 if the convolution layer has biases
    if(has_bias)
    {
        *out_ptr = static_cast<float>(1);
    }
}

} // namespace

NEABMInputTransposeKernel::NEABMInputTransposeKernel()
    :  _input(nullptr), _output(nullptr), _convolved_dims(), _conv_info(), _kernel_width(0), _kernel_height(0), _has_bias(false), _dilation(1U, 1U)
{
}

void NEABMInputTransposeKernel::configure(const ITensor *input, ITensor *output, const Size2D &kernel_dims, const PadStrideInfo &conv_info,
                               bool has_bias, const Size2D &dilation, unsigned int num_groups)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), output->info(), kernel_dims, conv_info, has_bias, dilation, num_groups));
    ARM_COMPUTE_UNUSED(num_groups);

    const DataLayout   data_layout = input->info()->data_layout();
    const unsigned int width_idx   = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const unsigned int height_idx  = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);

    _input          = input;
    _output         = output;
    _conv_info      = conv_info;
    _kernel_width   = kernel_dims.width;
    _kernel_height  = kernel_dims.height;
    _dilation       = dilation;
    _convolved_dims = scaled_dimensions(input->info()->dimension(width_idx), input->info()->dimension(height_idx),
                                        _kernel_width, _kernel_height,
                                        _conv_info, _dilation);
    _has_bias = has_bias;
    // Configure kernel window
    auto win_config = validate_and_configure_window(input->info(), output->info(), kernel_dims, conv_info, has_bias, dilation);
    //std::cout<<_output->info()->dimension(0)<<' '<<_output->info()->dimension(1)<<' '<<_output->info()->dimension(2)<<std::endl;
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    INEKernel::configure(win_config.second);
}

Status NEABMInputTransposeKernel::validate(const ITensorInfo *input, const ITensorInfo *output, const Size2D &kernel_dims, const PadStrideInfo &conv_info,
                                bool has_bias, const Size2D &dilation, unsigned int num_groups)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output, kernel_dims, conv_info, has_bias, dilation, num_groups));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(input->clone().get(), output->clone().get(), kernel_dims, conv_info, has_bias, dilation).first);
    return Status{};
}

void NEABMInputTransposeKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    const DataLayout   data_layout = _input->info()->data_layout();
    const unsigned int width_idx   = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const unsigned int height_idx  = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);
    const unsigned int channel_idx = get_data_layout_dimension_index(data_layout, DataLayoutDimension::CHANNEL);

    const int input_w        = _input->info()->dimension(width_idx);
    const int input_h        = _input->info()->dimension(height_idx);
    const int input_c        = _input->info()->dimension(channel_idx);
    const int input_stride_x = _input->info()->strides_in_bytes().x();
    const int input_stride_y = _input->info()->strides_in_bytes().y();
    const int input_stride_z = _input->info()->strides_in_bytes().z();
    const int pad_left       = _conv_info.pad_left();
    const int pad_top        = _conv_info.pad_top();
    const int stride_x       = _conv_info.stride().first;
    const int stride_y       = _conv_info.stride().second;
    const int pad_value      = 0;
    bool has_pads=(_conv_info.has_padding())?true:false;
    const int output_dimension_x=_output->info()->dimension(0);
    
    Window window_in_out(window);
    //全部需要通过绝对坐标访问！
    // The first three dimensions of the input and output are increased by the inner loops
    window_in_out.set(Window::DimX, Window::Dimension(0, 0, 0));
    window_in_out.set(Window::DimY, Window::Dimension(0, 0, 0));
    window_in_out.set(Window::DimZ, Window::Dimension(0, 0, 0));

    // Create iterators
    Iterator in(_input, window_in_out);
    Iterator out(_output, window_in_out);

    execute_window_loop(window, [&](const Coordinates & id)
    {
        const int start_w = id[width_idx] * stride_x - pad_left;
        const int start_h = id[height_idx] * stride_y - pad_top;

        // Get pointers
        const uint8_t *const input_ptr  = in.ptr();
        float                 *output_ptr = reinterpret_cast<float *>(_output->buffer() + (id[width_idx] + id[height_idx] * _convolved_dims.first) * 4);

        linearize_volume_nchw(input_ptr,
                                               output_ptr,
                                               _has_bias,
                                               start_w,
                                               start_h,
                                               _kernel_width,
                                               _kernel_height,
                                               input_c,
                                               input_w,
                                               input_h,
                                               input_stride_x,
                                               input_stride_y,
                                               input_stride_z,
                                               pad_value,
                                               _dilation.x(),
                                               _dilation.y(),
                                               has_pads,
                                               output_dimension_x);
    },
    in, out);
}


