/*
 *  * Copyright (c) 2017-2019 ARM Limited.
 *   *
 *    * SPDX-License-Identifier: MIT
 *     *
 *      * Permission is hereby granted, free of charge, to any person obtaining a copy
 *       * of this software and associated documentation files (the "Software"), to
 *        * deal in the Software without restriction, including without limitation the
 *         * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 *          * sell copies of the Software, and to permit persons to whom the Software is
 *           * furnished to do so, subject to the following conditions:
 *            *
 *             * The above copyright notice and this permission notice shall be included in all
 *              * copies or substantial portions of the Software.
 *               *
 *                * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *                 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *                  * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *                   * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *                    * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *                     * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *                      * SOFTWARE.
 *                       */
#include "arm_compute/core/NEON/kernels/NEMulAddTestKernel.h"

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
	std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input_a,  ITensorInfo *input_b,  ITensorInfo *output)
	{
		    output->set_data_type(input_a->data_type());
		        output->set_num_channels(input_a->num_channels());
			    output->set_tensor_shape(input_a->tensor_shape());
			        output->set_quantization_info(input_a->quantization_info());
				    output->set_data_layout(input_a->data_layout());

				        Window win = calculate_max_window(*input_a, Steps());

					    win.set(Window::DimX, Window::Dimension(0, input_a->dimension(0)-2,1));   
					        win.set(Window::DimY, Window::Dimension(0, input_a->dimension(1),1));   
						    win.set(Window::DimZ, Window::Dimension(0, input_a->dimension(2),1));   
						        
						        return std::make_pair(Status{}, win);
	}
} 


NEMulAddTestKernel::NEMulAddTestKernel()
	    :  _input_a(nullptr),  _input_b(nullptr), _output(nullptr)
{
}

void NEMulAddTestKernel::configure(const ITensor *input_a,  const ITensor *input_b,  ITensor *output)
{
	    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

	        _input_a=input_a;
		    _input_b=input_b;
		        _output=output;

			    auto win_config = validate_and_configure_window(input_a->info(), input_b->info(), output->info());
			        ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
				    INEKernel::configure(win_config.second);
}

Status NEMulAddTestKernel::validate(const ITensorInfo *input_a,  const ITensorInfo *input_b,  const ITensorInfo *output)
{
	    return Status{};
}

void NEMulAddTestKernel::run(const Window &window, const ThreadInfo &info)
{
	    ARM_COMPUTE_UNUSED(info);
	        ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
		    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

		        Window win(window);
			    win.set(Window::DimX, Window::Dimension(0, 0, 0));
			        win.set(Window::DimY, Window::Dimension(0, 0, 0));
				    win.set(Window::DimZ, Window::Dimension(0, 0, 0));
				        Iterator in_a(_input_a, win);
					    Iterator in_b(_input_b, win);
					        Iterator out(_output, win);

						    execute_window_loop(window, [&](const Coordinates & id)
								        {
									        size_t x=id[0];size_t y=id[1]; size_t z=id[2];
										        size_t offset=_input_a->info()->offset_element_in_bytes(Coordinates(x,y,z));
											        int16_t *in_a_addr=reinterpret_cast<int16_t*>(_input_a->buffer()+offset);
												        int16_t *in_b_addr=reinterpret_cast<int16_t*>(_input_b->buffer()+offset);
													        int16_t *out_addr=reinterpret_cast<int16_t*>(_output->buffer()+offset);
														        /*两次读数据，一次写数据*/
														        short a=(*in_a_addr);
															        short b=(*in_b_addr);
																        /*short res=a*b;*/
																        (*out_addr)=a+b;
																	    },
																	        in_a,  in_b, out);
}

/*
 * namespace
 * {
 * std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input_a,  ITensorInfo *input_b,  ITensorInfo *output)
 * {
 *     output->set_data_type(input_a->data_type());
 *         output->set_num_channels(input_a->num_channels());
 *             output->set_tensor_shape(input_a->tensor_shape());
 *                 output->set_quantization_info(input_a->quantization_info());
 *                     output->set_data_layout(input_a->data_layout());
 *
 *                         Window win = calculate_max_window(*input_a, Steps());
 *
 *                             win.set(Window::DimX, Window::Dimension(0, input_a->dimension(0),8));   
 *                                 win.set(Window::DimY, Window::Dimension(0, input_a->dimension(1),1));   
 *                                     win.set(Window::DimZ, Window::Dimension(0, input_a->dimension(2),1));   
 *                                         
 *                                             return std::make_pair(Status{}, win);
 *                                             }
 *                                             } 
 *
 *
 *                                             NEMulAddTestKernel::NEMulAddTestKernel()
 *                                                 :  _input_a(nullptr),  _input_b(nullptr), _output(nullptr)
 *                                                 {
 *                                                 }
 *
 *                                                 void NEMulAddTestKernel::configure(const ITensor *input_a,  const ITensor *input_b,  ITensor *output)
 *                                                 {
 *                                                     ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
 *
 *                                                         _input_a=input_a;
 *                                                             _input_b=input_b;
 *                                                                 _output=output;
 *
 *                                                                     auto win_config = validate_and_configure_window(input_a->info(), input_b->info(), output->info());
 *                                                                         ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
 *                                                                             INEKernel::configure(win_config.second);
 *                                                                             }
 *
 *                                                                             Status NEMulAddTestKernel::validate(const ITensorInfo *input_a,  const ITensorInfo *input_b,  const ITensorInfo *output)
 *                                                                             {
 *                                                                                 return Status{};
 *                                                                                 }
 *
 *                                                                                 void NEMulAddTestKernel::run(const Window &window, const ThreadInfo &info)
 *                                                                                 {
 *                                                                                     ARM_COMPUTE_UNUSED(info);
 *                                                                                         ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
 *                                                                                             ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);
 *
 *                                                                                                 Window win(window);
 *                                                                                                     win.set(Window::DimX, Window::Dimension(0, 0, 0));
 *                                                                                                         win.set(Window::DimY, Window::Dimension(0, 0, 0));
 *                                                                                                             win.set(Window::DimZ, Window::Dimension(0, 0, 0));
 *                                                                                                                 Iterator in_a(_input_a, win);
 *                                                                                                                     Iterator in_b(_input_b, win);
 *                                                                                                                         Iterator out(_output, win);
 *
 *                                                                                                                             execute_window_loop(window, [&](const Coordinates & id)
 *                                                                                                                                 {
 *                                                                                                                                         size_t x=id[0];size_t y=id[1]; size_t z=id[2];
 *                                                                                                                                                 size_t offset=_input_a->info()->offset_element_in_bytes(Coordinates(x,y,z));
 *                                                                                                                                                         int16_t *in_a_addr=reinterpret_cast<int16_t*>(_input_a->buffer()+offset);
 *                                                                                                                                                                 int16_t *in_b_addr=reinterpret_cast<int16_t*>(_input_b->buffer()+offset);
 *                                                                                                                                                                         int16_t *out_addr=reinterpret_cast<int16_t*>(_output->buffer()+offset);
 *                                                                                                                                                                                 int16x8_t input1=vld1q_s16(in_a_addr);
 *                                                                                                                                                                                         int16x8_t input2=vld1q_s16(in_b_addr);
 *                                                                                                                                                                                                 int16x8_t result=vmulq_s16(input1, input2);
 *                                                                                                                                                                                                         vst1q_s16(out_addr, result);
 *                                                                                                                                                                                                             },
*                                                                                                                                                                                                                 in_a,  in_b, out);
*                                                                                                                                                                                                                 }
*                                                                                                                                                                                                                 */
