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
#include "arm_compute/core/NEON/kernels/NEFP2AdditionKernel.h"

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
	std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *output)
	{
		   
		        output->set_data_type(DataType::F32);
			        output->set_num_channels(input->num_channels());
				        output->set_tensor_shape(input->tensor_shape());
					        output->set_quantization_info(input->quantization_info());
						        output->set_data_layout(input->data_layout());
							    /*auto_init_if_empty(*output, input->clone()->set_tensor_shape(compute_col2im_shape(*input, convolved_dims, false)));*/

							    /*Configure kernel window*/
							    Window win = calculate_max_window(*input, Steps());

							        /* The NECol2ImKernel doesn't need padding so update_window_and_padding() can be skipped*/
							        win.set(Window::DimX, Window::Dimension(0, input->dimension(0),input->dimension(0)));   
								    win.set(Window::DimY, Window::Dimension(0, input->dimension(1),input->dimension(1)));   
								        win.set(Window::DimZ, Window::Dimension(0, input->dimension(2),1));   
									    
									    
									    return std::make_pair(Status{}, win);
	}
} 


NEFP2AdditionKernel::NEFP2AdditionKernel()
	    :  _input_a(nullptr), _input_b(nullptr),_output(nullptr),a_mul_value(0),b_mul_value(0)
{
}

void NEFP2AdditionKernel::configure(const ITensor *input_a,const ITensor *input_b,  ITensor *output, int *fp)
{
	    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
	        /*ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), output->info()));*/

	        _input_a          = input_a;
		    _input_b          = input_b;
		        _output         = output;
			    a_mul_value=pow(2,fp[2]-fp[0]);
			        b_mul_value=pow(2,fp[2]-fp[1]);
				   
				    auto win_config = validate_and_configure_window(input_a->info(), output->info());
				        ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
					    INEKernel::configure(win_config.second);
}

Status NEFP2AdditionKernel::validate(const ITensorInfo *input_a, const ITensorInfo *input_b, const ITensorInfo *output)
{
	    return Status{};
}

void NEFP2AdditionKernel::run(const Window &window, const ThreadInfo &info)
{
	    ARM_COMPUTE_UNUSED(info);
	        ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
		    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);
		        Window win_out(window);
			    win_out.set(Window::DimX, Window::Dimension(0, 0, 0));
			        win_out.set(Window::DimY, Window::Dimension(0, 0, 0));
				    win_out.set(Window::DimZ, Window::Dimension(0, 0, 0));
				        Iterator in_a(_input_a, win_out);
					    Iterator in_b(_input_b,win_out);
					        Iterator out(_output, win_out);

						    float32x4_t max_value=vdupq_n_f32(127);
						        float32x4_t min_value=vdupq_n_f32(-128);

							    float32x4_t a_mul=vdupq_n_f32(a_mul_value);
							        float32x4_t b_mul=vdupq_n_f32(b_mul_value);
								    

								    execute_window_loop(window, [&](const Coordinates & id)
										        {
											        size_t z_dimension=id[2];
												        uint8_t *in_a_ptr=_input_a->buffer();
													        uint8_t *in_b_ptr=_input_b->buffer();
														        uint8_t *out_ptr=_output->buffer();
															        for(unsigned int y=0;y<_input_a->info()->dimension(1);y++)
																        {
																	            unsigned int nums=_input_a->info()->dimension(0)/4;
																		                for(unsigned int x=0;x<nums*4;x+=4)
																				            {
																					                    float *start_a_addr=reinterpret_cast<float*>(in_a_ptr+_input_a->info()->offset_element_in_bytes(Coordinates(x,y,z_dimension)));
																							                    float *start_b_addr=reinterpret_cast<float*>(in_b_ptr+_input_b->info()->offset_element_in_bytes(Coordinates(x,y,z_dimension)));
																									                    float *out_addr=reinterpret_cast<float*>(out_ptr+_output->info()->offset_element_in_bytes(Coordinates(x,y,z_dimension)));
																											                    float32x4_t ope_a=vld1q_f32(start_a_addr);
																													                    float32x4_t ope_b=vld1q_f32(start_b_addr);

																															                    float32x4_t ope_a_mul, ope_b_mul;
																																	                    ope_a_mul=vmulq_f32(ope_a,a_mul);
																																			                    ope_b_mul=vmulq_f32(ope_b,b_mul);
																																					                    float32x4_t result=vaddq_f32(ope_a_mul,ope_b_mul);
																																							                    result=vminq_f32(max_value,vmaxq_f32(min_value,result));
																																									                    vst1q_f32(out_addr,result);
																																											                }
																				            for(unsigned int x=nums*4;x<_input_a->info()->dimension(0);x++)
																						                {
																									                float *start_a_addr=reinterpret_cast<float*>(in_a_ptr+_input_a->info()->offset_element_in_bytes(Coordinates(x,y,z_dimension)));
																											                float *start_b_addr=reinterpret_cast<float*>(in_b_ptr+_input_b->info()->offset_element_in_bytes(Coordinates(x,y,z_dimension)));
																													                float *out_addr=reinterpret_cast<float*>(out_ptr+_output->info()->offset_element_in_bytes(Coordinates(x,y,z_dimension)));
																															                float r1=(*start_a_addr)*a_mul_value;
																																	                float r2=(*start_b_addr)*b_mul_value;
																																			                float result=r1+r2;
																																					                float real_result;
																																							                if(result>127){real_result=127;}
																																									                else if(result<-128){real_result=-128;}
																																											                else{
																																														                    real_result=result;
																																																                    }
																																													                (*out_addr)=real_result;
																																															            }
																					         } },
									        in_a,in_b,out);
}

