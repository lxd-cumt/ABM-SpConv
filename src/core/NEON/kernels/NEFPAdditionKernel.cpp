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
#include "arm_compute/core/NEON/kernels/NEFPAdditionKernel.h"

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
		  
		        output->set_data_type(DataType::S8);
			        output->set_num_channels(input->num_channels());
				        output->set_tensor_shape(input->tensor_shape());
					        output->set_quantization_info(input->quantization_info());
						        output->set_data_layout(input->data_layout());

							    Window win = calculate_max_window(*input, Steps());

							        win.set(Window::DimX, Window::Dimension(0, input->dimension(0),input->dimension(0)));   
								    win.set(Window::DimY, Window::Dimension(0, input->dimension(1),input->dimension(1)));   
								        win.set(Window::DimZ, Window::Dimension(0, input->dimension(2),1));   
									    

									    return std::make_pair(Status{}, win);
	}
} 


NEFPAdditionKernel::NEFPAdditionKernel()
	    :  _input_a(nullptr), _input_b(nullptr),_output(nullptr),a_mul_value(0),b_mul_value(0)
{
}

void NEFPAdditionKernel::configure(const ITensor *input_a,const ITensor *input_b,  ITensor *output, int *fp)
{
	    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

	        _input_a          = input_a;
		    _input_b          = input_b;
		        _output         = output;
			    a_mul_value=pow(2,fp[2]-fp[0]);
			        b_mul_value=pow(2,fp[2]-fp[1]);
				    auto win_config = validate_and_configure_window(input_a->info(), output->info());
				        ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
					    INEKernel::configure(win_config.second);
}

Status NEFPAdditionKernel::validate(const ITensorInfo *input_a, const ITensorInfo *input_b, const ITensorInfo *output)
{
	   
	    return Status{};
}

void NEFPAdditionKernel::run(const Window &window, const ThreadInfo &info)
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
						    int16x8_t max_value=vdupq_n_s16(127);
						        int16x8_t min_value=vdupq_n_s16(-128);

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
																	            unsigned int nums=_input_a->info()->dimension(0)/8;
																		                for(unsigned int x=0;x<nums*8;x+=8)
																				            {
																					                    int8_t *start_a_addr=reinterpret_cast<int8_t*>(in_a_ptr+_input_a->info()->offset_element_in_bytes(Coordinates(x,y,z_dimension)));
																							                    int8_t *start_b_addr=reinterpret_cast<int8_t*>(in_b_ptr+_input_b->info()->offset_element_in_bytes(Coordinates(x,y,z_dimension)));
																									                    int8_t *out_addr=reinterpret_cast<int8_t*>(out_ptr+_output->info()->offset_element_in_bytes(Coordinates(x,y,z_dimension)));
																											                    int8x8_t ope_a=vld1_s8(start_a_addr);
																													                    int8x8_t ope_b=vld1_s8(start_b_addr);
																															                   
																															                    int16x8_t ope_a_long=vmovl_s8(ope_a);
																																	                    int16x8_t ope_b_long=vmovl_s8(ope_b);

																																			                    int16x4_t ope_a_high=vget_high_s16(ope_a_long);
																																					                    int16x4_t ope_a_low=vget_low_s16(ope_a_long);
																																							                    int16x4_t ope_b_high=vget_high_s16(ope_b_long);
																																									                    int16x4_t ope_b_low=vget_low_s16(ope_b_long);

																																											                    int32x4_t ope_a_high_long=vmovl_s16(ope_a_high);
																																													                    int32x4_t ope_a_low_long=vmovl_s16(ope_a_low);
																																															                    int32x4_t ope_b_high_long=vmovl_s16(ope_b_high);
																																																	                    int32x4_t ope_b_low_long=vmovl_s16(ope_b_low);

																																																			                    float32x4_t a_high=vcvtq_f32_s32(ope_a_high_long);
																																																					                    float32x4_t a_low=vcvtq_f32_s32(ope_a_low_long);
																																																							                    float32x4_t b_high=vcvtq_f32_s32(ope_b_high_long);
																																																									                    float32x4_t b_low=vcvtq_f32_s32(ope_b_low_long);


																																																											                    a_high=vmulq_f32(a_high,a_mul);
																																																													                    a_low=vmulq_f32(a_low,a_mul);
																																																															                    b_high=vmulq_f32(b_high,b_mul);
																																																																	                    b_low=vmulq_f32(b_low,b_mul);

																																																																			                    float32x4_t result1=vaddq_f32(a_high,b_high);
																																																																					                    float32x4_t result2=vaddq_f32(a_low,b_low);


																																																																							                    int32x4_t result1_s32=vcvtq_s32_f32(result1);
																																																																									                    int32x4_t result2_s32=vcvtq_s32_f32(result2);

																																																																											                    int16x4_t result1_s16=vmovn_s32(result1_s32);
																																																																													                    int16x4_t result2_s16=vmovn_s32(result2_s32);

																																																																															                    int16x8_t result=vcombine_s16(result2_s16,result1_s16);

																																																																																	                    result=vminq_s16(max_value,vmaxq_s16(min_value,result));
																																																																																			                    int8x8_t real_result=vmovn_s16(result);
																																																																																					                    vst1_s8(out_addr,real_result);
																																																																																							                }
																				            for(unsigned int x=nums*8;x<_input_a->info()->dimension(0);x++)
																						                {
																									                int8_t *start_a_addr=reinterpret_cast<int8_t*>(in_a_ptr+_input_a->info()->offset_element_in_bytes(Coordinates(x,y,z_dimension)));
																											                int8_t *start_b_addr=reinterpret_cast<int8_t*>(in_b_ptr+_input_b->info()->offset_element_in_bytes(Coordinates(x,y,z_dimension)));
																													                int8_t *out_addr=reinterpret_cast<int8_t*>(out_ptr+_output->info()->offset_element_in_bytes(Coordinates(x,y,z_dimension)));

																															                float ope_a=(float)(*start_a_addr);
																																	                float ope_b=(float)(*start_b_addr);
																																			                ope_a=ope_a*a_mul_value;
																																					                ope_b=ope_b*b_mul_value;
																																							                float result;
																																									                result=ope_a+ope_b;
																																											                float real_result=0;
																																													                if(result>127){real_result=127;}
																																															                else if(result<-128){real_result=-128;}
																																																	                else{
																																																				                    real_result=result;
																																																						                    }
																																																			                (*out_addr)=(int8_t)real_result;
																																																					            }
																					         } },
									        in_a,in_b,out);
}

