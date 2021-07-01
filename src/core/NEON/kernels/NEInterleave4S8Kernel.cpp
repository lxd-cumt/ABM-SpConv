/*
 *  * Copyright (c) 2016-2019 ARM Limited.
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
#include "arm_compute/core/NEON/kernels/NEInterleave4S8Kernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/NEON/INEKernel.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"

#include <arm_neon.h>
#include <cstddef>
#include <cstdint>
#include <tuple>

using namespace arm_compute;
using namespace arm_compute::misc::shape_calculator;

namespace
{
	Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output)
	{
		    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QASYMM8, DataType::U8, DataType::S8,
				                                                             DataType::U16, DataType::S16, DataType::U32, DataType::S32,
											                                                              DataType::F16, DataType::F32);
		        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);

			    if(output->total_size() != 0)
				        {
						        TensorShape output_shape = input->tensor_shape();
							        output_shape.set(0, input->dimension(0) * 8);
								        output_shape.set(1, std::ceil(input->dimension(1) / 8.0f));
									        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(output->tensor_shape(), output_shape);
										        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
											        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_QUANTIZATION_INFO(input, output);
												    }

			        return Status{};
	}

	std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *output)
	{
		    Window                win = calculate_max_window(*input, Steps());
		        win.set(Window::DimX, Window::Dimension(0, input->dimension(0), input->dimension(0)));  
			    if(input->dimension(1)<8){
				            win.set(Window::DimY, Window::Dimension(0, input->dimension(1), input->dimension(1)));  
					        }
			        else{
					         win.set(Window::DimY, Window::Dimension(0, input->dimension(1), 8));  
						     }
				    win.set(Window::DimZ, Window::Dimension(0, input->dimension(2),input->dimension(2)));   
				        return std::make_pair(Status{}, win);
	}

	void gemm_interleave_8bit_elements(const ITensor *input, ITensor *output, const Window &window)
	{
		    const size_t in_stride = input->info()->strides_in_bytes()[1];
		        unsigned int y_last=output->info()->dimension(1)%8;
			    Window window_in_out(window);
			        window_in_out.set(Window::DimX, Window::Dimension(0, 0, 0));
				    window_in_out.set(Window::DimY, Window::Dimension(0, 0, 0));
				        window_in_out.set(Window::DimZ, Window::Dimension(0, 0, 0));
					    Iterator in(input,window_in_out);
					        Iterator out(output,window_in_out);

						    execute_window_loop(window, [&](const Coordinates &id)
								        {
									           size_t y_dimension=id[1];
										              size_t input_offset=input->info()->offset_element_in_bytes(Coordinates(0,y_dimension,0));
											                 size_t output_offset=output->info()->offset_element_in_bytes(Coordinates(0,y_dimension/8,0));
													            int8_t *out_addr=reinterpret_cast<int8_t*>(output->buffer()+output_offset);
														               if(y_dimension/8!=input->info()->dimension(1)/8)
															                  {
																	                  unsigned int ope_times=input->info()->dimension(0)/8;
																			                  for(unsigned int i=0;i<ope_times*8;i=i+8)
																					                  {
																							                      int8_t *start_addr=reinterpret_cast<int8_t*>(input->buffer()+input_offset+i);
																									                          int8x8x4_t data1 =
																												                      {
																														                              {
																																	                                  vld1_s8(start_addr+ 0 * in_stride),
																																					                              vld1_s8(start_addr +1 * in_stride),
																																								                                  vld1_s8(start_addr+ 2 * in_stride),
																																												                              vld1_s8(start_addr +3 * in_stride),
																																															                              }
																																																		                          };
																												                      int8x8x4_t data2 =
																															                          {
																																			                          {
																																							                              vld1_s8(start_addr+ 4 * in_stride),
																																										                                  vld1_s8(start_addr +5 * in_stride),
																																														                              vld1_s8(start_addr+ 6 * in_stride),
																																																	                                  vld1_s8(start_addr +7 * in_stride),
																																																					                          }
																																						                      };
																														                          for(unsigned int j=0;j<4;j++)
																																		                      {
																																					                              int8x8x2_t temp=vzip_s8(data1.val[j],data2.val[j]);
																																								                              data1.val[j]=temp.val[0];
																																											                              data2.val[j]=temp.val[1];
																																														                          }
																																	                      vst4_s8(out_addr,data1);
																																			                          vst4_s8(out_addr+32,data2);
																																						                      
																																						                      out_addr=out_addr+64;
																																								                      }
																					                  for(unsigned int i=ope_times*8;i<input->info()->dimension(0);i++)
																								                  {
																											                      uint8_t *start_addr_2=input->buffer()+input_offset+i;
																													                         for(unsigned int j=0;j<8;j++)
																																	                    {
																																				                           int8_t *in_addr=reinterpret_cast<int8_t*>(start_addr_2+j*in_stride);
																																							                          (*out_addr)=(*(in_addr));
																																										                         out_addr++;
																																													                    }
																																                 }
																							                  /*
																									   *                for(unsigned int i=0;i<input->info()->dimension(0);i++)
																									   *                               {
																									   *                                                  uint8_t *start_addr=input->buffer()+input_offset+i;
																									   *                                                                     for(unsigned int j=0;j<8;j++)
																									   *                                                                                        {
																									   *                                                                                                               int8_t *in_addr=reinterpret_cast<int8_t*>(start_addr+j*in_stride);
																									   *                                                                                                                                      (*out_addr)=(*(in_addr));
																									   *                                                                                                                                                             out_addr++;
																									   *                                                                                                                                                                                }
																									   *                                                                                                                                                                                               }
																									   *                                                                                                                                                                                                              */
																							             }
															                  else{
																		                 if(y_last==0){y_last=8;}
																				                for(unsigned int i=0;i<input->info()->dimension(0);i++)
																							               {
																									                          uint8_t *start_addr=input->buffer()+input_offset+i;
																												                     for(unsigned int j=0;j<8;j++)
																															                        {
																																			                       if(j<y_last){
																																						                                        int8_t *in_addr=reinterpret_cast<int8_t*>(start_addr+j*in_stride);
																																											                                (*out_addr)=(*(in_addr));
																																															                                            out_addr++;
																																																				                           }
																																					                              else{
																																									                                        (*out_addr)=0;
																																														                                out_addr++;
																																																		                       }
																																								                         }
																														                    }
																						           }
																	      },
							        in, out);

	}
} 

NEInterleave4S8Kernel::NEInterleave4S8Kernel()
	    :_x_last(0),_y_last(0)
{
}

void NEInterleave4S8Kernel::configure(const ITensor *input, ITensor *output)
{
	    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

	        TensorShape shape{input->info()->tensor_shape()};
		    shape.set(0,input->info()->dimension(0)*8);
		        shape.set(1,std::ceil(input->info()->dimension(1) / 8.0f));
			    output->info()->set_data_type(DataType::S8);
			        output->info()->set_num_channels(input->info()->num_channels());
				    output->info()->set_tensor_shape(shape);
				        output->info()->set_quantization_info(input->info()->quantization_info());
					    output->info()->set_data_layout(input->info()->data_layout());

					        ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), output->info()));

						    _input  = input;
						        _output = output;
							    _x_last=input->info()->dimension(0)%8;
							        _y_last=input->info()->dimension(1)%8;

								    auto win_config = validate_and_configure_window(input->info(), output->info());
								        ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
									    INEKernel::configure(win_config.second);
}

Status NEInterleave4S8Kernel::validate(const ITensorInfo *input, const ITensorInfo *output)
{
	    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output));
	        ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(input->clone().get(), output->clone().get()).first);

		    return Status{};
}

void NEInterleave4S8Kernel::run(const Window &window, const ThreadInfo &info)
{
	    ARM_COMPUTE_UNUSED(info);
	        ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
		    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);
		        gemm_interleave_8bit_elements(_input,_output,window);

}

