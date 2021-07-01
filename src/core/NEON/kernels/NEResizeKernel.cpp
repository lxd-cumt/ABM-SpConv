#include "arm_compute/core/NEON/kernels/NEResizeKernel.h"

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

	std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *output, TensorShape output_shape)
	{
		    output->set_data_type(input->data_type());
		        output->set_num_channels(input->num_channels());
			    output->set_tensor_shape(output_shape);
			        output->set_quantization_info(input->quantization_info());
				    output->set_data_layout(input->data_layout());
				     
				        Window win = calculate_max_window(*input, Steps());

					    win.set(Window::DimX, Window::Dimension(0, input->dimension(0),input->dimension(0)));   
					        win.set(Window::DimY, Window::Dimension(0, input->dimension(1),input->dimension(1)));   
						    win.set(Window::DimZ, Window::Dimension(0, input->dimension(2),input->dimension(2)));   
						        
						        
						        return std::make_pair(Status{}, win);
	}
} 


NEResizeKernel::NEResizeKernel()
	    :  _input(nullptr), _output(nullptr), _output_shape()
{
}

void NEResizeKernel::configure(const ITensor *input, ITensor *output, TensorShape output_shape)
{
	    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

	        _input          = input;
		    _output         = output;
		        _output_shape=output_shape;
			   
			    auto win_config = validate_and_configure_window(input->info(), output->info(), output_shape);
			        ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
				    INEKernel::configure(win_config.second);
}

Status NEResizeKernel::validate(const ITensorInfo *input, const ITensorInfo *output)
{
	    return Status{};
}

void NEResizeKernel::run(const Window &window, const ThreadInfo &info)
{
	    ARM_COMPUTE_UNUSED(info);
	        ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
		    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

		        Window win(window);
			    win.set(Window::DimX, Window::Dimension(0, 0, 0));
			        win.set(Window::DimY, Window::Dimension(0, 0, 0));
				    win.set(Window::DimZ, Window::Dimension(0, 0, 0));
				        Iterator in(_input, win);
					    Iterator out(_output, win);

					        execute_window_loop(window, [&](const Coordinates & id)
								    {
								            uint8_t *in_ptr=_input->buffer();
									            float *in_addr=reinterpret_cast<float*>(in_ptr);
										            int8_t *in_addr_s8=reinterpret_cast<int8_t*>(in_ptr);

											            uint8_t *out_ptr=_output->buffer();
												            float *out_addr=reinterpret_cast<float*>(out_ptr);
													            int8_t *out_addr_s8=reinterpret_cast<int8_t*>(out_ptr);
														            
														            for(unsigned int z=0; z<_output_shape.z(); z++)
															            {
																                for(unsigned int y=0; y<_output_shape.y(); y++)
																		            {
																			                    for(unsigned int x=0; x<_output_shape.x(); x++)
																					                    {
																							                        if(_output->info()->data_type()==DataType::F32)
																										                    {
																												                            (*out_addr)=(*in_addr);
																															                            out_addr++; in_addr++;
																																		                        }
																										                    else if(_output->info()->data_type()==DataType::S8)
																													                        {
																																	                        (*out_addr_s8)=(*in_addr_s8);
																																				                        out_addr_s8++;in_addr_s8++;
																																							                    }
																												                    }
																					                }
																		        }
															        },
							    in, out);
}

