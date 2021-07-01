#include "arm_compute/core/NEON/kernels/NEActLayerKernel.h"

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
		    output->set_data_type(input->data_type());
		        output->set_num_channels(input->num_channels());
			    output->set_tensor_shape(input->tensor_shape());
			        output->set_quantization_info(input->quantization_info());
				    output->set_data_layout(input->data_layout());

				        Window win = calculate_max_window(*input, Steps());

					   
					    win.set(Window::DimX, Window::Dimension(0, input->dimension(0),input->dimension(0)));   
					        win.set(Window::DimY, Window::Dimension(0, input->dimension(1),1));   
						    win.set(Window::DimZ, Window::Dimension(0, input->dimension(2),1));   
						        
						        
						        return std::make_pair(Status{}, win);
	}
} 


NEActLayerKernel::NEActLayerKernel()
	    :  _input(nullptr), _output(nullptr)
{
}

void NEActLayerKernel::configure(const ITensor *input, ITensor *output)
{
	    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

	        _input          = input;
		    _output         = output;
		       
		        auto win_config = validate_and_configure_window(input->info(), output->info());
			    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
			        INEKernel::configure(win_config.second);
}

Status NEActLayerKernel::validate(const ITensorInfo *input, const ITensorInfo *output)
{
	    return Status{};
}

void NEActLayerKernel::run(const Window &window, const ThreadInfo &info)
{
	    Window wout(window);
	        wout.set(Window::DimX, Window::Dimension(0, 0, 0));
		    wout.set(Window::DimY, Window::Dimension(0, 0, 0));
		        wout.set(Window::DimZ, Window::Dimension(0, 0, 0));
			    Iterator input(_input, wout);
			        Iterator output(_output, wout);

				    int8x16_t const_0=vdupq_n_s8(0);
				        int8x16_t tmp=vdupq_n_s8(0);

					    int window_end=static_cast<int>(_input->info()->dimension(0));

					        execute_window_loop(window, [&](const Coordinates &id)
								    {
								            size_t y_dimension=id[1], z_dimension=id[2];
									           
									            const int8_t* input_ptr  = reinterpret_cast<const int8_t*>(_input->buffer()+_input->info()->offset_element_in_bytes(Coordinates(0, y_dimension, z_dimension)));
										            int8_t* output_ptr = reinterpret_cast<int8_t*>(_output->buffer()+_output->info()->offset_element_in_bytes(Coordinates(0, y_dimension, z_dimension)));

											            int x = 0;
												            
												            for(; x <= (window_end - 16); x += 16)
													            {
														                int8x16_t vin = vld1q_s8(input_ptr + x);
																            tmp = vmaxq_s8(const_0, vin);
																	                vst1q_s8(output_ptr + x, tmp);
																			        }
																				        

																				        for(; x < window_end; ++x)
																					        {
																						            const int8_t in = *(reinterpret_cast<const int8_t *>(input_ptr + x));
																							                int8_t       tmp2;
																									            tmp2 = std::max<int8_t>(static_cast<int8_t>(0), in);
																										                *(output_ptr + x) = tmp2;
																												        }
																					    },
							    input, output);
}

