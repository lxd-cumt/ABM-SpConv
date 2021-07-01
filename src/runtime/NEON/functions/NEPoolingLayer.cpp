#include "arm_compute/runtime/NEON/functions/NEPoolingLayer.h"

#include "arm_compute/core/ITensor.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"

#include "support/ToolchainSupport.h"
#include <fstream>
#include <chrono>
using namespace std;

using namespace arm_compute;

NEPoolingLayer::NEPoolingLayer()
	    : _pooling_layer_kernel(), _border_handler(), _is_global_pooling_layer(false), _data_layout(DataLayout::NCHW),
	        _layer_time(0.f), count(100), now(0)
{
}

void NEPoolingLayer::configure(ITensor *input, ITensor *output, const PoolingLayerInfo &pool_info)
{
	    _is_global_pooling_layer = (input->info()->dimension(0) == pool_info.pool_size().width) && (input->info()->dimension(1) == pool_info.pool_size().height);

	        _data_layout = input->info()->data_layout();

		    _pooling_layer_kernel.configure(input, output, pool_info);

		        switch(_data_layout)
				    {
					            case DataLayout::NCHW:
							            {
									                BorderMode border_mode = (pool_info.pool_type() == PoolingType::MAX) ? BorderMode::REPLICATE : BorderMode::CONSTANT;
											            PixelValue zero_value(0.f);
												                if(is_data_type_quantized_asymmetric(input->info()->data_type()) && !pool_info.exclude_padding())
															            {
																	                    zero_value = PixelValue(static_cast<uint32_t>(input->info()->quantization_info().uniform().offset));
																			                }
														            _border_handler.configure(input, _pooling_layer_kernel.border_size(), border_mode, zero_value);
															                break;
																	        }
								            case DataLayout::NHWC:
								                break;
										        default:
										            ARM_COMPUTE_ERROR("Data layout not supported");
											        }
}

Status NEPoolingLayer::validate(const ITensorInfo *input, const ITensorInfo *output, const PoolingLayerInfo &pool_info)
{
	    return NEPoolingLayerKernel::validate(input, output, pool_info);
}

void NEPoolingLayer::run()
{
	    ofstream out("./lab2/alexnet/alexnet_avg_time.csv", ios::out | ios::app);
	        auto b=std::chrono::high_resolution_clock::now();

		    switch(_data_layout)
			        {
					        case DataLayout::NCHW:
							            NEScheduler::get().schedule(&_border_handler, Window::DimY);

								                NEScheduler::get().schedule(&_pooling_layer_kernel, _is_global_pooling_layer ? Window::DimZ : Window::DimY);
										            break;
											            case DataLayout::NHWC:
											                NEScheduler::get().schedule(&_pooling_layer_kernel, Window::DimX);
													            break;
														            default:
														                ARM_COMPUTE_ERROR("Data layout not supported");
																    }

		        auto e=std::chrono::high_resolution_clock::now();
			    double ttime=std::chrono::duration_cast<std::chrono::duration<double>>(e - b).count(); 
			        if(now>0)
					    {
						            _layer_time+=(ttime*1000);
							        }
				    if(now==(count-1))
					        {
							        _layer_time=_layer_time/(count-1);
								        out<<"pool_layer"<<","<<_layer_time;
									        out<<std::endl;
										        out.close();
											    }
				        now++;
}
