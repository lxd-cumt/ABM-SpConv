#include "arm_compute/runtime/NEON/functions/NEBatchNormalizationLayer.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include <fstream>
#include <chrono>
using namespace std;

using namespace arm_compute;

NEBatchNormalizationLayer::NEBatchNormalizationLayer()
	    : _norm_kernel(),
	        _layer_time(0.f), count(100), now(0)
{
}

void NEBatchNormalizationLayer::configure(ITensor *input, ITensor *output, const ITensor *mean, const ITensor *var, const ITensor *beta, const ITensor *gamma, float epsilon,
		                                          ActivationLayerInfo act_info)
{
	    _norm_kernel.configure(input, output, mean, var, beta, gamma, epsilon, act_info);
}

Status NEBatchNormalizationLayer::validate(const ITensorInfo *input, const ITensorInfo *output, const ITensorInfo *mean, const ITensorInfo *var, const ITensorInfo *beta, const ITensorInfo *gamma,
		                                           float epsilon, ActivationLayerInfo act_info)
{
	    ARM_COMPUTE_RETURN_ON_ERROR(NEBatchNormalizationLayerKernel::validate(input, output, mean, var, beta, gamma, epsilon, act_info));
	        return Status{};
}

void NEBatchNormalizationLayer::run()
{
	    ofstream out("./lab2/alexnet/alexnet_avg_time.csv", ios::out | ios::app);
	        auto b=std::chrono::high_resolution_clock::now();

		    NEScheduler::get().schedule(&_norm_kernel, Window::DimY);

		        auto e=std::chrono::high_resolution_clock::now();
			    double ttime=std::chrono::duration_cast<std::chrono::duration<double>>(e - b).count(); 
			        if(now>0)
					    {
						            _layer_time+=(ttime*1000);
							        }
				    if(now==(count-1))
					        {
							        _layer_time=_layer_time/(count-1);
								        out<<"norm_layer"<<","<<_layer_time;
									        out<<std::endl;
										        out.close();
											    }
				        now++;
}
