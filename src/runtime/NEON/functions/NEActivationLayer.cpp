/*
 * #include "arm_compute/runtime/NEON/functions/NEActivationLayer.h"
 *
 * #include "support/ToolchainSupport.h"
 * #include "arm_compute/core/Error.h"
 * #include "arm_compute/core/TensorInfo.h"
 * #include "arm_compute/core/Types.h"
 * #include "arm_compute/core/Validate.h"
 * #include "arm_compute/runtime/NEON/NEScheduler.h"
 *
 * using namespace arm_compute;
 *
 * NEActivationLayer::NEActivationLayer()
 *     :_act_kernel()
 *     {
 *     }
 *
 *
 *     void NEActivationLayer::configure(ITensor *input, ITensor *output, ActivationLayerInfo activation_info)
 *     {
 *        _act_kernel.configure(input, output, activation_info);
 *        }
 *
 *        Status NEActivationLayer::validate(const ITensorInfo *input, const ITensorInfo *output, const ActivationLayerInfo &act_info)
 *        {
 *            ARM_COMPUTE_RETURN_ON_ERROR(NEActivationLayerKernel::validate(input, output, act_info));
 *                return Status{};
 *                }
 *
 *                void NEActivationLayer::run()
 *                {
 *                    NEScheduler::get().schedule(&_act_kernel, Window::DimY);
 *                    }
 *                    */

#include "arm_compute/runtime/NEON/functions/NEActivationLayer.h"

#include "arm_compute/core/NEON/kernels/NEActivationLayerKernel.h"
#include "support/ToolchainSupport.h"

using namespace arm_compute;

void NEActivationLayer::configure(ITensor *input, ITensor *output, ActivationLayerInfo activation_info)
{
	    auto k = arm_compute::support::cpp14::make_unique<NEActivationLayerKernel>();
	        k->configure(input, output, activation_info);
		    _kernel = std::move(k);
}

Status NEActivationLayer::validate(const ITensorInfo *input, const ITensorInfo *output, const ActivationLayerInfo &act_info)
{
	    return NEActivationLayerKernel::validate(input, output, act_info);
}
