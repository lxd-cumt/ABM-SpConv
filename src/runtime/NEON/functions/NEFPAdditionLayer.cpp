#include "arm_compute/runtime/NEON/functions/NEFPAdditionLayer.h"


#include "arm_compute/core/Size2D.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "utils/ImageLoader.h"
#include "utils/Utils.h"

using namespace arm_compute;

NEFPAdditionLayer::NEFPAdditionLayer():_add_kernel()
{
}

void NEFPAdditionLayer::configure(const ITensor *input_a, const ITensor *input_b, ITensor *output,  int *fp)

{
    _add_kernel.configure(input_a,input_b,output,fp);
}

Status NEFPAdditionLayer::validate(const ITensorInfo *input_a, const ITensorInfo *input_b, const ITensorInfo *output)
{
    // ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(weights);
    // ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(weights, 1, DataType::QASYMM8, DataType::QSYMM8_PER_CHANNEL, DataType::F16, DataType::F32);
    // ARM_COMPUTE_RETURN_ERROR_ON(weights->num_dimensions() > 4);
    // NEABMWeightsReshapeKernel::validate(weights,quantity,location);
    return Status{};

}

void NEFPAdditionLayer::run()
{
    NEScheduler::get().schedule(&_add_kernel, Window::DimZ);//Note in which dimension the scheduler execute on parallel!
}

