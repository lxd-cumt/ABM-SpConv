#include "arm_compute/runtime/NEON/functions/NEF32toS8Layer.h"


#include "arm_compute/core/Size2D.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "utils/ImageLoader.h"
#include "utils/Utils.h"

using namespace arm_compute;

NEF32toS8Layer::NEF32toS8Layer():_f32_to_s8_kernel()
{
}

void NEF32toS8Layer::configure(const ITensor *input, ITensor *output)

{
    _f32_to_s8_kernel.configure(input,output);
}

Status NEF32toS8Layer::validate(const ITensorInfo *input, const ITensorInfo *output)
{
    // ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(weights);
    // ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(weights, 1, DataType::QASYMM8, DataType::QSYMM8_PER_CHANNEL, DataType::F16, DataType::F32);
    // ARM_COMPUTE_RETURN_ERROR_ON(weights->num_dimensions() > 4);
    // NEABMWeightsReshapeKernel::validate(weights,quantity,location);
    return Status{};

}

void NEF32toS8Layer::run()
{
    NEScheduler::get().schedule(&_f32_to_s8_kernel, Window::DimZ);//Note in which dimension the scheduler execute on parallel!
}

