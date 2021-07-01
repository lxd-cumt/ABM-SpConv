#include "arm_compute/runtime/NEON/functions/NES8toF32Layer.h"


#include "arm_compute/core/Size2D.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "utils/ImageLoader.h"
#include "utils/Utils.h"

using namespace arm_compute;

NES8toF32Layer::NES8toF32Layer():_s8_to_f32_kernel()
{
}

void NES8toF32Layer::configure(const ITensor *input, ITensor *output)

{
    _s8_to_f32_kernel.configure(input,output);
}

Status NES8toF32Layer::validate(const ITensorInfo *input, const ITensorInfo *output)
{
    // ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(weights);
    // ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(weights, 1, DataType::QASYMM8, DataType::QSYMM8_PER_CHANNEL, DataType::F16, DataType::F32);
    // ARM_COMPUTE_RETURN_ERROR_ON(weights->num_dimensions() > 4);
    // NEABMWeightsReshapeKernel::validate(weights,quantity,location);
    return Status{};

}

void NES8toF32Layer::run()
{
    NEScheduler::get().schedule(&_s8_to_f32_kernel, Window::DimZ);//Note in which dimension the scheduler execute on parallel!
}

