#include "arm_compute/runtime/NEON/functions/NEMulAddTestLayer.h"


#include "arm_compute/core/Size2D.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "utils/ImageLoader.h"
#include "utils/Utils.h"

using namespace arm_compute;

NEMulAddTestLayer::NEMulAddTestLayer():_test_kernel()
{
}

void NEMulAddTestLayer::configure(const ITensor *input_a,  const ITensor *input_b,  ITensor *output)

{
	    _test_kernel.configure(input_a, input_b,  output);
}

Status NEMulAddTestLayer::validate(const ITensorInfo *input_a, const ITensorInfo *input_b,  const ITensorInfo *output)
{
	    return Status{};

}

void NEMulAddTestLayer::run()
{
	    NEScheduler::get().schedule(&_test_kernel, Window::DimX);
}

