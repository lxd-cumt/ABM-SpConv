#include "arm_compute/runtime/NEON/functions/NEActLayer.h"


#include "arm_compute/core/Size2D.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "utils/ImageLoader.h"
#include "utils/Utils.h"

using namespace arm_compute;

NEActLayer::NEActLayer():_act_kernel()
{
}

void NEActLayer::configure(const ITensor *input, ITensor *output)

{
	    _act_kernel.configure(input,output);
}

Status NEActLayer::validate(const ITensorInfo *input, const ITensorInfo *output)
{
	    return Status{};

}

void NEActLayer::run()
{
	    NEScheduler::get().schedule(&_act_kernel, Window::DimZ);
}

