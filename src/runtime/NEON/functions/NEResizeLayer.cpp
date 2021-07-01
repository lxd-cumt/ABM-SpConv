#include "arm_compute/runtime/NEON/functions/NEResizeLayer.h"


#include "arm_compute/core/Size2D.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "utils/ImageLoader.h"
#include "utils/Utils.h"

using namespace arm_compute;

NEResizeLayer::NEResizeLayer():_resize_kernel()
{
}

void NEResizeLayer::configure(const ITensor *input, ITensor *output, TensorShape output_shape)

{
	   _resize_kernel.configure(input, output, output_shape);
}

Status NEResizeLayer::validate(const ITensorInfo *input, const ITensorInfo *output)
{
	    return Status{};

}

void NEResizeLayer::run()
{
	    NEScheduler::get().schedule(&_resize_kernel, Window::DimZ);
}

