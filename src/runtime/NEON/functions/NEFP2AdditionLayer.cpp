#include "arm_compute/runtime/NEON/functions/NEFP2AdditionLayer.h"


#include "arm_compute/core/Size2D.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "utils/ImageLoader.h"
#include "utils/Utils.h"

using namespace arm_compute;

NEFP2AdditionLayer::NEFP2AdditionLayer():_add_kernel()
{
}

void NEFP2AdditionLayer::configure(const ITensor *input_a, const ITensor *input_b, ITensor *output,  int *fp)

{
	    _add_kernel.configure(input_a,input_b,output,fp);
}

Status NEFP2AdditionLayer::validate(const ITensorInfo *input_a, const ITensorInfo *input_b, const ITensorInfo *output)
{
	    
	    return Status{};

}

void NEFP2AdditionLayer::run()
{
	    NEScheduler::get().schedule(&_add_kernel, Window::DimZ);
}

