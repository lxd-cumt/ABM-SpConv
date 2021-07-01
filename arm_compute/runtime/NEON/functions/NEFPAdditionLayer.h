#ifndef _ARM_COMPUTE_NEFPADDITIONLAYER_H_

#define _ARM_COMPUTE_NEFPADDITIONLAYER_H_



#include "arm_compute/runtime/IFunction.h"

#include "arm_compute/core/NEON/kernels/NEFPAdditionKernel.h"

#include "arm_compute/core/Types.h"

#include "arm_compute/runtime/MemoryGroup.h"

#include "arm_compute/runtime/Tensor.h"

#include <memory>

using namespace std;

namespace arm_compute
{
class ITensor;

class NEFPAdditionLayer : public IFunction

{

public:

    /** Constructor */

    NEFPAdditionLayer();

    /** Prevent instances of this class from being copied (As this class contains pointers) */

    NEFPAdditionLayer(const NEFPAdditionLayer &) = delete;

    /** Default move constructor */

    NEFPAdditionLayer(NEFPAdditionLayer &&) = default;

    /** Prevent instances of this class from being copied (As this class contains pointers) */

    NEFPAdditionLayer &operator=(const NEFPAdditionLayer &) = delete;

    /** Default move assignment operator */

    NEFPAdditionLayer &operator=(NEFPAdditionLayer &&) = default;



    void configure(const ITensor *input_a, const ITensor *input_b, ITensor *output,  int *fp);



    static Status validate(const ITensorInfo *input_a, const ITensorInfo *input_b, const ITensorInfo *output);



    // Inherited methods overridden:

    void run() override;



private:

    NEFPAdditionKernel _add_kernel;

};

}
#endif
