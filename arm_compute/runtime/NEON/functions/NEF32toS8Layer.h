#ifndef _ARM_COMPUTE_NEF32TOS8LAYER_H_

#define _ARM_COMPUTE_NEF32TOS8LAYER_H_



#include "arm_compute/runtime/IFunction.h"

#include "arm_compute/core/NEON/kernels/NEF32toS8Kernel.h"

#include "arm_compute/core/Types.h"

#include "arm_compute/runtime/MemoryGroup.h"

#include "arm_compute/runtime/Tensor.h"

#include <memory>

using namespace std;

namespace arm_compute
{
class ITensor;

class NEF32toS8Layer : public IFunction

{

public:

    /** Constructor */

    NEF32toS8Layer();

    /** Prevent instances of this class from being copied (As this class contains pointers) */

    NEF32toS8Layer(const NEF32toS8Layer &) = delete;

    /** Default move constructor */

    NEF32toS8Layer(NEF32toS8Layer &&) = default;

    /** Prevent instances of this class from being copied (As this class contains pointers) */

    NEF32toS8Layer &operator=(const NEF32toS8Layer &) = delete;

    /** Default move assignment operator */

    NEF32toS8Layer &operator=(NEF32toS8Layer &&) = default;



    void configure(const ITensor *input, ITensor *output);



    static Status validate(const ITensorInfo *input, const ITensorInfo *output);



    // Inherited methods overridden:

    void run() override;



private:

    NEF32toS8Kernel _f32_to_s8_kernel;

};

}
#endif
