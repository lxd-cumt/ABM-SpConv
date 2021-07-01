#ifndef _ARM_COMPUTE_NES8TOF32LAYER_H_

#define _ARM_COMPUTE_NES8TOF32LAYER_H_



#include "arm_compute/runtime/IFunction.h"

#include "arm_compute/core/NEON/kernels/NES8toF32Kernel.h"

#include "arm_compute/core/Types.h"

#include "arm_compute/runtime/MemoryGroup.h"

#include "arm_compute/runtime/Tensor.h"

#include <memory>

using namespace std;

namespace arm_compute
{
class ITensor;

class NES8toF32Layer : public IFunction

{

public:

    /** Constructor */

    NES8toF32Layer();

    /** Prevent instances of this class from being copied (As this class contains pointers) */

    NES8toF32Layer(const NES8toF32Layer &) = delete;

    /** Default move constructor */

    NES8toF32Layer(NES8toF32Layer &&) = default;

    /** Prevent instances of this class from being copied (As this class contains pointers) */

    NES8toF32Layer &operator=(const NES8toF32Layer &) = delete;

    /** Default move assignment operator */

    NES8toF32Layer &operator=(NES8toF32Layer &&) = default;



    void configure(const ITensor *input, ITensor *output);



    static Status validate(const ITensorInfo *input, const ITensorInfo *output);



    // Inherited methods overridden:

    void run() override;



private:

    NES8toF32Kernel _s8_to_f32_kernel;

};

}
#endif
