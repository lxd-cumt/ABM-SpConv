#ifndef __ARM_COMPUTE_NEABMINTERLEAVEBLOCKKERNEL_H__

#define __ARM_COMPUTE_NEABMINTERLEAVEBLOCKKERNEL_H__



#include "arm_compute/core/NEON/INEKernel.h"

#include "arm_compute/core/Types.h"



#include<vector>

#include<map>



using namespace std;



namespace arm_compute

{

class ITensor;



class NEABMInterleaveBlockKernel : public INEKernel

{

public:

    const char *name() const override

    {

        return "NEABMInterleaveBlockKernel";

    }

    /** Constructor */

    NEABMInterleaveBlockKernel();

    /** Prevent instances of this class from being copied (As this class contains pointers) */

    NEABMInterleaveBlockKernel(const NEABMInterleaveBlockKernel &) = delete;

    /** Prevent instances of this class from being copied (As this class contains pointers) */

    NEABMInterleaveBlockKernel &operator=(const NEABMInterleaveBlockKernel &) = delete;

    /** Allow instances of this class to be moved */

    NEABMInterleaveBlockKernel(NEABMInterleaveBlockKernel &&) = default;

    /** Allow instances of this class to be moved */

    NEABMInterleaveBlockKernel &operator=(NEABMInterleaveBlockKernel &&) = default;



    void configure(const ITensor *input,  ITensor *output, unsigned int num_convs);

    

    static Status validate(const ITensorInfo *input, const ITensorInfo *output);



    void run(const Window &window, const ThreadInfo &info) override;



private:

    const ITensor *_input;

    ITensor *_output;

    unsigned int _num_convs;
};

}

#endif