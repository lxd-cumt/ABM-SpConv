#ifndef __ARM_COMPUTE_NEABMMMLASTKERNEL_H__

#define __ARM_COMPUTE_NEABMMMLASTKERNEL_H__



#include "arm_compute/core/NEON/INEKernel.h"

#include "arm_compute/core/Types.h"



#include<vector>

#include<map>



using namespace std;



namespace arm_compute

{

class ITensor;



class NEABMMMLastKernel : public INEKernel

{

public:

    const char *name() const override

    {

        return "NEABMMMLastKernel";

    }

    /** Constructor */

    NEABMMMLastKernel();

    /** Prevent instances of this class from being copied (As this class contains pointers) */

    NEABMMMLastKernel(const NEABMMMLastKernel &) = delete;

    /** Prevent instances of this class from being copied (As this class contains pointers) */

    NEABMMMLastKernel &operator=(const NEABMMMLastKernel &) = delete;

    /** Allow instances of this class to be moved */

    NEABMMMLastKernel(NEABMMMLastKernel &&) = default;

    /** Allow instances of this class to be moved */

    NEABMMMLastKernel &operator=(NEABMMMLastKernel &&) = default;



    void configure(const ITensor *input, const ITensor *Q_table, const ITensor *WT_buffer,const ITensor *bias, 

                    ITensor *output,unsigned int precision[],float alpha, unsigned int num_groups=1);

    

    static Status validate(const ITensorInfo *input, const ITensorInfo *Q_table, const ITensorInfo *WT_buffer, const ITensorInfo *bias, 

                    const ITensorInfo *output, float alpha, unsigned int num_groups);





    void run(const Window &window, const ThreadInfo &info) override;



private:

    const ITensor *_input;

    const ITensor *_Q_table;

    const ITensor *_WT_buffer;

    const ITensor *_bias;

    ITensor       *_output;

    unsigned int  _num_groups;


    double _a_mul_value;

    double _b_mul_value;


    float          _alpha;

    int _convolution_last;
    
    int _kernel_last;


};

}

#endif
