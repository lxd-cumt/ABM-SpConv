#ifndef __ARM_COMPUTE_NEABMMATRIXMULTIPLYKERNEL_H__



#define __ARM_COMPUTE_NEABMMATRIXMULTIPLYKERNEL_H__







#include "arm_compute/core/NEON/INEKernel.h"



#include "arm_compute/core/Types.h"







#include<vector>



#include<map>







using namespace std;







namespace arm_compute



{



class ITensor;







class NEABMMatrixMultiplyKernel : public INEKernel



{



public:



    const char *name() const override



    {



        return "NEABMMatrixMultiplyKernel";



    }



    /** Constructor */



    NEABMMatrixMultiplyKernel();



    /** Prevent instances of this class from being copied (As this class contains pointers) */



    NEABMMatrixMultiplyKernel(const NEABMMatrixMultiplyKernel &) = delete;



    /** Prevent instances of this class from being copied (As this class contains pointers) */



    NEABMMatrixMultiplyKernel &operator=(const NEABMMatrixMultiplyKernel &) = delete;



    /** Allow instances of this class to be moved */



    NEABMMatrixMultiplyKernel(NEABMMatrixMultiplyKernel &&) = default;



    /** Allow instances of this class to be moved */



    NEABMMatrixMultiplyKernel &operator=(NEABMMatrixMultiplyKernel &&) = default;







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



/*   

    unsigned int _weights_precision;



    unsigned int _bias_precision;



    unsigned int _input_precision;

    

    unsigned int _output_precision;



*/



    double _a_mul_value;



    double _b_mul_value;



/*

    unsigned int _x_index;



    unsigned int _y_index;



    unsigned int _z_index;



    unsigned int _y_index_bit;



    unsigned int _x_index_bit;



*/



    float          _alpha;



    int _kernel_last;





};



}



#endif
