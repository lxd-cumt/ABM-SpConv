#ifndef _ARM_COMPUTE_NEABMMATRIXMULTIPLY_H_







#define _ARM_COMPUTE_NEABMMATRIXMULTIPLY_H















#include "arm_compute/core/NEON/kernels/NEArithmeticAdditionKernel.h"







#include "arm_compute/core/NEON/kernels/NEABMMatrixMultiplyKernel.h"







#include "arm_compute/core/NEON/kernels/NEABMMMLastKernel.h"







#include "arm_compute/core/NEON/kernels/NEGEMMMatrixAdditionKernel.h"







#include "arm_compute/runtime/IFunction.h"







#include "arm_compute/runtime/IMemoryManager.h"







#include "arm_compute/runtime/MemoryGroup.h"







#include "arm_compute/runtime/NEON/functions/NEActivationLayer.h"







#include "arm_compute/runtime/Tensor.h"















#include<vector>







#include<map>







using namespace std;







namespace arm_compute







{















class NEABMMatrixMultiply : public IFunction







{







public:







    /** Constructor */







    NEABMMatrixMultiply(std::shared_ptr<IMemoryManager> memory_manager=nullptr );







    /** Prevent instances of this class from being copied (As this class contains pointers) */







    NEABMMatrixMultiply(const NEABMMatrixMultiply &) = delete;







    /** Default move constructor */







    NEABMMatrixMultiply(NEABMMatrixMultiply &&) = default;







    /** Prevent instances of this class from being copied (As this class contains pointers) */







    NEABMMatrixMultiply &operator=(const NEABMMatrixMultiply &) = delete;







    /** Default move assignment operator */







    NEABMMatrixMultiply &operator=(NEABMMatrixMultiply &&) = default;















    void configure(const ITensor *input, const ITensor *Q_table, const ITensor *WT_buffer, const  ITensor *bias,  ITensor *output, unsigned int precision[],







                    float alpha, float beta, const GEMMInfo &gemm_info, unsigned int num_groups=1);















    static Status validate(const ITensorInfo *input, const ITensorInfo *Q_table, const ITensorInfo *WT_buffer, const ITensorInfo *bias, const ITensorInfo *output, 







                    float alpha, float beta, const GEMMInfo &gemm_info, unsigned int num_groups);







    std::pair<double, double> print_time();







    void run() override;







    /*void prepare() override;*/







private:







    MemoryGroup                _memory_group;







    NEABMMatrixMultiplyKernel _matrix_multiply_kernel;



    NEABMMMLastKernel _mm_last_kernel;



    NEActivationLayer          _alpha_scale_func;



    NEArithmeticAdditionKernel _add_bias_kernel;



    NEActivationLayer          _activation_func;







    Tensor         _tmp_d;



    bool           _run_addition;



    bool           _run_bias_addition;



    bool           _run_activation;



    bool           _is_prepared;







    double _t_matrix_multiply=0;



    double _t_mmlast=0;







};



}



#endif
