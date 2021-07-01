#include "arm_compute/runtime/NEON/functions/NEABMMatrixMultiply.h"

#include "arm_compute/core/CPP/Validate.h"

#include "arm_compute/core/Error.h"

#include "arm_compute/core/Helpers.h"

#include "arm_compute/core/ITensor.h"

#include "arm_compute/core/TensorInfo.h"

#include "arm_compute/core/Types.h"

#include "arm_compute/core/Validate.h"

#include "arm_compute/runtime/NEON/NEScheduler.h"

#include "arm_compute/runtime/CPP/CPPScheduler.h"

#include "arm_compute/runtime/TensorAllocator.h"

#include <cmath>

#include <chrono>



namespace arm_compute

{

    NEABMMatrixMultiply::NEABMMatrixMultiply(std::shared_ptr<IMemoryManager> memory_manager)

            :_memory_group(memory_manager),_matrix_multiply_kernel(),_mm_last_kernel(),  _alpha_scale_func(), _add_bias_kernel(), _activation_func(), 

            _tmp_d(),_run_addition(false), _run_bias_addition(false), _run_activation(false), _is_prepared(false), _t_matrix_multiply(0), _t_mmlast(0)

    {

    }



    void NEABMMatrixMultiply::configure(const ITensor *input,const ITensor *Q_table, const ITensor *WT_buffer, const  ITensor *bias,  ITensor *output, unsigned int precision[],

                    float alpha, float beta, const GEMMInfo &gemm_info, unsigned int num_groups)

    {

        ARM_COMPUTE_ERROR_THROW_ON(NEABMMatrixMultiply::validate(input->info(),Q_table->info(),WT_buffer->info(),(bias != nullptr) ? bias->info() : nullptr,output->info(),alpha, beta, gemm_info));

        _run_bias_addition                =false;

        _run_addition                     =false;

        _run_activation                   =false;

        ITensor *multiply_output_to_use = output;

        if(_run_bias_addition)

        {

            multiply_output_to_use = &_tmp_d;

            _memory_group.manage(&_tmp_d);

        }

        _matrix_multiply_kernel.configure(input,Q_table,WT_buffer,bias,multiply_output_to_use,precision,alpha,num_groups);

        _mm_last_kernel.configure(input,Q_table,WT_buffer,bias,multiply_output_to_use,precision,alpha,num_groups);

        if(_run_bias_addition)

        {

            printf("bias_configure\n");

            _add_bias_kernel.configure(multiply_output_to_use, bias, output, ConvertPolicy::SATURATE);

            _tmp_d.allocator()->allocate();

        }

        if(_run_addition)

        {

            printf("addition_configure\n");

        }

        const ActivationLayerInfo &activation = gemm_info.activation_info();

        if(_run_activation)

        {

            printf("activation_configure\n");

            _activation_func.configure(output, nullptr, activation);

        }



    }





    Status NEABMMatrixMultiply::validate(const ITensorInfo *input, const ITensorInfo *Q_table, const ITensorInfo *WT_buffer, const ITensorInfo *bias, const ITensorInfo *output, 

                    float alpha, float beta, const GEMMInfo &gemm_info,unsigned int num_groups)



    {

            return Status{};

    }







    std::pair<double, double> NEABMMatrixMultiply::print_time()

    {

        std::pair<double, double> p(_t_matrix_multiply, _t_mmlast);

        return p;



    }







    void NEABMMatrixMultiply::run()



    {



        auto begin3=std::chrono::high_resolution_clock::now();



        /*



        arm_compute::CPPScheduler::Hints hint(Window::DimY, arm_compute::IScheduler::StrategyHint::DYNAMIC);



        NEScheduler::get().schedule(&_matrix_multiply_kernel, hint);



        */

        

        

        NEScheduler::get().schedule(&_matrix_multiply_kernel, Window::DimY);

        

        



        auto end3=std::chrono::high_resolution_clock::now();



        



        auto begin4=std::chrono::high_resolution_clock::now();



        NEScheduler::get().schedule(&_mm_last_kernel,Window::DimY );



        auto end4=std::chrono::high_resolution_clock::now();







        _t_matrix_multiply = std::chrono::duration_cast<std::chrono::duration<double>>(end3 - begin3).count();



        _t_mmlast = std::chrono::duration_cast<std::chrono::duration<double>>(end4 - begin4).count();







        if(_run_bias_addition)







        {







            printf("bias_run\n");







            NEScheduler::get().schedule(&_add_bias_kernel, Window::DimY);







        }







        if(_run_addition)







        {







            printf("addition_run\n");







        }







        if(_run_activation)







        {







            printf("activation_run\n");







            _activation_func.run();







        }



    }



}/*end namespace arm_compute;*/






