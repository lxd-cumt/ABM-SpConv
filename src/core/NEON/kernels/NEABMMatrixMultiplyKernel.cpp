#include "arm_compute/core/NEON/kernels/NEABMMatrixMultiplyKernel.h"

#include "arm_compute/core/AccessWindowStatic.h"

#include "arm_compute/core/CPP/Validate.h"

#include "arm_compute/core/Error.h"

#include "arm_compute/core/Helpers.h"

#include "arm_compute/core/IAccessWindow.h"

#include "arm_compute/core/ITensor.h"

#include "arm_compute/core/NEON/NEFixedPoint.h"

#include "arm_compute/core/TensorInfo.h"



#include "arm_compute/core/Utils.h"

#include "arm_compute/core/Validate.h"

#include "arm_compute/core/Window.h"

#include "arm_compute/core/utils/helpers/float_ops.h"

#include "arm_compute/core/utils/misc/ShapeCalculator.h"

#include "arm_compute/core/NEON/NEFixedPoint.h"

#include "arm_compute/core/utils/helpers/float_ops.h"







#include <arm_neon.h>

#include <cstddef>

#include <cstdint>

#include <tuple>

#include <cmath>



/*

#include <mutex>

std::mutex m;

*/





#define num_processed_kernels 16

/*#define num_processed_kernels 4*/



using namespace arm_compute;



namespace

{



inline Status validate_arguments(const ITensorInfo *input, const ITensorInfo *Q_table, const ITensorInfo *WT_buffer,

                    const ITensorInfo *bias, const ITensorInfo *output, float alpha, unsigned int num_groups)



{

        return Status{};

}



inline int8_t judge_overflow(float ope)

{

        int8_t real_ope=0;

        if(ope>127){

                real_ope=127;

        }

        else if(ope<-128){

                real_ope=-128;

        }

        else{

                real_ope=(int8_t)ope;

        }

        return real_ope;

}





inline std::pair<Status,Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *output)



{       

        size_t kernels=input->dimension(1);

        size_t single_core_kernels=kernels/4;

        // num_processed_kernels=single_core_kernels;



        Window window=calculate_max_window(*input,Steps());

        if(input->dimension(1)<num_processed_kernels)

        {

                window.set(Window::DimY, Window::Dimension(0, input->dimension(1),input->dimension(1)));   

        }

        else{

                 window.set(Window::DimY, Window::Dimension(0, input->dimension(1),single_core_kernels));     

        }

        if(input->dimension(0)<32)

        {

                window.set(Window::DimX, Window::Dimension(0, 0, 0) );  

        }

        else{

        int end_blocks=input->dimension(0)/32;

        window.set(Window::DimX, Window::Dimension(0, end_blocks*32 , 32));         

        }     

        window.set(Window::DimZ, Window::Dimension(0, 1, 1));

         return std::make_pair(Status{}, window);

}



}



NEABMMatrixMultiplyKernel::NEABMMatrixMultiplyKernel()

        :_input(nullptr),_Q_table(nullptr),_WT_buffer(nullptr),_bias(nullptr), _output(nullptr), _num_groups(0)

        ,_a_mul_value(0),_b_mul_value(0),_alpha(1.0f),_kernel_last(0)

{

}

void NEABMMatrixMultiplyKernel::configure(const ITensor *input, const ITensor *Q_table, const ITensor *WT_buffer,const ITensor *bias, 

                    ITensor *output,unsigned int precision[],float alpha, unsigned int num_groups)

{

        ARM_COMPUTE_ERROR_ON_NULLPTR(input,output);

        _input=input;

        _Q_table=Q_table;

        _WT_buffer=WT_buffer;

        _bias=(bias!=nullptr)?bias:nullptr;

        _output=output;

        _num_groups=num_groups;



        unsigned int _weights_precision=precision[0];

        unsigned int _bias_precision=precision[1];

        unsigned int _input_precision=precision[2];

        unsigned int _output_precision=precision[3];

        int temp1=_output_precision-_weights_precision-_input_precision;

        _a_mul_value=pow(2,temp1);

        int temp2=_output_precision-_bias_precision;

        _b_mul_value=pow(2,temp2);



        _alpha=alpha;



        long long int kernel_numbers=output->info()->dimension(1);

        _kernel_last=kernel_numbers%num_processed_kernels;

        _kernel_last=0;



        auto win_config = validate_and_configure_window(output->info(),output->info());

        ARM_COMPUTE_ERROR_THROW_ON(win_config.first);

        INEKernel::configure(win_config.second);

}







Status NEABMMatrixMultiplyKernel::validate(const ITensorInfo *input,const ITensorInfo *Q_table, const ITensorInfo *WT_buffer, const ITensorInfo *bias,

                    const ITensorInfo *output, float alpha, unsigned int num_groups)



{

        /*ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, quantity, location, alpha, reshape_info));*/

        /*ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(input->clone().get(), output->clone().get()).first);*/

        return Status{};

}







void NEABMMatrixMultiplyKernel::run(const Window &window, const ThreadInfo &info)



{



        /*std::unique_lock<std::mutex> lock(m);

        std::cout<<"Thread: "<<info.thread_id<<std::endl;

        std::cout<<window.x().start()<<" "<<window.x().end()<<" "<<window.x().step();

        std::cout<<"Y-Dimension: "<<window.y().start()<<" "<<window.y().end()<<" "<<window.y().step();

        std::cout<<window.z().start()<<" "<<window.z().end()<<" "<<window.z().step();

        lock.unlock();*/



        ARM_COMPUTE_UNUSED(info);

        ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);

        ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);



        Window wout(window);

        wout.set(Window::DimX, Window::Dimension(0, 0, 0));

        wout.set(Window::DimY, Window::Dimension(0, 0, 0));

        wout.set(Window::DimZ, Window::Dimension(0, 0, 0));



        Iterator out(_output,wout);



        int16x8_t max_value=vdupq_n_s16(127);

        int16x8_t min_value=vdupq_n_s16(-128);



        uint8_t *Q_table_buffer=_Q_table->buffer();

        const unsigned int Q_table_stride_z=_Q_table->info()->strides_in_bytes().z();

        uint8_t *WT_buffer_buffer=_WT_buffer->buffer();

        const unsigned int WT_buffer_stride_z=_WT_buffer->info()->strides_in_bytes().z();

        uint8_t *bias_buffer=(_bias!=nullptr)?_bias->buffer():nullptr;



        float32x4_t a_mul=vdupq_n_f32(_a_mul_value);

        float32x4_t b_mul=vdupq_n_f32(_b_mul_value);



        size_t single_core_kernels=_output->info()->dimension(1)/4;



        execute_window_loop(window, [&](const Coordinates & id){

                  

                size_t x_dimension=id[0], y_dimension=id[1]; 



                uint8_t *temp_input_ptr=_input->buffer();                            

                size_t input_ptr_offset=_input->info()->offset_element_in_bytes(Coordinates(0,x_dimension/8,0));

                size_t input_ptr_offset1=_input->info()->offset_element_in_bytes(Coordinates(0,x_dimension/8+1,0));

                size_t input_ptr_offset2=_input->info()->offset_element_in_bytes(Coordinates(0,x_dimension/8+2,0));

                size_t input_ptr_offset3=_input->info()->offset_element_in_bytes(Coordinates(0,x_dimension/8+3,0));

                uint8_t *input_addr=temp_input_ptr+input_ptr_offset;

                uint8_t *input_addr1=temp_input_ptr+input_ptr_offset1;

                uint8_t *input_addr2=temp_input_ptr+input_ptr_offset2;

                uint8_t *input_addr3=temp_input_ptr+input_ptr_offset3;



                int output_x_dimension=x_dimension;

                size_t output_ptr_offset=0;

                output_ptr_offset=_output->info()->offset_element_in_bytes(Coordinates(output_x_dimension,0,0));



                unsigned int start_kernel=0, end_kernel=0;

                if(_kernel_last==0){

                        start_kernel=y_dimension;

                        end_kernel=y_dimension+single_core_kernels;

                }

                else{

                        if(y_dimension/num_processed_kernels==_output->info()->dimension(1)/num_processed_kernels)

                        {

                                start_kernel=y_dimension;

                                end_kernel=_output->info()->dimension(1);

                        }

                        else{

                                start_kernel=y_dimension;

                                end_kernel=y_dimension+num_processed_kernels;

                        }

                }

                  for(unsigned int j=start_kernel; j<end_kernel; j++)

                  {  

                        unsigned int id_groups=0;unsigned int group_step=_output->info()->dimension(1)/_num_groups;

                        for( unsigned int i=0;i<_output->info()->dimension(1);i+=group_step)

                        {

                                if(j>=i&&j<(i+group_step))

                                {

                                        break;

                                }

                                else{

                                        id_groups++;

                                }

                        }

                        uint8_t* input_group_addr=input_addr+id_groups*(_input->info()->dimension(0)/_num_groups);

                        uint8_t* input_group_addr1=input_addr1+id_groups*(_input->info()->dimension(0)/_num_groups);

                        uint8_t* input_group_addr2=input_addr2+id_groups*(_input->info()->dimension(0)/_num_groups);

                        uint8_t* input_group_addr3=input_addr3+id_groups*(_input->info()->dimension(0)/_num_groups);

                        int16x8_t acc=vdupq_n_s16(0.f);

                        int16x8_t acc1=vdupq_n_s16(0.f);

                        int16x8_t acc2=vdupq_n_s16(0.f);

                        int16x8_t acc3=vdupq_n_s16(0.f);



                        uint8_t *Q_table_start_ptr=Q_table_buffer+j*Q_table_stride_z;                      

                        uint8_t *WT_buffer_start_ptr=WT_buffer_buffer+j*WT_buffer_stride_z;

                        int16_t *Q_table_ptr = reinterpret_cast<int16_t *>(Q_table_start_ptr);

                        uint16_t *WT_buffer_ptr=reinterpret_cast<uint16_t *>(WT_buffer_start_ptr);



                        short value_count=*Q_table_ptr;Q_table_ptr++;



                        while(value_count--)

                        {

                                int16x8_t onesum=vdupq_n_s16(0.f);

                                int16x8_t onesum1=vdupq_n_s16(0.f);

                                int16x8_t onesum2=vdupq_n_s16(0.f);

                                int16x8_t onesum3=vdupq_n_s16(0.f);

                                unsigned short result=0;

                                int16x8_t input_value=vdupq_n_s16(0.f);

                                int16x8_t input_value1=vdupq_n_s16(0.f);

                                int16x8_t input_value2=vdupq_n_s16(0.f);

                                int16x8_t input_value3=vdupq_n_s16(0.f);

                                signed short q=(*Q_table_ptr);Q_table_ptr++;

                                int8_t temp_value=(q&0xff00)>>8;signed short temp_quantity=(q&0x00ff);

                                while(temp_quantity>0)

                                {

                                        result=(*WT_buffer_ptr);WT_buffer_ptr++;



                                        input_value=vmovl_s8(vld1_s8(reinterpret_cast<int8_t *>(input_group_addr+result)));

                                        onesum=vaddq_s16(onesum,input_value);

                                        input_value1=vmovl_s8(vld1_s8(reinterpret_cast<int8_t *>(input_group_addr1+result)));

                                        onesum1=vaddq_s16(onesum1,input_value1);

                                        input_value2=vmovl_s8(vld1_s8(reinterpret_cast<int8_t *>(input_group_addr2+result)));

                                        onesum2=vaddq_s16(onesum2,input_value2);

                                        input_value3=vmovl_s8(vld1_s8(reinterpret_cast<int8_t *>(input_group_addr3+result)));

                                        onesum3=vaddq_s16(onesum3,input_value3);



                                        temp_quantity--;

                                }

                                int16_t real_value=(int16_t)(temp_value);     

                                int16x8_t onevalue = vdupq_n_s16(real_value);

                                acc=vaddq_s16(acc,vmulq_s16(onesum,onevalue));          

                                acc1=vaddq_s16(acc1,vmulq_s16(onesum1,onevalue));          

                                acc2=vaddq_s16(acc2,vmulq_s16(onesum2,onevalue));          

                                acc3=vaddq_s16(acc3,vmulq_s16(onesum3,onevalue));          

                        }



                        int8_t bias_value=0;

                        if(bias_buffer!=nullptr){

                                        int8_t *bias_ptr= reinterpret_cast<int8_t *>(bias_buffer+(int)j);

                                        bias_value=(*bias_ptr);

                                }

                        int16_t real_bias=(int16_t)bias_value;

                        int16x8_t acc_bias=vdupq_n_s16(real_bias);



                        int16x8_t ope_a=acc;

                        int16x8_t ope_a1=acc1;

                        int16x8_t ope_a2=acc2;

                        int16x8_t ope_a3=acc3;

                        int16x8_t ope_b=acc_bias;



                        int16x4_t ope_a_high=vget_high_s16(ope_a);

                        int16x4_t ope_a_low=vget_low_s16(ope_a);

                        int16x4_t ope_a1_high=vget_high_s16(ope_a1);

                        int16x4_t ope_a1_low=vget_low_s16(ope_a1);

                        int16x4_t ope_a2_high=vget_high_s16(ope_a2);

                        int16x4_t ope_a2_low=vget_low_s16(ope_a2);

                        int16x4_t ope_a3_high=vget_high_s16(ope_a3);

                        int16x4_t ope_a3_low=vget_low_s16(ope_a3);

                        int16x4_t ope_b_high=vget_high_s16(ope_b);

                        int16x4_t ope_b_low=vget_low_s16(ope_b);



                        int32x4_t ope_a_high_long=vmovl_s16(ope_a_high);

                        int32x4_t ope_a_low_long=vmovl_s16(ope_a_low);

                        int32x4_t ope_a1_high_long=vmovl_s16(ope_a1_high);

                        int32x4_t ope_a1_low_long=vmovl_s16(ope_a1_low);

                        int32x4_t ope_a2_high_long=vmovl_s16(ope_a2_high);

                        int32x4_t ope_a2_low_long=vmovl_s16(ope_a2_low);

                        int32x4_t ope_a3_high_long=vmovl_s16(ope_a3_high);

                        int32x4_t ope_a3_low_long=vmovl_s16(ope_a3_low);

                        int32x4_t ope_b_high_long=vmovl_s16(ope_b_high);

                        int32x4_t ope_b_low_long=vmovl_s16(ope_b_low);



                        float32x4_t a_high=vcvtq_f32_s32(ope_a_high_long);

                        float32x4_t a_low=vcvtq_f32_s32(ope_a_low_long);

                        float32x4_t a1_high=vcvtq_f32_s32(ope_a1_high_long);

                        float32x4_t a1_low=vcvtq_f32_s32(ope_a1_low_long);

                        float32x4_t a2_high=vcvtq_f32_s32(ope_a2_high_long);

                        float32x4_t a2_low=vcvtq_f32_s32(ope_a2_low_long);

                        float32x4_t a3_high=vcvtq_f32_s32(ope_a3_high_long);

                        float32x4_t a3_low=vcvtq_f32_s32(ope_a3_low_long);

                        float32x4_t b_high=vcvtq_f32_s32(ope_b_high_long);

                        float32x4_t b_low=vcvtq_f32_s32(ope_b_low_long);





                        a_high=vmulq_f32(a_high,a_mul);

                        a_low=vmulq_f32(a_low,a_mul);

                        a1_high=vmulq_f32(a1_high,a_mul);

                        a1_low=vmulq_f32(a1_low,a_mul);

                        a2_high=vmulq_f32(a2_high,a_mul);

                        a2_low=vmulq_f32(a2_low,a_mul);

                        a3_high=vmulq_f32(a3_high,a_mul);

                        a3_low=vmulq_f32(a3_low,a_mul);

                        b_high=vmulq_f32(b_high,b_mul);

                        b_low=vmulq_f32(b_low,b_mul);



                        float32x4_t res1=vaddq_f32(a_high,b_high);

                        float32x4_t res2=vaddq_f32(a_low,b_low);

                        float32x4_t res1_1=vaddq_f32(a1_high,b_high);

                        float32x4_t res2_1=vaddq_f32(a1_low,b_low);

                        float32x4_t res1_2=vaddq_f32(a2_high,b_high);

                        float32x4_t res2_2=vaddq_f32(a2_low,b_low);

                        float32x4_t res1_3=vaddq_f32(a3_high,b_high);

                        float32x4_t res2_3=vaddq_f32(a3_low,b_low);



                        int32x4_t res1_s32=vcvtq_s32_f32(res1);

                        int32x4_t res2_s32=vcvtq_s32_f32(res2);

                        int32x4_t res1_1_s32=vcvtq_s32_f32(res1_1);

                        int32x4_t res2_1_s32=vcvtq_s32_f32(res2_1);

                        int32x4_t res1_2_s32=vcvtq_s32_f32(res1_2);

                        int32x4_t res2_2_s32=vcvtq_s32_f32(res2_2);

                        int32x4_t res1_3_s32=vcvtq_s32_f32(res1_3);

                        int32x4_t res2_3_s32=vcvtq_s32_f32(res2_3);



                        int16x4_t res1_s16=vmovn_s32(res1_s32);

                        int16x4_t res2_s16=vmovn_s32(res2_s32);

                        int16x4_t res1_1_s16=vmovn_s32(res1_1_s32);

                        int16x4_t res2_1_s16=vmovn_s32(res2_1_s32);

                        int16x4_t res1_2_s16=vmovn_s32(res1_2_s32);

                        int16x4_t res2_2_s16=vmovn_s32(res2_2_s32);

                        int16x4_t res1_3_s16=vmovn_s32(res1_3_s32);

                        int16x4_t res2_3_s16=vmovn_s32(res2_3_s32);

                

                        int16x8_t res_c=vcombine_s16(res2_s16,res1_s16);

                        int16x8_t res_c1=vcombine_s16(res2_1_s16,res1_1_s16);

                        int16x8_t res_c2=vcombine_s16(res2_2_s16,res1_2_s16);

                        int16x8_t res_c3=vcombine_s16(res2_3_s16,res1_3_s16);



                        res_c=vminq_s16(max_value,vmaxq_s16(min_value,res_c));

                        res_c1=vminq_s16(max_value,vmaxq_s16(min_value,res_c1));

                        res_c2=vminq_s16(max_value,vmaxq_s16(min_value,res_c2));

                        res_c3=vminq_s16(max_value,vmaxq_s16(min_value,res_c3));



                        int8x8_t real_res=vmovn_s16(res_c);

                        int8x8_t real_res1=vmovn_s16(res_c1);

                        int8x8_t real_res2=vmovn_s16(res_c2);

                        int8x8_t real_res3=vmovn_s16(res_c3);



                        int8_t *out_addr=reinterpret_cast<int8_t*>(_output->buffer()+j*_output->info()->dimension(0)+output_ptr_offset);

                        vst1_s8(out_addr,real_res);

                        vst1_s8(out_addr+8,real_res1);

                        vst1_s8(out_addr+16,real_res2);

                        vst1_s8(out_addr+24,real_res3);

                  }

        },out);





        

}
