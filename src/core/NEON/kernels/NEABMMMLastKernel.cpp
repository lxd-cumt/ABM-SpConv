#include "arm_compute/core/NEON/kernels/NEABMMMLastKernel.h"
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

#define num_processed_kernels 16

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
        Window window=calculate_max_window(*input,Steps());
        if(input->dimension(1)<num_processed_kernels)
        {
                window.set(Window::DimY, Window::Dimension(0, input->dimension(1),input->dimension(1)));   
        }
        else{
                 window.set(Window::DimY, Window::Dimension(0, input->dimension(1),num_processed_kernels));     
        }
        int end_blocks=input->dimension(0)/32;
        window.set(Window::DimX, Window::Dimension(end_blocks*32, input->dimension(0),8));          
        window.set(Window::DimZ, Window::Dimension(0, 1, 1));
         return std::make_pair(Status{}, window);
}

}

NEABMMMLastKernel::NEABMMMLastKernel()
        :_input(nullptr),_Q_table(nullptr),_WT_buffer(nullptr),_bias(nullptr), _output(nullptr), _num_groups(0)
        ,_a_mul_value(0),_b_mul_value(0),_alpha(1.0f),_convolution_last(0),_kernel_last(0)
{
}
void NEABMMMLastKernel::configure(const ITensor *input, const ITensor *Q_table, const ITensor *WT_buffer,const ITensor *bias, 
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

        long long int convolution_steps=output->info()->dimension(0);
        long long int kernel_numbers=output->info()->dimension(1);
        _convolution_last=convolution_steps%8;
        _kernel_last=kernel_numbers%num_processed_kernels;

        auto win_config = validate_and_configure_window(output->info(),output->info());
        ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
        INEKernel::configure(win_config.second);
}



Status NEABMMMLastKernel::validate(const ITensorInfo *input,const ITensorInfo *Q_table, const ITensorInfo *WT_buffer, const ITensorInfo *bias,
                    const ITensorInfo *output, float alpha, unsigned int num_groups)

{
        return Status{};
}



void NEABMMMLastKernel::run(const Window &window, const ThreadInfo &info)
{
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

        execute_window_loop(window, [&](const Coordinates & id){
                  
                size_t x_dimension=id[0], y_dimension=id[1]; 

                uint8_t *temp_input_ptr=_input->buffer();                            
                size_t input_ptr_offset=_input->info()->offset_element_in_bytes(Coordinates(0,x_dimension/8,0));
                uint8_t *input_addr=temp_input_ptr+input_ptr_offset;

                uint8_t *temp_output_ptr=_output->buffer();
                int output_x_dimension=x_dimension;
                size_t output_ptr_offset=0;
                output_ptr_offset=_output->info()->offset_element_in_bytes(Coordinates(output_x_dimension,0,0));

                unsigned int start_kernel=0, end_kernel=0;
                if(_kernel_last==0){
                        start_kernel=y_dimension;
                        end_kernel=y_dimension+num_processed_kernels;
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
                        
                        int16x8_t acc=vdupq_n_s16(0.f);

                        uint8_t *Q_table_start_ptr=Q_table_buffer+j*Q_table_stride_z;                      
                        uint8_t *WT_buffer_start_ptr=WT_buffer_buffer+j*WT_buffer_stride_z;
                        int16_t *Q_table_ptr = reinterpret_cast<int16_t *>(Q_table_start_ptr);
                        uint16_t *WT_buffer_ptr=reinterpret_cast<uint16_t *>(WT_buffer_start_ptr);

                        short value_count=*Q_table_ptr;Q_table_ptr++;

                        while(value_count--)
                        {
                                int16x8_t onesum=vdupq_n_s16(0.f);
                                signed short q=(*Q_table_ptr);Q_table_ptr++;
                                int8_t temp_value=(q&0xff00)>>8;signed short temp_quantity=(q&0x00ff);
                                while(temp_quantity>0)
                                {

                                        unsigned short result=(*WT_buffer_ptr);WT_buffer_ptr++;
                                        
                                        int16x8_t input_value=vdupq_n_s16(0.f);
                                        int8_t *input_ptr = reinterpret_cast<int8_t *>(input_group_addr+result);   
                                        /*weights_ptr=reinterpret_cast<int16_t *>(temp_input_ptr+input_ptr_offset);*/
                                        input_value=vmovl_s8(vld1_s8(input_ptr));
                                        onesum=vaddq_s16(onesum,input_value);
                                        temp_quantity--;
                                }
                                int16_t real_value=(int16_t)(temp_value);     
                                int16x8_t onevalue = vdupq_n_s16(real_value);
                                acc=vaddq_s16(acc,vmulq_s16(onesum,onevalue));          
                        }
                        if(_convolution_last==0)
                        {
                                int8_t bias_value=0;
                                if(bias_buffer!=nullptr){
                                         int8_t *bias_ptr= reinterpret_cast<int8_t *>(bias_buffer+(int)j);
                                         bias_value=(*bias_ptr);
                                 }
                                int16_t real_bias=(int16_t)bias_value;
                                int16x8_t acc_bias=vdupq_n_s16(real_bias);

                                int16x8_t ope_a=acc;int16x8_t ope_b=acc_bias;

                                int16x4_t ope_a_high=vget_high_s16(ope_a);
                                int16x4_t ope_a_low=vget_low_s16(ope_a);
                                int16x4_t ope_b_high=vget_high_s16(ope_b);
                                int16x4_t ope_b_low=vget_low_s16(ope_b);

                                int32x4_t ope_a_high_long=vmovl_s16(ope_a_high);
                                int32x4_t ope_a_low_long=vmovl_s16(ope_a_low);
                                int32x4_t ope_b_high_long=vmovl_s16(ope_b_high);
                                int32x4_t ope_b_low_long=vmovl_s16(ope_b_low);

                                float32x4_t a_high=vcvtq_f32_s32(ope_a_high_long);
                                float32x4_t a_low=vcvtq_f32_s32(ope_a_low_long);
                                float32x4_t b_high=vcvtq_f32_s32(ope_b_high_long);
                                float32x4_t b_low=vcvtq_f32_s32(ope_b_low_long);


                                a_high=vmulq_f32(a_high,a_mul);
                                a_low=vmulq_f32(a_low,a_mul);
                                b_high=vmulq_f32(b_high,b_mul);
                                b_low=vmulq_f32(b_low,b_mul);

                                float32x4_t res1=vaddq_f32(a_high,b_high);
                                float32x4_t res2=vaddq_f32(a_low,b_low);

                                int32x4_t res1_s32=vcvtq_s32_f32(res1);
                                int32x4_t res2_s32=vcvtq_s32_f32(res2);

                                int16x4_t res1_s16=vmovn_s32(res1_s32);
                                int16x4_t res2_s16=vmovn_s32(res2_s32);

                                int16x8_t res=vcombine_s16(res2_s16,res1_s16);

                                res=vminq_s16(max_value,vmaxq_s16(min_value,res));

                                int8x8_t real_res=vmovn_s16(res);

                                int8_t *out_addr=reinterpret_cast<int8_t*>(_output->buffer()+j*_output->info()->dimension(0)+output_ptr_offset);
                                vst1_s8(out_addr,real_res);

                        }

                        else{
                                if(x_dimension/8==_output->info()->dimension(0)/8)
                                {
                                        int16_t a[8];
                                        vst1q_s16(a,acc);
                                        float temp_a=0,temp_b=0;
                                        int8_t bias_value=0;
                                        if(bias_buffer!=nullptr){
                                                int8_t *bias_ptr= reinterpret_cast<int8_t *>(bias_buffer+(int)j);
                                                bias_value=(*bias_ptr);
                                        }
                                        temp_b=(float)bias_value;

                                        if(_convolution_last==1){
                                                int8_t *myout_ptr3= reinterpret_cast<int8_t *>(temp_output_ptr+(int)output_ptr_offset+j*_output->info()->strides_in_bytes().y());
                                                temp_a=(float)(a[0]);
                                                *myout_ptr3=judge_overflow(temp_a*_a_mul_value+temp_b*_b_mul_value);
                                        }
                                        else if(_convolution_last==2){
                                                int8_t *myout_ptr3= reinterpret_cast<int8_t *>(temp_output_ptr+(int)output_ptr_offset+j*_output->info()->strides_in_bytes().y());
                                                temp_a=(float)(a[0]);
                                                *myout_ptr3=judge_overflow(temp_a*_a_mul_value+temp_b*_b_mul_value);
                                                temp_a=(float)(a[1]);
                                                int8_t *myout_ptr31= reinterpret_cast<int8_t *>(temp_output_ptr+(int)output_ptr_offset+1+j*_output->info()->strides_in_bytes().y());
                                                 *myout_ptr31=judge_overflow(temp_a*_a_mul_value+temp_b*_b_mul_value);
                                        }
                                        else if(_convolution_last==3){
                                                int8_t *myout_ptr3= reinterpret_cast<int8_t *>(temp_output_ptr+(int)output_ptr_offset+j*_output->info()->strides_in_bytes().y());
                                                temp_a=(float)(a[0]);
                                               *myout_ptr3=judge_overflow(temp_a*_a_mul_value+temp_b*_b_mul_value);
                                                temp_a=(float)(a[1]);
                                                int8_t *myout_ptr31= reinterpret_cast<int8_t *>(temp_output_ptr+(int)output_ptr_offset+1+j*_output->info()->strides_in_bytes().y());
                                                *myout_ptr31=judge_overflow(temp_a*_a_mul_value+temp_b*_b_mul_value);
                                                temp_a=(float)(a[2]);
                                                int8_t *myout_ptr32= reinterpret_cast<int8_t *>(temp_output_ptr+(int)output_ptr_offset+2+j*_output->info()->strides_in_bytes().y());
                                                *myout_ptr32=judge_overflow(temp_a*_a_mul_value+temp_b*_b_mul_value);
                                        }
                                           else if(_convolution_last==4){
                                                int8_t *myout_ptr3= reinterpret_cast<int8_t *>(temp_output_ptr+(int)output_ptr_offset+j*_output->info()->strides_in_bytes().y());
                                                temp_a=(float)(a[0]);
                                               *myout_ptr3=judge_overflow(temp_a*_a_mul_value+temp_b*_b_mul_value);
                                                temp_a=(float)(a[1]);
                                                int8_t *myout_ptr31= reinterpret_cast<int8_t *>(temp_output_ptr+(int)output_ptr_offset+1+j*_output->info()->strides_in_bytes().y());
                                                *myout_ptr31=judge_overflow(temp_a*_a_mul_value+temp_b*_b_mul_value);
                                                temp_a=(float)(a[2]);
                                                int8_t *myout_ptr32= reinterpret_cast<int8_t *>(temp_output_ptr+(int)output_ptr_offset+2+j*_output->info()->strides_in_bytes().y());
                                                *myout_ptr32=judge_overflow(temp_a*_a_mul_value+temp_b*_b_mul_value);
                                                temp_a=(float)(a[3]);
                                                int8_t *myout_ptr33= reinterpret_cast<int8_t *>(temp_output_ptr+(int)output_ptr_offset+3+j*_output->info()->strides_in_bytes().y());
                                                *myout_ptr33=judge_overflow(temp_a*_a_mul_value+temp_b*_b_mul_value);
                                        }
                                           else if(_convolution_last==5){
                                                int8_t *myout_ptr3= reinterpret_cast<int8_t *>(temp_output_ptr+(int)output_ptr_offset+j*_output->info()->strides_in_bytes().y());
                                                temp_a=(float)(a[0]);
                                                *myout_ptr3=judge_overflow(temp_a*_a_mul_value+temp_b*_b_mul_value);
                                                temp_a=(float)(a[1]);
                                                int8_t *myout_ptr31= reinterpret_cast<int8_t *>(temp_output_ptr+(int)output_ptr_offset+1+j*_output->info()->strides_in_bytes().y());
                                                *myout_ptr31=judge_overflow(temp_a*_a_mul_value+temp_b*_b_mul_value);
                                                temp_a=(float)(a[2]);
                                                int8_t *myout_ptr32= reinterpret_cast<int8_t *>(temp_output_ptr+(int)output_ptr_offset+2+j*_output->info()->strides_in_bytes().y());
                                                *myout_ptr32=judge_overflow(temp_a*_a_mul_value+temp_b*_b_mul_value);
                                                temp_a=(float)(a[3]);
                                                int8_t *myout_ptr33= reinterpret_cast<int8_t *>(temp_output_ptr+(int)output_ptr_offset+3+j*_output->info()->strides_in_bytes().y());
                                                *myout_ptr33=judge_overflow(temp_a*_a_mul_value+temp_b*_b_mul_value);
                                                temp_a=(float)(a[4]);
                                                int8_t *myout_ptr34= reinterpret_cast<int8_t *>(temp_output_ptr+(int)output_ptr_offset+4+j*_output->info()->strides_in_bytes().y());
                                                *myout_ptr34=judge_overflow(temp_a*_a_mul_value+temp_b*_b_mul_value);
                                        }
                                           else if(_convolution_last==6){
                                                int8_t *myout_ptr3= reinterpret_cast<int8_t *>(temp_output_ptr+(int)output_ptr_offset+j*_output->info()->strides_in_bytes().y());
                                                temp_a=(float)(a[0]);
                                               *myout_ptr3=judge_overflow(temp_a*_a_mul_value+temp_b*_b_mul_value);
                                                temp_a=(float)(a[1]);
                                                int8_t *myout_ptr31= reinterpret_cast<int8_t *>(temp_output_ptr+(int)output_ptr_offset+1+j*_output->info()->strides_in_bytes().y());
                                                *myout_ptr31=judge_overflow(temp_a*_a_mul_value+temp_b*_b_mul_value);
                                                temp_a=(float)(a[2]);
                                                int8_t *myout_ptr32= reinterpret_cast<int8_t *>(temp_output_ptr+(int)output_ptr_offset+2+j*_output->info()->strides_in_bytes().y());
                                                *myout_ptr32=judge_overflow(temp_a*_a_mul_value+temp_b*_b_mul_value);
                                                temp_a=(float)(a[3]);
                                                int8_t *myout_ptr33= reinterpret_cast<int8_t *>(temp_output_ptr+(int)output_ptr_offset+3+j*_output->info()->strides_in_bytes().y());
                                                *myout_ptr33=judge_overflow(temp_a*_a_mul_value+temp_b*_b_mul_value);
                                                temp_a=(float)(a[4]);
                                                int8_t *myout_ptr34= reinterpret_cast<int8_t *>(temp_output_ptr+(int)output_ptr_offset+4+j*_output->info()->strides_in_bytes().y());
                                                *myout_ptr34=judge_overflow(temp_a*_a_mul_value+temp_b*_b_mul_value);
                                                temp_a=(float)(a[5]);
                                                int8_t *myout_ptr35= reinterpret_cast<int8_t *>(temp_output_ptr+(int)output_ptr_offset+5+j*_output->info()->strides_in_bytes().y());
                                                *myout_ptr35=judge_overflow(temp_a*_a_mul_value+temp_b*_b_mul_value);

                                        }
                                           else if(_convolution_last==7){
                                                int8_t *myout_ptr3= reinterpret_cast<int8_t *>(temp_output_ptr+(int)output_ptr_offset+j*_output->info()->strides_in_bytes().y());
                                                temp_a=(float)(a[0]);
                                               *myout_ptr3=judge_overflow(temp_a*_a_mul_value+temp_b*_b_mul_value);
                                                temp_a=(float)(a[1]);
                                                int8_t *myout_ptr31= reinterpret_cast<int8_t *>(temp_output_ptr+(int)output_ptr_offset+1+j*_output->info()->strides_in_bytes().y());
                                                *myout_ptr31=judge_overflow(temp_a*_a_mul_value+temp_b*_b_mul_value);
                                                temp_a=(float)(a[2]);
                                                int8_t *myout_ptr32= reinterpret_cast<int8_t *>(temp_output_ptr+(int)output_ptr_offset+2+j*_output->info()->strides_in_bytes().y());
                                                *myout_ptr32=judge_overflow(temp_a*_a_mul_value+temp_b*_b_mul_value);
                                                temp_a=(float)(a[3]);
                                                int8_t *myout_ptr33= reinterpret_cast<int8_t *>(temp_output_ptr+(int)output_ptr_offset+3+j*_output->info()->strides_in_bytes().y());
                                                *myout_ptr33=judge_overflow(temp_a*_a_mul_value+temp_b*_b_mul_value);
                                                temp_a=(float)(a[4]);
                                                int8_t *myout_ptr34= reinterpret_cast<int8_t *>(temp_output_ptr+(int)output_ptr_offset+4+j*_output->info()->strides_in_bytes().y());
                                                *myout_ptr34=judge_overflow(temp_a*_a_mul_value+temp_b*_b_mul_value);
                                                temp_a=(float)(a[5]);
                                                int8_t *myout_ptr35= reinterpret_cast<int8_t *>(temp_output_ptr+(int)output_ptr_offset+5+j*_output->info()->strides_in_bytes().y());
                                                *myout_ptr35=judge_overflow(temp_a*_a_mul_value+temp_b*_b_mul_value);
                                                temp_a=(float)(a[6]);
                                                int8_t *myout_ptr36= reinterpret_cast<int8_t *>(temp_output_ptr+(int)output_ptr_offset+6+j*_output->info()->strides_in_bytes().y());
                                                *myout_ptr36=judge_overflow(temp_a*_a_mul_value+temp_b*_b_mul_value);
                                        }
                                }

                                else{
                                int8_t bias_value=0;
                                  if(bias_buffer!=nullptr){
                                         int8_t *bias_ptr= reinterpret_cast<int8_t *>(bias_buffer+(int)j);
                                         bias_value=(*bias_ptr);
                                  }
                                int16_t real_bias=(int16_t)bias_value;
                                int16x8_t acc_bias=vdupq_n_s16(real_bias);

                                int16x8_t ope_a=acc;int16x8_t ope_b=acc_bias;

                                int16x4_t ope_a_high=vget_high_s16(ope_a);
                                int16x4_t ope_a_low=vget_low_s16(ope_a);
                                int16x4_t ope_b_high=vget_high_s16(ope_b);
                                int16x4_t ope_b_low=vget_low_s16(ope_b);

                                int32x4_t ope_a_high_long=vmovl_s16(ope_a_high);
                                int32x4_t ope_a_low_long=vmovl_s16(ope_a_low);
                                int32x4_t ope_b_high_long=vmovl_s16(ope_b_high);
                                int32x4_t ope_b_low_long=vmovl_s16(ope_b_low);

                                float32x4_t a_high=vcvtq_f32_s32(ope_a_high_long);
                                float32x4_t a_low=vcvtq_f32_s32(ope_a_low_long);
                                float32x4_t b_high=vcvtq_f32_s32(ope_b_high_long);
                                float32x4_t b_low=vcvtq_f32_s32(ope_b_low_long);


                                a_high=vmulq_f32(a_high,a_mul);
                                a_low=vmulq_f32(a_low,a_mul);
                                b_high=vmulq_f32(b_high,b_mul);
                                b_low=vmulq_f32(b_low,b_mul);

                                float32x4_t res1=vaddq_f32(a_high,b_high);
                                float32x4_t res2=vaddq_f32(a_low,b_low);


                                int32x4_t res1_s32=vcvtq_s32_f32(res1);
                                int32x4_t res2_s32=vcvtq_s32_f32(res2);

                                int16x4_t res1_s16=vmovn_s32(res1_s32);
                                int16x4_t res2_s16=vmovn_s32(res2_s32);

                                int16x8_t res=vcombine_s16(res2_s16,res1_s16);

                                res=vminq_s16(max_value,vmaxq_s16(min_value,res));

                                int8x8_t real_res=vmovn_s16(res);

                                int8_t *out_addr=reinterpret_cast<int8_t*>(_output->buffer()+j*_output->info()->dimension(0)+output_ptr_offset);
                                vst1_s8(out_addr,real_res);
                                }
                        }
                  }
        },out);
}

