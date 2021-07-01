#include "arm_compute/core/NEON/kernels/NEABMInterleaveBlockKernel.h"
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

using namespace arm_compute;

namespace
{

inline Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output)

{
        return Status{};
}


inline std::pair<Status,Window> validate_and_configure_window(ITensorInfo *input, unsigned int num_convs)

{
        Window window=calculate_max_window(*input,Steps());
        unsigned int num_blocks=num_convs/32;
        window.set(Window::DimX, Window::Dimension(0, input->dimension(0),8));   
        if(num_blocks==0){
            window.set(Window::DimY, Window::Dimension(0, 0,0));
        }
        else{
            window.set(Window::DimY, Window::Dimension(0, 4*num_blocks,4));
        }
        window.set(Window::DimZ, Window::Dimension(0, input->dimension(2), input->dimension(2)));
        return std::make_pair(Status{}, window);
}

}

NEABMInterleaveBlockKernel::NEABMInterleaveBlockKernel()
        :_input(nullptr),_output(nullptr), _num_convs(0)
{
}
void NEABMInterleaveBlockKernel::configure(const ITensor *input, ITensor *output, unsigned int num_convs)
{
        ARM_COMPUTE_ERROR_ON_NULLPTR(input,output);

        TensorShape shape{input->info()->tensor_shape()};
        shape.set(0,input->info()->dimension(0)*4);
        shape.set(1,num_convs/32);
        output->info()->set_data_type(DataType::S8);
        output->info()->set_num_channels(input->info()->num_channels());
        output->info()->set_tensor_shape(shape);
        output->info()->set_quantization_info(input->info()->quantization_info());
        output->info()->set_data_layout(input->info()->data_layout());

        ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), output->info()));

        _input=input;
        _output=output;
        _num_convs=num_convs;

        auto win_config = validate_and_configure_window(input->info(),num_convs);
        ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
        INEKernel::configure(win_config.second);
}



Status NEABMInterleaveBlockKernel::validate(const ITensorInfo *input,const ITensorInfo *output)

{
        return Status{};
}



void NEABMInterleaveBlockKernel::run(const Window &window, const ThreadInfo &info)
{
        ARM_COMPUTE_UNUSED(info);
        ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
        ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

        Window wout(window);
        wout.set(Window::DimX, Window::Dimension(0, 0, 0));
        wout.set(Window::DimY, Window::Dimension(0, 0, 0));
        wout.set(Window::DimZ, Window::Dimension(0, 0, 0));

        Iterator out(_output,wout);

        execute_window_loop(window, [&](const Coordinates & id){
                
                size_t x_dimension=id[0], y_dimension=id[1]; 

                uint8_t *temp_input_ptr=_input->buffer();      
                uint8_t *temp_output_ptr=_output->buffer();

                int8x8_t a1=vld1_s8(reinterpret_cast<int8_t*>(temp_input_ptr+_input->info()->offset_element_in_bytes(Coordinates(x_dimension, y_dimension, 0))));
                int8x8_t a2=vld1_s8(reinterpret_cast<int8_t*>(temp_input_ptr+_input->info()->offset_element_in_bytes(Coordinates(x_dimension, y_dimension+1, 0))));
                int8x8_t a3=vld1_s8(reinterpret_cast<int8_t*>(temp_input_ptr+_input->info()->offset_element_in_bytes(Coordinates(x_dimension, y_dimension+2, 0))));
                int8x8_t a4=vld1_s8(reinterpret_cast<int8_t*>(temp_input_ptr+_input->info()->offset_element_in_bytes(Coordinates(x_dimension, y_dimension+3, 0))));
                
                vst1_s8(reinterpret_cast<int8_t*>(temp_output_ptr+_output->info()->offset_element_in_bytes(Coordinates(x_dimension*4, y_dimension/4, 0))), a1);
                vst1_s8(reinterpret_cast<int8_t*>(temp_output_ptr+_output->info()->offset_element_in_bytes(Coordinates(x_dimension*4+8, y_dimension/4, 0))), a2);
                vst1_s8(reinterpret_cast<int8_t*>(temp_output_ptr+_output->info()->offset_element_in_bytes(Coordinates(x_dimension*4+16, y_dimension/4, 0))), a3);
                vst1_s8(reinterpret_cast<int8_t*>(temp_output_ptr+_output->info()->offset_element_in_bytes(Coordinates(x_dimension*4+24, y_dimension/4, 0))), a4);
               
                
        },out);

        /*
        std::cout<<"thread-id"<<std::endl;
        std::cout<<info.thread_id<<std::endl;
        std::cout<<"xyz"<<std::endl;
        std::cout<<window.x().start()<<"  "<<window.x().end()<<"  "<<window.x().step()<<std::endl;
        std::cout<<window.y().start()<<"  "<<window.y().end()<<"  "<<window.y().step()<<std::endl;
        std::cout<<window.z().start()<<"  "<<window.z().end()<<"  "<<window.z().step()<<std::endl;
        */
}