#include "arm_compute/runtime/NEON/functions/NEABMConvolutionLayer.h"











#include "arm_compute/core/Size2D.h"



#include "arm_compute/core/Utils.h"



#include "arm_compute/core/Validate.h"



#include "arm_compute/runtime/NEON/NEScheduler.h"



#include "utils/ImageLoader.h"



#include "utils/Utils.h"







#include <set>



#include <tuple>



#include <chrono>







using namespace arm_compute;







NEABMConvolutionLayer::NEABMConvolutionLayer(const std::shared_ptr<IMemoryManager> &memory_manager )



    :_memory_group(memory_manager),_im2col_kernel_s8(),_interleave_kernel(),_matrix_multiply(memory_manager),_col2im_kernel(),



     input_im2col(),input_interleave(),multiply_output(),_data_layout(DataLayout::NCHW),_skip_im2col(false),_skip_col2im(false),



     _is_quantized(false), _is_prepared(false), _im2col_time(0), _interleave_time(0), _matrix_multiply_time(0), _mmlast_time(0), _col2im_time(0)



{    



}







void NEABMConvolutionLayer::configure_mm(const ITensor *input,const ITensor *Q_table,const ITensor *WT_buffer, const ITensor *biases, ITensor *output, const ActivationLayerInfo &act_info, unsigned int precision[],int gemm_3d_depth,unsigned int num_groups)



{



    /*ARM_COMPUTE_ERROR_ON_NULLPTR(input, weights);



    ARM_COMPUTE_ERROR_THROW_ON(validate_mm(input->info(), weights->info(), biases == nullptr ? nullptr : biases->info(), output == nullptr ? nullptr : output->info(),



                                           act_info, gemm_3d_depth, _skip_im2col));



    Create GEMMInfo structure*/



    const GEMMInfo &gemm_info = GEMMInfo(false, false, true /* Reshape weights only for the first run */,



                                         gemm_3d_depth, _skip_im2col /* Reinterpret the input as 3D if im2col is skipped */,



                                         false, GEMMLowpOutputStageInfo(), false, false, act_info);



   



    const std::set<ActivationLayerInfo::ActivationFunction> supported_acts = { ActivationLayerInfo::ActivationFunction::RELU,



                                                                               ActivationLayerInfo::ActivationFunction::BOUNDED_RELU,



                                                                               ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU



                                                                             };



    _matrix_multiply.configure(input, Q_table,WT_buffer, biases, output,precision, 1.0f, 0.0f, gemm_info,num_groups);



}







void NEABMConvolutionLayer::configure(const ITensor *input, const ITensor *Q_table, const ITensor *WT_buffer, const ITensor *biases, ITensor *output, const PadStrideInfo &conv_info, const WeightsInfo &weights_info,unsigned int precision[], 



                                       unsigned int num_groups, const Size2D &dilation, const ActivationLayerInfo &act_info)



{



    ARM_COMPUTE_ERROR_ON_NULLPTR(input, Q_table,WT_buffer, output);



    ARM_COMPUTE_ERROR_THROW_ON(NEABMConvolutionLayer::validate(input->info(),







                                                                Q_table->info(),







                                                                WT_buffer->info();







                                                                biases != nullptr ? biases->info() : nullptr,







                                                                output->info(),







                                                                conv_info,







                                                                weights_info,







                                                                dilation,







                                                                act_info,







                                                                num_groups));















    /*const DataType   data_type   = input->info()->data_type();*/



    const DataLayout data_layout = input->info()->data_layout();



    const int        idx_width   = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);



    const int        idx_height  = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);



    /*const int        idx_channel=get_data_layout_dimension_index(data_layout,DataLayoutDimension::CHANNEL);*/



    /*const int        idx_kernels = get_data_layout_dimension_index(data_layout, DataLayoutDimension::BATCHES);*/







    const unsigned int kernel_width  =weights_info.kernel_size().first;



    const unsigned int kernel_height =weights_info.kernel_size().second;



    unsigned int mat_weights_cols =weights_info.num_kernels();



    _is_prepared      = weights_info.retain_internal_weights();



    /*_is_quantized     = is_data_type_quantized_asymmetric(input->info()->data_type());*/







    _data_layout      = data_layout;











    unsigned int conv_w = 0;



    unsigned int conv_h = 0;



    std::tie(conv_w, conv_h) = scaled_dimensions(input->info()->dimension(idx_width),



                                                 input->info()->dimension(idx_height),



                                                 kernel_width,



                                                 kernel_height,



                                                 conv_info,



                                                 dilation);











    unsigned int stride_x = 0;



    unsigned int stride_y = 0;



    std::tie(stride_x, stride_y) = conv_info.stride();











  /*



    unsigned int Q=Q_WT[0];unsigned int WT=Q_WT[1]; 



    _save_filename=save_file_name;







    TensorShape Q_table_shape=input->info()->tensor_shape();







    Q_table_shape.set(0,1);Q_table_shape.set(1,Q);Q_table_shape.set(2,mat_weights_cols);







    TensorInfo Q_table_info(Q_table_shape,1,DataType::S16);







    Q_table.allocator()->init(Q_table_info);







    Q_table.allocator()->allocate();











    TensorShape WT_buffer_shape=input->info()->tensor_shape();







    WT_buffer_shape.set(0,1);WT_buffer_shape.set(1,WT);WT_buffer_shape.set(2,mat_weights_cols);//暂定为100







    TensorInfo WT_buffer_info(WT_buffer_shape,1,DataType::U16);







    WT_buffer.allocator()->init(WT_buffer_info);







    WT_buffer.allocator()->allocate();







    ITensor *Q_table_use=&Q_table;







    ITensor *WT_buffer_use=&WT_buffer;











    _reshape_weights.configure(weights,Q_table_use,WT_buffer_use,index,Q,WT);



*/



   







    /*Create tensor to store im2col reshaped inputs*/







    if(!_skip_im2col)







    {



        _memory_group.manage(&input_im2col);



        _im2col_kernel_s8.configure(input, &input_im2col, Size2D(kernel_width, kernel_height), conv_info, false, dilation);



        input_im2col.allocator()->allocate();







        _memory_group.manage(&input_interleave);



        _interleave_kernel.configure(&input_im2col,&input_interleave);



        input_interleave.allocator()->allocate();



        



    }











    if(!_skip_col2im)







    {







        TensorShape multiply_output_shape;



        multiply_output_shape = input_im2col.info()->tensor_shape();



        multiply_output_shape.set(0, conv_w * conv_h);           



        multiply_output_shape.set(1, mat_weights_cols);          



        TensorInfo multiply_output_info(multiply_output_shape, 1, DataType::S8);



        multiply_output.allocator()->init(multiply_output_info);



        multiply_output.allocator()->allocate();



        _memory_group.manage(&multiply_output);







    }







    const unsigned int gemm_3d_depth = _skip_col2im ? conv_h : 0;



    const ITensor *input_interleave_to_use=&input_interleave;



    ITensor       *multiply_output_to_use = output;



    multiply_output_to_use = &multiply_output;



    configure_mm(input_interleave_to_use, Q_table,WT_buffer, biases, multiply_output_to_use, act_info, precision,gemm_3d_depth,num_groups);











    if(!_skip_col2im)







    {



        if(_data_layout == DataLayout::NCHW)



        {



            _col2im_kernel.configure(multiply_output_to_use, output, Size2D(conv_w, conv_h));     



        }







    }



}







Status NEABMConvolutionLayer::validate(const ITensorInfo *input, const ITensorInfo *Q_table, const ITensorInfo *WT_buffer, const ITensorInfo *biases, const ITensorInfo *output, const PadStrideInfo &conv_info,



                                        const WeightsInfo &weights_info, unsigned int num_groups, const Size2D &dilation, const ActivationLayerInfo &act_info)







{



    return Status{};



}







void NEABMConvolutionLayer::run()



{



 /*   prepare();           */







    if(!_skip_im2col)



    {



        auto begin1=std::chrono::high_resolution_clock::now();



        NEScheduler::get().schedule(&_im2col_kernel_s8, Window::DimY);



        auto end1=std::chrono::high_resolution_clock::now();







        auto begin2=std::chrono::high_resolution_clock::now();



        NEScheduler::get().schedule(&_interleave_kernel,Window::DimY);



        auto end2=std::chrono::high_resolution_clock::now();







        _im2col_time = std::chrono::duration_cast<std::chrono::duration<double>>(end1 - begin1).count();



        _interleave_time = std::chrono::duration_cast<std::chrono::duration<double>>(end2 - begin2).count();





    }                     

    /*



    arm_compute::utils::NPYLoader save;



    save.save_to_npy2(input_interleave, "/media/sdcard/ComputeLibrary/data/neon_abm/interleave.npy",false);



    */



    _matrix_multiply.run();



    // _matrix_multiply_time=_matrix_multiply.print_time().first;

    

    // _mmlast_time=_matrix_multiply.print_time().second;







    if(!_skip_col2im)



     {



        if(_data_layout == DataLayout::NCHW)



        {

            auto begin5=std::chrono::high_resolution_clock::now();



            NEScheduler::get().schedule(&_col2im_kernel, Window::DimY);



            auto end5=std::chrono::high_resolution_clock::now();



            _col2im_time=std::chrono::duration_cast<std::chrono::duration<double>>(end5 - begin5).count();

        }



    }



}



std::tuple<double, double, double, double, double> NEABMConvolutionLayer::print_kernel_time()

{

    std::tuple<double, double, double, double, double> kernel_tuple(_im2col_time, _interleave_time, _matrix_multiply_time, _mmlast_time, _col2im_time);

    return kernel_tuple;

}



/*



void NEABMConvolutionLayer::prepare()



{



    if(!_is_prepared)



    {



            _reshape_weights.run();



            _original_weights->mark_as_unused();



            _matrix_multiply.prepare();



             _is_prepared = true;



    }



}



*/










