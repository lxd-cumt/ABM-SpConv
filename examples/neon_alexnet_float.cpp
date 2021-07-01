#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/CLFunctions.h"
#include "arm_compute/runtime/NEON/NEFunctions.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "arm_compute/runtime/Allocator.h"
#include "utils/ImageLoader.h"
#include "utils/Utils.h"
#include <ctime>
#include <cstdlib>
#include <sched.h>
#include <unistd.h>
#include <sys/times.h>

using namespace arm_compute;
using namespace utils;
using namespace std;


class NEONALEXExample : public Example
{
public:

    bool do_setup(int argc, char **argv) override
    {
        // Layer conv 0
        constexpr unsigned int width_src  = 227;
        constexpr unsigned int height_src = 227;
        constexpr unsigned int ifm_src     = 3;
        const TensorShape src_shape(width_src, height_src, ifm_src);
        src.allocator()->init(TensorInfo(src_shape, 1, DataType::F32));

        constexpr unsigned int kernel_x_conv0 = 11;
        constexpr unsigned int kernel_y_conv0 = 11;
        constexpr unsigned int ofm_conv0      = 96;
        const TensorShape weights_conv0_shape(kernel_x_conv0, kernel_y_conv0, ifm_src, ofm_conv0);
        const TensorShape bias_conv0_shape(ofm_conv0);
        weights_conv0.allocator()->init(TensorInfo(weights_conv0_shape, 1, DataType::F32));
        bias_conv0.allocator()->init(TensorInfo(bias_conv0_shape, 1, DataType::F32));


        //Layer conv 1
        constexpr unsigned int width_input_conv1_l  = 27;
        constexpr unsigned int height_input_conv1_l = 27;
        constexpr unsigned int ifm_input_conv1_l      = 48;
        const TensorShape input_conv1_l_shape(width_input_conv1_l, height_input_conv1_l, ifm_input_conv1_l);
        input_conv1_l.allocator()->init(TensorInfo(input_conv1_l_shape, 1, DataType::F32));
        constexpr unsigned int width_input_conv1_r  = 27;
        constexpr unsigned int height_input_conv1_r = 27;
        constexpr unsigned int ifm_input_conv1_r      = 48;
        const TensorShape input_conv1_r_shape(width_input_conv1_r, height_input_conv1_r, ifm_input_conv1_r);
        input_conv1_r.allocator()->init(TensorInfo(input_conv1_r_shape, 1, DataType::F32));

        constexpr unsigned int kernel_x_conv1 = 5;
        constexpr unsigned int kernel_y_conv1 = 5;
        constexpr unsigned int ofm_conv1      = 128;
        const TensorShape weights_conv1_l_shape(kernel_x_conv1, kernel_y_conv1, ifm_input_conv1_l, ofm_conv1);
        const TensorShape bias_conv1_l_shape(ofm_conv1);
        weights_conv1_l.allocator()->init(TensorInfo(weights_conv1_l_shape, 1, DataType::F32));
        bias_conv1_l.allocator()->init(TensorInfo(bias_conv1_l_shape, 1, DataType::F32));
        const TensorShape weights_conv1_r_shape(kernel_x_conv1, kernel_y_conv1, ifm_input_conv1_r, ofm_conv1);
        const TensorShape bias_conv1_r_shape(ofm_conv1);
        weights_conv1_r.allocator()->init(TensorInfo(weights_conv1_r_shape, 1, DataType::F32));
        bias_conv1_r.allocator()->init(TensorInfo(bias_conv1_r_shape, 1, DataType::F32));

        constexpr unsigned int width_out_conv1  = 27;
        constexpr unsigned int height_out_conv1 = 27;
        constexpr unsigned int ifm_out_conv1     = 256;
        const TensorShape out_conv1_shape(width_out_conv1, height_out_conv1, ifm_out_conv1);
        out_conv1.allocator()->init(TensorInfo(out_conv1_shape, 1, DataType::F32));



        //Layer conv 2
        constexpr unsigned int kernel_x_conv2 = 3;
        constexpr unsigned int kernel_y_conv2 = 3;
        constexpr unsigned int ofm_conv2      = 384;
        const TensorShape weights_conv2_shape(kernel_x_conv2, kernel_y_conv2, ifm_out_conv1, ofm_conv2);
        const TensorShape bias_conv2_shape(ofm_conv2);
        weights_conv2.allocator()->init(TensorInfo(weights_conv2_shape, 1, DataType::F32));
        bias_conv2.allocator()->init(TensorInfo(bias_conv2_shape, 1, DataType::F32));
        

        //Layer conv 3
        constexpr unsigned int width_input_conv3_l  = 13;
        constexpr unsigned int height_input_conv3_l = 13;
        constexpr unsigned int ifm_input_conv3_l      = 192;
        const TensorShape input_conv3_l_shape(width_input_conv3_l, height_input_conv3_l, ifm_input_conv3_l);
        input_conv3_l.allocator()->init(TensorInfo(input_conv3_l_shape, 1, DataType::F32));
        constexpr unsigned int width_input_conv3_r  = 13;
        constexpr unsigned int height_input_conv3_r = 13;
        constexpr unsigned int ifm_input_conv3_r      = 192;
        const TensorShape input_conv3_r_shape(width_input_conv3_r, height_input_conv3_r, ifm_input_conv3_r);
        input_conv3_r.allocator()->init(TensorInfo(input_conv3_r_shape, 1, DataType::F32));

        constexpr unsigned int kernel_x_conv3 = 3;
        constexpr unsigned int kernel_y_conv3 = 3;
        constexpr unsigned int ofm_conv3      = 192;
        const TensorShape weights_conv3_l_shape(kernel_x_conv3, kernel_y_conv3, ifm_input_conv3_l, ofm_conv3);
        const TensorShape bias_conv3_l_shape(ofm_conv3);
        weights_conv3_l.allocator()->init(TensorInfo(weights_conv3_l_shape, 1, DataType::F32));
        bias_conv3_l.allocator()->init(TensorInfo(bias_conv3_l_shape, 1, DataType::F32));
        const TensorShape weights_conv3_r_shape(kernel_x_conv3, kernel_y_conv3, ifm_input_conv3_r, ofm_conv3);
        const TensorShape bias_conv3_r_shape(ofm_conv3);
        weights_conv3_r.allocator()->init(TensorInfo(weights_conv3_r_shape, 1, DataType::F32));
        bias_conv3_r.allocator()->init(TensorInfo(bias_conv3_r_shape, 1, DataType::F32));

        constexpr unsigned int width_out_conv3  = 13;
        constexpr unsigned int height_out_conv3 = 13;
        constexpr unsigned int ifm_out_conv3     = 384;
        const TensorShape out_conv3_shape(width_out_conv3, height_out_conv3, ifm_out_conv3);
        out_conv3.allocator()->init(TensorInfo(out_conv3_shape, 1, DataType::F32));


        //Layer conv 4
        constexpr unsigned int width_input_conv4_l  = 13;
        constexpr unsigned int height_input_conv4_l = 13;
        constexpr unsigned int ifm_input_conv4_l      = 192;
        const TensorShape input_conv4_l_shape(width_input_conv4_l, height_input_conv4_l, ifm_input_conv4_l);
        input_conv4_l.allocator()->init(TensorInfo(input_conv4_l_shape, 1, DataType::F32));
        constexpr unsigned int width_input_conv4_r  = 13;
        constexpr unsigned int height_input_conv4_r = 13;
        constexpr unsigned int ifm_input_conv4_r      = 192;
        const TensorShape input_conv4_r_shape(width_input_conv4_r, height_input_conv4_r, ifm_input_conv4_r);
        input_conv4_r.allocator()->init(TensorInfo(input_conv4_r_shape, 1, DataType::F32));

        constexpr unsigned int kernel_x_conv4 = 3;
        constexpr unsigned int kernel_y_conv4 = 3;
        constexpr unsigned int ofm_conv4      = 128;
        const TensorShape weights_conv4_l_shape(kernel_x_conv4, kernel_y_conv4, ifm_input_conv4_l, ofm_conv4);
        const TensorShape bias_conv4_l_shape(ofm_conv4);
        weights_conv4_l.allocator()->init(TensorInfo(weights_conv4_l_shape, 1, DataType::F32));
        bias_conv4_l.allocator()->init(TensorInfo(bias_conv4_l_shape, 1, DataType::F32));
        const TensorShape weights_conv4_r_shape(kernel_x_conv4, kernel_y_conv4, ifm_input_conv4_r, ofm_conv4);
        const TensorShape bias_conv4_r_shape(ofm_conv4);
        weights_conv4_r.allocator()->init(TensorInfo(weights_conv4_r_shape, 1, DataType::F32));
        bias_conv4_r.allocator()->init(TensorInfo(bias_conv4_r_shape, 1, DataType::F32));

        constexpr unsigned int width_out_conv4  = 13;
        constexpr unsigned int height_out_conv4 = 13;
        constexpr unsigned int ifm_out_conv4     = 256;
        const TensorShape out_conv4_shape(width_out_conv4, height_out_conv4, ifm_out_conv4);
        out_conv4.allocator()->init(TensorInfo(out_conv4_shape, 1, DataType::F32));


        //Layer fc 0
        const TensorShape weights_fc0_shape(6,6,256,4096);
        const TensorShape bias_fc0_shape(4096);
        weights_fc0.allocator()->init(TensorInfo(weights_fc0_shape, 1, DataType::F32));
        bias_fc0.allocator()->init(TensorInfo(bias_fc0_shape, 1, DataType::F32));
       
        //Layer fc 1
        const TensorShape weights_fc1_shape(1,1,4096,4096);
        const TensorShape bias_fc1_shape(4096);
        weights_fc1.allocator()->init(TensorInfo(weights_fc1_shape, 1, DataType::F32));
        bias_fc1.allocator()->init(TensorInfo(bias_fc1_shape, 1, DataType::F32));



        //Layer fc 2
        const TensorShape weights_fc2_shape(1,1,4096,1000);
        const TensorShape bias_fc2_shape(1000);
        weights_fc2.allocator()->init(TensorInfo(weights_fc2_shape, 1, DataType::F32));
        bias_fc2.allocator()->init(TensorInfo(bias_fc2_shape, 1, DataType::F32));

        //std::cout<<"configure start"<<std::endl;
        // configuration //
        conv0.configure(&src, &weights_conv0, &bias_conv0, &out_conv0, PadStrideInfo(4, 4, 0, 0));
        //std::cout<<"configure start"<<std::endl;
        act0.configure(&out_conv0, &out_act0, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        norm0.configure(&out_act0, &out_norm0, NormalizationLayerInfo(NormType::CROSS_MAP, 5, 0.0001f, 0.75f));
        pool0.configure(&out_norm0, &out_pool0, PoolingLayerInfo(PoolingType::MAX, 2, PadStrideInfo(2, 2, 0, 0)));
        //std::cout<<"configure start"<<std::endl;
        conv1_l.configure(&input_conv1_l, &weights_conv1_l, &bias_conv1_l, &out_conv1_l, PadStrideInfo(1, 1, 2, 2));
        conv1_r.configure(&input_conv1_r, &weights_conv1_r, &bias_conv1_r, &out_conv1_r, PadStrideInfo(1, 1, 2, 2));
        act1.configure(&out_conv1, &out_act1, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        norm1.configure(&out_act1, &out_norm1, NormalizationLayerInfo(NormType::CROSS_MAP, 5, 0.0001f, 0.75f));
        pool1.configure(&out_norm1, &out_pool1, PoolingLayerInfo(PoolingType::MAX, 2, PadStrideInfo(2, 2, 0, 0)));
        //std::cout<<"configure start"<<std::endl;
        conv2.configure(&out_pool1, &weights_conv2, &bias_conv2, &out_conv2, PadStrideInfo(1, 1, 1, 1));
        act2.configure(&out_conv2, &out_act2, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        //std::cout<<"configure start"<<std::endl;
        conv3_l.configure(&input_conv3_l, &weights_conv3_l, &bias_conv3_l, &out_conv3_l, PadStrideInfo(1, 1, 1, 1));
        conv3_r.configure(&input_conv3_r, &weights_conv3_r, &bias_conv3_r, &out_conv3_r, PadStrideInfo(1, 1, 1, 1));
        act3.configure(&out_conv3, &out_act3, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        //std::cout<<"configure start"<<std::endl;
        conv4_l.configure(&input_conv4_l, &weights_conv4_l, &bias_conv4_l, &out_conv4_l, PadStrideInfo(1, 1, 1, 1));
        conv4_r.configure(&input_conv4_r, &weights_conv4_r, &bias_conv4_r, &out_conv4_r, PadStrideInfo(1, 1, 1, 1));
        act4.configure(&out_conv4, &out_act4, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        pool4.configure(&out_act4, &out_pool4, PoolingLayerInfo(PoolingType::MAX, 2, PadStrideInfo(2, 2, 0, 0)));
        //std::cout<<"configure start"<<std::endl;
        fc0.configure(&out_pool4, &weights_fc0, &bias_fc0, &out_fc0,PadStrideInfo(1, 1, 0, 0));
        act5.configure(&out_fc0, &out_act5, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        //std::cout<<"configure start"<<std::endl;
        fc1.configure(&out_act5, &weights_fc1, &bias_fc1, &out_fc1,PadStrideInfo(1, 1, 0, 0));
        act6.configure(&out_fc1, &out_act6, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        //std::cout<<"configure start"<<std::endl;
        fc2.configure(&out_act6, &weights_fc2, &bias_fc2, &out_fc2,PadStrideInfo(1, 1, 0, 0));
        softmax.configure(&out_fc2, &out_softmax);

        
        
        //std::cout<<"allocate start"<<std::endl;
        //  allocate  //
        src.allocator()->allocate(); 
        weights_conv0.allocator()->allocate(); bias_conv0.allocator()->allocate();
        out_conv0.allocator()->allocate();
        out_act0.allocator()->allocate(); out_norm0.allocator()->allocate(); out_pool0.allocator()->allocate();
        
        input_conv1_l.allocator()->allocate(); input_conv1_r.allocator()->allocate();
        weights_conv1_l.allocator()->allocate(); weights_conv1_r.allocator()->allocate(); bias_conv1_l.allocator()->allocate(); bias_conv1_r.allocator()->allocate();
        out_conv1_l.allocator()->allocate(); out_conv1_r.allocator()->allocate(); out_conv1.allocator()->allocate();
        out_act1.allocator()->allocate(); out_norm1.allocator()->allocate(); out_pool1.allocator()->allocate();

        weights_conv2.allocator()->allocate(); bias_conv2.allocator()->allocate();
        out_conv2.allocator()->allocate();
        out_act2.allocator()->allocate();

        
        input_conv3_l.allocator()->allocate(); input_conv3_r.allocator()->allocate();
        weights_conv3_l.allocator()->allocate(); weights_conv3_r.allocator()->allocate(); bias_conv3_l.allocator()->allocate(); bias_conv3_r.allocator()->allocate();
        out_conv3_l.allocator()->allocate(); out_conv3_r.allocator()->allocate(); out_conv3.allocator()->allocate();
        out_act3.allocator()->allocate();

        input_conv4_l.allocator()->allocate(); input_conv4_r.allocator()->allocate();
        weights_conv4_l.allocator()->allocate(); weights_conv4_r.allocator()->allocate(); bias_conv4_l.allocator()->allocate(); bias_conv4_r.allocator()->allocate();
        out_conv4_l.allocator()->allocate(); out_conv4_r.allocator()->allocate(); out_conv4.allocator()->allocate();
        out_act4.allocator()->allocate(); out_pool4.allocator()->allocate();

        weights_fc0.allocator()->allocate(); bias_fc0.allocator()->allocate();
        out_fc0.allocator()->allocate();
        out_act5.allocator()->allocate();

        weights_fc1.allocator()->allocate(); bias_fc1.allocator()->allocate();
        out_fc1.allocator()->allocate();
        out_act6.allocator()->allocate();

        weights_fc2.allocator()->allocate(); bias_fc2.allocator()->allocate();
        out_fc2.allocator()->allocate();
        out_softmax.allocator()->allocate();

        // std::cout<<out_pool4.allocator()->info().tensor_shape().x()<<" "<<out_pool4.allocator()->info().tensor_shape().y()<<" "<<out_pool4.allocator()->info().tensor_shape().z()<<std::endl;
        // std::cout<<weights_fc0.allocator()->info().tensor_shape().x()<<" "<<weights_fc0.allocator()->info().tensor_shape().y()<<" "<<weights_fc0.allocator()->info().tensor_shape().z()<<" "<<weights_fc0.allocator()->info().tensor_shape()[3]<<std::endl;
        // std::cout<<bias_fc0.allocator()->info().tensor_shape().x()<<" "<<bias_fc0.allocator()->info().tensor_shape().y()<<" "<<bias_fc0.allocator()->info().tensor_shape().z()<<std::endl;
        // std::cout<<out_fc0.allocator()->info().tensor_shape().x()<<" "<<out_fc0.allocator()->info().tensor_shape().y()<<" "<<out_fc0.allocator()->info().tensor_shape().z()<<std::endl;
        
        
        return true;

    }

    void do_run() override
    {
        double conv_layer=0, act_layer=0, norm_layer=0, pool_layer=0, fc_layer=0, other_layer=0;
        double t_conv0 = 0, t_act0 = 0, t_norm0 = 0, t_pool0 = 0;
        double t_conv1 = 0, t_act1 = 0, t_norm1 = 0, t_pool1 = 0;
        double t_conv2 = 0, t_act2 = 0;
        double t_conv3 = 0, t_act3 = 0;
        double t_conv4 = 0, t_act4 = 0, t_pool4 = 0;
        double t_fc0   = 0, t_act5 = 0;
        double t_fc1   = 0, t_act6 = 0;
        double t_fc2   = 0, t_softmax = 0;
        // double lend01=0, lend02=0, lend03=0, lend04=0;
        // double lend11=0, lend12=0, lend13=0, lend14=0;
        // double lend21=0, lend22=0;
        // double lend31=0, lend32=0;
        // double lend41=0, lend42=0, lend43=0;
        // double lend51=0, lend52=0;
        // double lend61=0, lend62=0;
        // double lend71=0, lend72=0;
        double time=0;
        double end_time=0;
        unsigned int cycles=101;

        std::string base_path = "/media/sdcard/ComputeLibrary";
        std::string output_file_path = "/model.csv";
        ofstream out(base_path+output_file_path, ios::out | ios::app);
        out<<"AlexNet GEMM"<<std::endl;
        for(unsigned int i=0; i<cycles; i++)
        {                 
            auto tbegin = std::chrono::high_resolution_clock::now();

            //std::cout<<"1"<<std::endl;
            conv0.run();auto end01 = std::chrono::high_resolution_clock::now();
            act0.run();auto end02 = std::chrono::high_resolution_clock::now();
            norm0.run();auto end03 = std::chrono::high_resolution_clock::now();
            pool0.run();auto end04 = std::chrono::high_resolution_clock::now();
            //std::cout<<"1"<<std::endl;
            conv1_l.run();conv1_r.run(); auto end11 = std::chrono::high_resolution_clock::now();
            act1.run();auto end12 = std::chrono::high_resolution_clock::now();
            norm1.run();auto end13 = std::chrono::high_resolution_clock::now();
            pool1.run();auto end14 = std::chrono::high_resolution_clock::now();
            //std::cout<<"1"<<std::endl;
            conv2.run();auto end21 = std::chrono::high_resolution_clock::now();
            act2.run();auto end22 = std::chrono::high_resolution_clock::now();
            //std::cout<<"1"<<std::endl;
            conv3_l.run(); conv3_r.run(); auto end31 = std::chrono::high_resolution_clock::now();
            act3.run();auto end32 = std::chrono::high_resolution_clock::now();
            //std::cout<<"1"<<std::endl;
            conv4_l.run(); conv4_r.run(); auto end41 = std::chrono::high_resolution_clock::now();
            act4.run();auto end42 = std::chrono::high_resolution_clock::now();
            pool4.run();auto end43 = std::chrono::high_resolution_clock::now();
            //std::cout<<"1"<<std::endl;
            fc0.run();auto end51 = std::chrono::high_resolution_clock::now();
            act5.run();auto end52 = std::chrono::high_resolution_clock::now();
            //std::cout<<"1"<<std::endl;
            fc1.run();auto end61 = std::chrono::high_resolution_clock::now();
            act6.run();auto end62 = std::chrono::high_resolution_clock::now();
            //std::cout<<"1"<<std::endl;
            fc2.run();auto end71 = std::chrono::high_resolution_clock::now();
            softmax.run();auto end72 = std::chrono::high_resolution_clock::now();     

            auto tend = std::chrono::high_resolution_clock::now();
            
            if(i>0)
            {
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end01 - tbegin).count(); conv_layer+=time; t_conv0 += time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end02 - end01 ).count(); act_layer +=time; t_act0  += time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end03 - end02 ).count(); norm_layer+=time; t_norm0 += time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end04 - end03 ).count(); pool_layer+=time; t_pool0 += time;

                time = std::chrono::duration_cast<std::chrono::duration<double>>(end11 - end04 ).count(); conv_layer+=time; t_conv1 += time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end12 - end11 ).count(); act_layer +=time; t_act1  += time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end13 - end12 ).count(); norm_layer+=time; t_norm1 += time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end14 - end13 ).count(); pool_layer+=time; t_pool1 += time;

                time = std::chrono::duration_cast<std::chrono::duration<double>>(end21 - end14 ).count(); conv_layer+=time; t_conv2 += time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end22 - end21 ).count(); act_layer +=time; t_act2  += time;

                time = std::chrono::duration_cast<std::chrono::duration<double>>(end31 - end22 ).count(); conv_layer+=time; t_conv3 += time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end32 - end31 ).count(); act_layer +=time; t_act3  += time;

                time = std::chrono::duration_cast<std::chrono::duration<double>>(end41 - end32 ).count(); conv_layer+=time; t_conv4 += time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end42 - end41 ).count(); act_layer +=time; t_act4  += time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end43 - end42 ).count(); pool_layer+=time; t_pool4 += time;

                time = std::chrono::duration_cast<std::chrono::duration<double>>(end51 - end43 ).count(); fc_layer  +=time; t_fc0   += time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end52 - end51 ).count(); act_layer +=time; t_act5  += time;

                time = std::chrono::duration_cast<std::chrono::duration<double>>(end61 - end52 ).count(); fc_layer  +=time; t_fc1   += time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end62 - end61 ).count(); act_layer +=time; t_act6  += time;

                time = std::chrono::duration_cast<std::chrono::duration<double>>(end71 - end62 ).count(); fc_layer  +=time; t_fc2   += time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end72 - end71 ).count(); other_layer+=time;t_softmax += time;

                time = std::chrono::duration_cast<std::chrono::duration<double>>(tend - tbegin ).count(); end_time+=time;

                if(i>0){
                    std::cout<<i<<"---run:"<<std::endl;
                    std::cout<<"time="<<time*1000<<"ms"<<std::endl;
                    out<<"one run time"<<","<<time*1000<<std::endl;
                    }
                if(i==0){
                    std::cout<<"First run:"<<std::endl;
                    std::cout<<"time="<<time*1000<<"ms"<<std::endl;
                }
            }
        }
        
        out<<"conv0"<<","<<t_conv0*1000/(cycles-1)<<std::endl;
        out<<"act0"<<","<<t_act0*1000/(cycles-1)<<std::endl;
        out<<"norm0"<<","<<t_norm0*1000/(cycles-1)<<std::endl;
        out<<"pool0"<<","<<t_pool0*1000/(cycles-1)<<std::endl;

        out<<"conv1"<<","<<t_conv1*1000/(cycles-1)<<std::endl;
        out<<"act1"<<","<<t_act1*1000/(cycles-1)<<std::endl;
        out<<"norm1"<<","<<t_norm1*1000/(cycles-1)<<std::endl;
        out<<"pool1"<<","<<t_pool1*1000/(cycles-1)<<std::endl;

        out<<"conv2"<<","<<t_conv2*1000/(cycles-1)<<std::endl;
        out<<"act2"<<","<<t_act2*1000/(cycles-1)<<std::endl;

        out<<"conv3"<<","<<t_conv3*1000/(cycles-1)<<std::endl;
        out<<"act3"<<","<<t_act3*1000/(cycles-1)<<std::endl;

        out<<"conv4"<<","<<t_conv4*1000/(cycles-1)<<std::endl;
        out<<"act4"<<","<<t_act4*1000/(cycles-1)<<std::endl;
        out<<"pool4"<<","<<t_pool4*1000/(cycles-1)<<std::endl;

        out<<"fc0"<<","<<t_fc0*1000/(cycles-1)<<std::endl;
        out<<"act5"<<","<<t_act5*1000/(cycles-1)<<std::endl;

        out<<"fc1"<<","<<t_fc1*1000/(cycles-1)<<std::endl;
        out<<"act6"<<","<<t_act6*1000/(cycles-1)<<std::endl;

        out<<"fc2"<<","<<t_fc2*1000/(cycles-1)<<std::endl;
        out<<"softmax"<<","<<t_softmax*1000/(cycles-1)<<std::endl;

        out<<"alexnet avg : "<<","<<end_time  *1000/(cycles-1)<<std::endl;
        out<<"conv layers: "<<","<<conv_layer*1000/(cycles-1)<<std::endl;
        out<<"act  layers: "<<","<<act_layer*1000/(cycles-1) <<std::endl;
        out<<"pool layers: "<<","<<pool_layer*1000/(cycles-1)<<std::endl;
        out<<"norm layers: "<<","<<norm_layer*1000/(cycles-1)<<std::endl;
        out<<"fc   layers: "<<","<<fc_layer*1000/(cycles-1)  <<std::endl;
        out<<"other layers: "<<","<<other_layer*1000/(cycles-1)<<std::endl;

        std::cout<<"alexnet avg : "<<end_time  *1000/(cycles-1)<<"ms"<<std::endl;
        std::cout<<"conv layers: "<<conv_layer*1000/(cycles-1)<<"ms"<<std::endl;
        std::cout<<"act  layers: "<<act_layer*1000/(cycles-1) <<"ms"<<std::endl;
        std::cout<<"pool layers: "<<pool_layer*1000/(cycles-1)<<"ms"<<std::endl;
        std::cout<<"norm layers: "<<norm_layer*1000/(cycles-1)<<"ms"<<std::endl;
        std::cout<<"fc   layers: "<<fc_layer*1000/(cycles-1)  <<"ms"<<std::endl;
        std::cout<<"other layers: "<<other_layer*1000/(cycles-1)<<"ms"<<std::endl;
    }

private:
    Tensor src{}; 
    Tensor weights_conv0{}, bias_conv0{};
    Tensor out_conv0{};
    Tensor out_act0{}, out_norm0{}, out_pool0{};
    NEGEMMConvolutionLayer conv0{};
    NEActivationLayer act0{};
    NENormalizationLayer norm0{};
    NEPoolingLayer pool0{};
    
    Tensor input_conv1_l{}, input_conv1_r{};
    Tensor weights_conv1_l{}, weights_conv1_r{}, bias_conv1_l{}, bias_conv1_r{};
    Tensor out_conv1_l{}, out_conv1_r{}, out_conv1{};
    Tensor out_act1{}, out_norm1{}, out_pool1{};
    NEGEMMConvolutionLayer conv1_l{}, conv1_r{};
    NEActivationLayer act1{};
    NENormalizationLayer norm1{};
    NEPoolingLayer pool1{};


    Tensor weights_conv2{}, bias_conv2{};
    Tensor out_conv2{};
    Tensor out_act2{};
    NEGEMMConvolutionLayer conv2{};
    NEActivationLayer act2{};

    
    Tensor input_conv3_l{}, input_conv3_r{};
    Tensor weights_conv3_l{}, weights_conv3_r{}, bias_conv3_l{}, bias_conv3_r{};
    Tensor out_conv3_l{}, out_conv3_r{}, out_conv3{};
    Tensor out_act3{};
    NEGEMMConvolutionLayer conv3_l{}, conv3_r{};
    NEActivationLayer act3{};

    Tensor input_conv4_l{}, input_conv4_r{};
    Tensor weights_conv4_l{}, weights_conv4_r{}, bias_conv4_l{}, bias_conv4_r{};
    Tensor out_conv4_l{}, out_conv4_r{}, out_conv4{};
    Tensor out_act4{}, out_pool4{};
    NEGEMMConvolutionLayer conv4_l{}, conv4_r{};
    NEActivationLayer act4{};
    NEPoolingLayer pool4{};

    Tensor fc_input{};
    Tensor weights_fc0{}, bias_fc0{};
    Tensor out_fc0{};
    Tensor out_act5{};
    NEGEMMConvolutionLayer fc0{};
    NEActivationLayer act5{};

    Tensor weights_fc1{}, bias_fc1{};
    Tensor out_fc1{};
    Tensor out_act6{};
    NEGEMMConvolutionLayer fc1{};
    NEActivationLayer act6{};

    Tensor weights_fc2{}, bias_fc2{};
    Tensor out_fc2{};
    Tensor out_softmax{};
    NEGEMMConvolutionLayer fc2{};
    NESoftmaxLayer softmax{};

};

int main(int argc, char **argv)
{
    return utils::run_example<NEONALEXExample>(argc, argv);
}
