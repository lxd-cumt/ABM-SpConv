#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/CLFunctions.h"
#include "arm_compute/runtime/Allocator.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "utils/ImageLoader.h"
#include "utils/Utils.h"
#include <ctime>
#include <sched.h>
#include "support/ToolchainSupport.h"
#include <unistd.h>

#include <iostream>
#include <cstdlib>
#include <memory>
#include <fstream>

using namespace arm_compute;
using namespace utils;
using namespace std;

class NEONALEXExample : public Example
{
public:

    bool do_setup(int argc, char **argv) override
    {
        /*---------------[init_model_alex]-----------------*/
        CLScheduler::get().default_init();

        unsigned int width_src_image  = atoi(argv[1]);
        unsigned int height_src_image = width_src_image;
        unsigned int ifm_src_img      = atoi(argv[2]);

        const TensorShape src_shape(width_src_image, height_src_image, ifm_src_img);
        src.allocator()->init(TensorInfo(src_shape, 1, DataType::F32));


        ///////////////  layer 1 
        // Initialize tensors of conv1
        unsigned int kernel_x_conv1 = atoi(argv[3]);
        unsigned int kernel_y_conv1 = kernel_x_conv1;
        unsigned int ofm_conv1      = atoi(argv[4]);
        int stride= atoi(argv[5])  , pad=0;
        unsigned int out_x_conv1    = (src_shape.x()-kernel_x_conv1 + pad*2)/stride+1;
        unsigned int out_y_conv1    = out_x_conv1;        
        const TensorShape weights_shape_conv1(kernel_x_conv1, kernel_y_conv1, src_shape.z(), ofm_conv1);
        const TensorShape biases_shape_conv1(weights_shape_conv1[3]);
        const TensorShape out_shape_conv1(out_x_conv1, out_y_conv1, weights_shape_conv1[3]);
        weights1.allocator()->init(TensorInfo(weights_shape_conv1, 1, DataType::F32));
        biases1.allocator()->init(TensorInfo(biases_shape_conv1, 1, DataType::F32));
        out_conv1.allocator()->init(TensorInfo(out_shape_conv1, 1, DataType::F32));
        // Initialize tensor of act0
        out_act1.allocator()->init(TensorInfo(out_shape_conv1, 1, DataType::F32));

        /* -----------------------End: [Initialize tensors] */

        /*-----------------BEGIN:[Configure Functions]--------------*/

        /* [Configure functions] */

        ///layer1
        conv1.configure(&src, &weights1, &biases1, &out_conv1, PadStrideInfo(stride, stride, pad, pad));
        act1.configure(&out_conv1, &out_act1, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        
      
        /* -----------------------End: [Configure functions] */

        /*---------------END:[Allocate tensors]---------------*/
        ///layer1
        out_conv1.allocator()->allocate();
        out_act1.allocator()->allocate();
       
        /* -----------------------End: [ Add tensors to memory manager ] */

        /* [Allocate tensors] */

        // Now that the padding requirements are known we can allocate all tensors
        src.allocator()->allocate();
        weights1.allocator()->allocate();

        biases1.allocator()->allocate();
        /* -----------------------End: [Allocate tensors] */
        

        ofstream out( "ymj_generate_conv_Cl.txt", ios::app);
        out<<atoi(argv[1])<<" "<<atoi(argv[2]) <<" "<<atoi(argv[3])<<" "<<atoi(argv[4])<<" "<<atoi(argv[5])<<" ";
        return true;

    }

    void do_run() override
    {
        ofstream out( "ymj_generate_conv_Cl.txt", ios::app);
        CLScheduler::get().sync(); 
        
        auto tbegin = std::chrono::high_resolution_clock::now();

        conv1.run();
        auto tend = std::chrono::high_resolution_clock::now();
        double cost0 = std::chrono::duration_cast<std::chrono::duration<double>>(tend - tbegin).count();
        double cost = cost0;
        std::cout << "conv:" << cost*1000 << std::endl;
        out << cost*1000<<endl;
        out.close();

    }

private:
    // The src tensor should contain the input image
    CLTensor src{};

    CLTensor weights0{},weights1{};
    CLTensor biases0{},biases1{};

    CLTensor out_conv0{};
    CLTensor out_conv1{};
    

    CLTensor out_act0{};
    CLTensor out_act1{};
    

    CLTensor out_norm0{};
    CLTensor out_norm1{};

    CLTensor out_pool0{};
    CLTensor out_pool1{};
    CLTensor out_pool2{};

    // Layers
    CLConvolutionLayer          conv0{};
    CLConvolutionLayer          conv1{};



    CLPoolingLayer              pool0{};
    CLPoolingLayer              pool1{};
    CLPoolingLayer              pool2{};

    CLActivationLayer           act0{};
    CLActivationLayer           act1{};

};

int main(int argc, char **argv)
{
    for(int k =0; k <4 ;k++){
        utils::run_example<NEONALEXExample>(argc, argv);
    }
}
