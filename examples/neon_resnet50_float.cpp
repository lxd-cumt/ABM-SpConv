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



class NEONRESNETExample : public Example

{

public:

	bool do_setup(int argc, char **argv) override

	{

		string data_path="/media/sdcard/ComputeLibrary/data/neon_resnet50_float/";



		NPYLoader npy0;npy0.open(data_path+"input.npy");npy0.init_tensor(src,DataType::F32);

		NPYLoader npy1;npy1.open(data_path+"1.conv1.weight.npy");npy1.init_tensor(weights_con0,DataType::F32);

		NPYLoader npyb1;npyb1.open(data_path+"1.conv1.bias.npy");npyb1.init_tensor(bias_con0,DataType::F32);

		const TensorShape out_shape_con0(112, 112, 64);

		out_con0.allocator()->init(TensorInfo(out_shape_con0, 1, DataType::F32));

		out_act0.allocator()->init(TensorInfo(out_shape_con0,1,DataType::F32));

		TensorShape out_shape_pool0 = out_shape_con0;

		out_shape_pool0.set(0, out_shape_pool0.x() / 2);

		out_shape_pool0.set(1, out_shape_pool0.y() / 2);

		out_pool0.allocator()->init(TensorInfo(out_shape_pool0, 1, DataType::F32));



		NPYLoader npy2;npy2.open(data_path+"1.layer1.0.conv1.weight.npy");npy2.init_tensor(weights_block1r_con0,DataType::F32);

		NPYLoader npyb2;npyb2.open(data_path+"1.layer1.0.conv1.bias.npy");npyb2.init_tensor(bias_block1r_con0,DataType::F32);

		const TensorShape out_shape_block1r_con0(56, 56, 64);

		out_block1r_con0.allocator()->init(TensorInfo(out_shape_block1r_con0, 1, DataType::F32));

		out_block1r_act0.allocator()->init(TensorInfo(out_shape_block1r_con0, 1, DataType::F32));

		NPYLoader npy3;npy3.open(data_path+"1.layer1.0.conv2.weight.npy");npy3.init_tensor(weights_block1r_con1,DataType::F32);

		NPYLoader npyb3;npyb3.open(data_path+"1.layer1.0.conv2.bias.npy");npyb3.init_tensor(bias_block1r_con1,DataType::F32);

		const TensorShape out_shape_block1r_con1(56, 56, 64);

		out_block1r_con1.allocator()->init(TensorInfo(out_shape_block1r_con1, 1, DataType::F32));

		out_block1r_act1.allocator()->init(TensorInfo(out_shape_block1r_con1, 1, DataType::F32));

		NPYLoader npy4;npy4.open(data_path+"1.layer1.0.conv3.weight.npy");npy4.init_tensor(weights_block1r_con2,DataType::F32);

		NPYLoader npyb4;npyb4.open(data_path+"1.layer1.0.conv3.bias.npy");npyb4.init_tensor(bias_block1r_con2,DataType::F32);

		const TensorShape out_shape_block1r_con2(56, 56,256);

		out_block1r_con2.allocator()->init(TensorInfo(out_shape_block1r_con2, 1, DataType::F32));

		NPYLoader npy5;npy5.open(data_path+"1.layer1.0.downsample.0.weight.npy");npy5.init_tensor(weights_block1l_con0,DataType::F32);

		NPYLoader npyb5;npyb5.open(data_path+"1.layer1.0.downsample.0.bias.npy");npyb5.init_tensor(bias_block1l_con0,DataType::F32);

		const TensorShape out_shape_block1l_con0(56, 56, 256);

		out_block1l_con0.allocator()->init(TensorInfo(out_shape_block1l_con0, 1, DataType::F32));

		TensorShape out_shape_block1_0 = out_shape_block1r_con2;

		out_block1_add0.allocator()->init(TensorInfo(out_shape_block1_0, 1, DataType::F32));

		out_block1_act0.allocator()->init(TensorInfo(out_shape_block1_0, 1, DataType::F32));



		NPYLoader npy6;npy6.open(data_path+"1.layer1.1.conv1.weight.npy");npy6.init_tensor(weights_block1r_con3,DataType::F32);

		NPYLoader npyb6;npyb6.open(data_path+"1.layer1.1.conv1.bias.npy");npyb6.init_tensor(bias_block1r_con3,DataType::F32);

		const TensorShape out_shape_block1r_con3(56, 56,64);

		out_block1r_con3.allocator()->init(TensorInfo(out_shape_block1r_con3, 1, DataType::F32));

		out_block1r_act2.allocator()->init(TensorInfo(out_shape_block1r_con3, 1, DataType::F32));

		NPYLoader npy7;npy7.open(data_path+"1.layer1.1.conv2.weight.npy");npy7.init_tensor(weights_block1r_con4,DataType::F32);

		NPYLoader npyb7;npyb7.open(data_path+"1.layer1.1.conv2.bias.npy");npyb7.init_tensor(bias_block1r_con4,DataType::F32);

		const TensorShape out_shape_block1r_con4(56, 56, 64);

		out_block1r_con4.allocator()->init(TensorInfo(out_shape_block1r_con4, 1, DataType::F32));

		out_block1r_act3.allocator()->init(TensorInfo(out_shape_block1r_con4, 1, DataType::F32));

		NPYLoader npy8;npy8.open(data_path+"1.layer1.1.conv3.weight.npy");npy8.init_tensor(weights_block1r_con5,DataType::F32);

		NPYLoader npyb8;npyb8.open(data_path+"1.layer1.1.conv3.bias.npy");npyb8.init_tensor(bias_block1r_con5,DataType::F32);

		const TensorShape out_shape_block1r_con5(56, 56,256);

		out_block1r_con5.allocator()->init(TensorInfo(out_shape_block1r_con5, 1, DataType::F32));

		TensorShape out_shape_block1_1 = out_shape_block1r_con5;

		out_block1_add1.allocator()->init(TensorInfo(out_shape_block1_1, 1, DataType::F32));

		out_block1_act1.allocator()->init(TensorInfo(out_shape_block1_1, 1, DataType::F32));



		NPYLoader npy9;npy9.open(data_path+"1.layer1.2.conv1.weight.npy");npy9.init_tensor(weights_block1r_con6,DataType::F32);

		NPYLoader npyb9;npyb9.open(data_path+"1.layer1.2.conv1.bias.npy");npyb9.init_tensor(bias_block1r_con6,DataType::F32);

		const TensorShape out_shape_block1r_con6(56, 56, 64);

		out_block1r_con6.allocator()->init(TensorInfo(out_shape_block1r_con6, 1, DataType::F32));

		out_block1r_act4.allocator()->init(TensorInfo(out_shape_block1r_con6, 1, DataType::F32));

		NPYLoader npy10;npy10.open(data_path+"1.layer1.2.conv2.weight.npy");npy10.init_tensor(weights_block1r_con7,DataType::F32);

		NPYLoader npyb10;npyb10.open(data_path+"1.layer1.2.conv2.bias.npy");npyb10.init_tensor(bias_block1r_con7,DataType::F32);

		const TensorShape out_shape_block1r_con7(56, 56,64);

		out_block1r_con7.allocator()->init(TensorInfo(out_shape_block1r_con7, 1, DataType::F32));

		out_block1r_act5.allocator()->init(TensorInfo(out_shape_block1r_con7, 1, DataType::F32));

		NPYLoader npy11;npy11.open(data_path+"1.layer1.2.conv3.weight.npy");npy11.init_tensor(weights_block1r_con8,DataType::F32);

		NPYLoader npyb11;npyb11.open(data_path+"1.layer1.2.conv3.bias.npy");npyb11.init_tensor(bias_block1r_con8,DataType::F32);

		const TensorShape out_shape_block1r_con8(56, 56,256);

		out_block1r_con8.allocator()->init(TensorInfo(out_shape_block1r_con8, 1, DataType::F32));

		TensorShape out_shape_block1_2 = out_shape_block1r_con8;

		out_block1_add2.allocator()->init(TensorInfo(out_shape_block1_2, 1, DataType::F32));

		out_block1_act2.allocator()->init(TensorInfo(out_shape_block1_2, 1, DataType::F32));





        NPYLoader npy12;npy12.open(data_path+"1.layer2.0.conv1.weight.npy");npy12.init_tensor(weights_block2r_con0,DataType::F32);

		NPYLoader npyb12;npyb12.open(data_path+"1.layer2.0.conv1.bias.npy");npyb12.init_tensor(bias_block2r_con0,DataType::F32);

		const TensorShape out_shape_block2r_con0(28, 28, 128);

		out_block2r_con0.allocator()->init(TensorInfo(out_shape_block2r_con0, 1, DataType::F32));

		out_block2r_act0.allocator()->init(TensorInfo(out_shape_block2r_con0, 1, DataType::F32));

		NPYLoader npy13;npy13.open(data_path+"1.layer2.0.conv2.weight.npy");npy13.init_tensor(weights_block2r_con1,DataType::F32);

		NPYLoader npyb13;npyb13.open(data_path+"1.layer2.0.conv2.bias.npy");npyb13.init_tensor(bias_block2r_con1,DataType::F32);

		const TensorShape out_shape_block2r_con1(28, 28, 128);

		out_block2r_con1.allocator()->init(TensorInfo(out_shape_block2r_con1, 1, DataType::F32));

		out_block2r_act1.allocator()->init(TensorInfo(out_shape_block2r_con1, 1, DataType::F32));

		NPYLoader npy14;npy14.open(data_path+"1.layer2.0.conv3.weight.npy");npy14.init_tensor(weights_block2r_con2,DataType::F32);

		NPYLoader npyb14;npyb14.open(data_path+"1.layer2.0.conv3.bias.npy");npyb14.init_tensor(bias_block2r_con2,DataType::F32);

		const TensorShape out_shape_block2r_con2(28, 28, 512);

		out_block2r_con2.allocator()->init(TensorInfo(out_shape_block2r_con2, 1, DataType::F32));

		NPYLoader npy15;npy15.open(data_path+"1.layer2.0.downsample.0.weight.npy");npy15.init_tensor(weights_block2l_con0,DataType::F32);

		NPYLoader npyb15;npyb15.open(data_path+"1.layer2.0.downsample.0.bias.npy");npyb15.init_tensor(bias_block2l_con0,DataType::F32);

		const TensorShape out_shape_block2l_con0(28, 28, 512);

		out_block2l_con0.allocator()->init(TensorInfo(out_shape_block2l_con0, 1, DataType::F32));

		TensorShape out_shape_block2_0 = out_shape_block2r_con2;

		out_block2_add0.allocator()->init(TensorInfo(out_shape_block2_0, 1, DataType::F32));

		out_block2_act0.allocator()->init(TensorInfo(out_shape_block2_0, 1, DataType::F32));



        NPYLoader npy16;npy16.open(data_path+"1.layer2.1.conv1.weight.npy");npy16.init_tensor(weights_block2r_con3,DataType::F32);

		NPYLoader npyb16;npyb16.open(data_path+"1.layer2.1.conv1.bias.npy");npyb16.init_tensor(bias_block2r_con3,DataType::F32);

		const TensorShape out_shape_block2r_con3(28, 28, 128);

		out_block2r_con3.allocator()->init(TensorInfo(out_shape_block2r_con3, 1, DataType::F32));

		out_block2r_act2.allocator()->init(TensorInfo(out_shape_block2r_con3, 1, DataType::F32));

		NPYLoader npy17;npy17.open(data_path+"1.layer2.1.conv2.weight.npy");npy17.init_tensor(weights_block2r_con4,DataType::F32);

		NPYLoader npyb17;npyb17.open(data_path+"1.layer2.1.conv2.bias.npy");npyb17.init_tensor(bias_block2r_con4,DataType::F32);

		const TensorShape out_shape_block2r_con4(28, 28,128);

		out_block2r_con4.allocator()->init(TensorInfo(out_shape_block2r_con4, 1, DataType::F32));

		out_block2r_act3.allocator()->init(TensorInfo(out_shape_block2r_con4, 1, DataType::F32));

		NPYLoader npy18;npy18.open(data_path+"1.layer2.1.conv3.weight.npy");npy18.init_tensor(weights_block2r_con5,DataType::F32);

		NPYLoader npyb18;npyb18.open(data_path+"1.layer2.1.conv3.bias.npy");npyb18.init_tensor(bias_block2r_con5,DataType::F32);

		const TensorShape out_shape_block2r_con5(28, 28,512);

		out_block2r_con5.allocator()->init(TensorInfo(out_shape_block2r_con5, 1, DataType::F32));

		TensorShape out_shape_block2_1 = out_shape_block2r_con5;

		out_block2_add1.allocator()->init(TensorInfo(out_shape_block2_1, 1, DataType::F32));

		out_block2_act1.allocator()->init(TensorInfo(out_shape_block2_1, 1, DataType::F32));



        NPYLoader npy19;npy19.open(data_path+"1.layer2.2.conv1.weight.npy");npy19.init_tensor(weights_block2r_con6,DataType::F32);

		 NPYLoader npyb19;npyb19.open(data_path+"1.layer2.2.conv1.bias.npy");npyb19.init_tensor(bias_block2r_con6,DataType::F32);

		const TensorShape out_shape_block2r_con6(28, 28, 128);

		out_block2r_con6.allocator()->init(TensorInfo(out_shape_block2r_con6, 1, DataType::F32));

		out_block2r_act4.allocator()->init(TensorInfo(out_shape_block2r_con6, 1, DataType::F32));

		NPYLoader npy20;npy20.open(data_path+"1.layer2.2.conv2.weight.npy");npy20.init_tensor(weights_block2r_con7,DataType::F32);

		NPYLoader npyb20;npyb20.open(data_path+"1.layer2.2.conv2.bias.npy");npyb20.init_tensor(bias_block2r_con7,DataType::F32);

		const TensorShape out_shape_block2r_con7(28, 28, 128);

		out_block2r_con7.allocator()->init(TensorInfo(out_shape_block2r_con7, 1, DataType::F32));

		out_block2r_act5.allocator()->init(TensorInfo(out_shape_block2r_con7, 1, DataType::F32));

		NPYLoader npy21;npy21.open(data_path+"1.layer2.2.conv3.weight.npy");npy21.init_tensor(weights_block2r_con8,DataType::F32);

		NPYLoader npyb21;npyb21.open(data_path+"1.layer2.2.conv3.bias.npy");npyb21.init_tensor(bias_block2r_con8,DataType::F32);

		const TensorShape out_shape_block2r_con8(28, 28, 512);

		out_block2r_con8.allocator()->init(TensorInfo(out_shape_block2r_con8, 1, DataType::F32));

		TensorShape out_shape_block2_2 = out_shape_block2r_con8;

		out_block2_add2.allocator()->init(TensorInfo(out_shape_block2_2, 1, DataType::F32));

		out_block2_act2.allocator()->init(TensorInfo(out_shape_block2_2, 1, DataType::F32));



        NPYLoader npy22;npy22.open(data_path+"1.layer2.3.conv1.weight.npy");npy22.init_tensor(weights_block2r_con9,DataType::F32);

		NPYLoader npyb22;npyb22.open(data_path+"1.layer2.3.conv1.bias.npy");npyb22.init_tensor(bias_block2r_con9,DataType::F32);

		const TensorShape out_shape_block2r_con9(28, 28,128);

		out_block2r_con9.allocator()->init(TensorInfo(out_shape_block2r_con9, 1, DataType::F32));

		out_block2r_act6.allocator()->init(TensorInfo(out_shape_block2r_con9, 1, DataType::F32));

		NPYLoader npy23;npy23.open(data_path+"1.layer2.3.conv2.weight.npy");npy23.init_tensor(weights_block2r_con10,DataType::F32);

		NPYLoader npyb23;npyb23.open(data_path+"1.layer2.3.conv2.bias.npy");npyb23.init_tensor(bias_block2r_con10,DataType::F32);

		const TensorShape out_shape_block2r_con10(28, 28, 128);

		out_block2r_con10.allocator()->init(TensorInfo(out_shape_block2r_con10, 1, DataType::F32));

		out_block2r_act7.allocator()->init(TensorInfo(out_shape_block2r_con10, 1, DataType::F32));

		NPYLoader npy24;npy24.open(data_path+"1.layer2.3.conv3.weight.npy");npy24.init_tensor(weights_block2r_con11,DataType::F32);

		NPYLoader npyb24;npyb24.open(data_path+"1.layer2.3.conv3.bias.npy");npyb24.init_tensor(bias_block2r_con11,DataType::F32);

		const TensorShape out_shape_block2r_con11(28, 28, 512);

		out_block2r_con11.allocator()->init(TensorInfo(out_shape_block2r_con11, 1, DataType::F32));

		TensorShape out_shape_block2_3 = out_shape_block2r_con11;

		out_block2_add3.allocator()->init(TensorInfo(out_shape_block2_3, 1, DataType::F32));

		out_block2_act3.allocator()->init(TensorInfo(out_shape_block2_3, 1, DataType::F32));





        NPYLoader npy25;npy25.open(data_path+"1.layer3.0.conv1.weight.npy");npy25.init_tensor(weights_block3r_con0,DataType::F32);

		NPYLoader npyb25;npyb25.open(data_path+"1.layer3.0.conv1.bias.npy");npyb25.init_tensor(bias_block3r_con0,DataType::F32);

		const TensorShape out_shape_block3r_con0(14, 14, 256);

		out_block3r_con0.allocator()->init(TensorInfo(out_shape_block3r_con0, 1, DataType::F32));

		out_block3r_act0.allocator()->init(TensorInfo(out_shape_block3r_con0, 1, DataType::F32));

		NPYLoader npy26;npy26.open(data_path+"1.layer3.0.conv2.weight.npy");npy26.init_tensor(weights_block3r_con1,DataType::F32);

		NPYLoader npyb26;npyb26.open(data_path+"1.layer3.0.conv2.bias.npy");npyb26.init_tensor(bias_block3r_con1,DataType::F32);

		const TensorShape out_shape_block3r_con1(14, 14, 256);

		out_block3r_con1.allocator()->init(TensorInfo(out_shape_block3r_con1, 1, DataType::F32));

		out_block3r_act1.allocator()->init(TensorInfo(out_shape_block3r_con1, 1, DataType::F32));

		NPYLoader npy27;npy27.open(data_path+"1.layer3.0.conv3.weight.npy");npy27.init_tensor(weights_block3r_con2,DataType::F32);

		NPYLoader npyb27;npyb27.open(data_path+"1.layer3.0.conv3.bias.npy");npyb27.init_tensor(bias_block3r_con2,DataType::F32);

		const TensorShape out_shape_block3r_con2(14, 14, 1024);

		out_block3r_con2.allocator()->init(TensorInfo(out_shape_block3r_con2, 1, DataType::F32));

		NPYLoader npy28;npy28.open(data_path+"1.layer3.0.downsample.0.weight.npy");npy28.init_tensor(weights_block3l_con0,DataType::F32);

		NPYLoader npyb28;npyb28.open(data_path+"1.layer3.0.downsample.0.bias.npy");npyb28.init_tensor(bias_block3l_con0,DataType::F32);

		const TensorShape out_shape_block3l_con0(14, 14, 1024);

		out_block3l_con0.allocator()->init(TensorInfo(out_shape_block3l_con0, 1, DataType::F32));

		TensorShape out_shape_block3_0 = out_shape_block3r_con2;

		out_block3_add0.allocator()->init(TensorInfo(out_shape_block3_0, 1, DataType::F32));

		out_block3_act0.allocator()->init(TensorInfo(out_shape_block3_0, 1, DataType::F32));



		NPYLoader npy29;npy29.open(data_path+"1.layer3.1.conv1.weight.npy");npy29.init_tensor(weights_block3r_con3,DataType::F32);

		NPYLoader npyb29;npyb29.open(data_path+"1.layer3.1.conv1.bias.npy");npyb29.init_tensor(bias_block3r_con3,DataType::F32);

		const TensorShape out_shape_block3r_con3(14, 14,256);

		out_block3r_con3.allocator()->init(TensorInfo(out_shape_block3r_con3, 1, DataType::F32));

		out_block3r_act2.allocator()->init(TensorInfo(out_shape_block3r_con3, 1, DataType::F32));

		NPYLoader npy30;npy30.open(data_path+"1.layer3.1.conv2.weight.npy");npy30.init_tensor(weights_block3r_con4,DataType::F32);

		NPYLoader npyb30;npyb30.open(data_path+"1.layer3.1.conv2.bias.npy");npyb30.init_tensor(bias_block3r_con4,DataType::F32);

		const TensorShape out_shape_block3r_con4(14, 14,256);

		out_block3r_con4.allocator()->init(TensorInfo(out_shape_block3r_con4, 1, DataType::F32));

		out_block3r_act3.allocator()->init(TensorInfo(out_shape_block3r_con4, 1, DataType::F32));

		NPYLoader npy31;npy31.open(data_path+"1.layer3.1.conv3.weight.npy");npy31.init_tensor(weights_block3r_con5,DataType::F32);

		NPYLoader npyb31;npyb31.open(data_path+"1.layer3.1.conv3.bias.npy");npyb31.init_tensor(bias_block3r_con5,DataType::F32);

		const TensorShape out_shape_block3r_con5(14, 14, 1024);

		out_block3r_con5.allocator()->init(TensorInfo(out_shape_block3r_con5, 1, DataType::F32));

		TensorShape out_shape_block3_1 = out_shape_block3r_con5;

		out_block3_add1.allocator()->init(TensorInfo(out_shape_block3_1, 1, DataType::F32));

		out_block3_act1.allocator()->init(TensorInfo(out_shape_block3_1, 1, DataType::F32));



		NPYLoader npy32;npy32.open(data_path+"1.layer3.2.conv1.weight.npy");npy32.init_tensor(weights_block3r_con6,DataType::F32);

		NPYLoader npyb32;npyb32.open(data_path+"1.layer3.2.conv1.bias.npy");npyb32.init_tensor(bias_block3r_con6,DataType::F32);

		const TensorShape out_shape_block3r_con6(14, 14, 256);

		out_block3r_con6.allocator()->init(TensorInfo(out_shape_block3r_con6, 1, DataType::F32));

		out_block3r_act4.allocator()->init(TensorInfo(out_shape_block3r_con6, 1, DataType::F32));

		NPYLoader npy33;npy33.open(data_path+"1.layer3.2.conv2.weight.npy");npy33.init_tensor(weights_block3r_con7,DataType::F32);

		NPYLoader npyb33;npyb33.open(data_path+"1.layer3.2.conv2.bias.npy");npyb33.init_tensor(bias_block3r_con7,DataType::F32);

		const TensorShape out_shape_block3r_con7(14, 14, 256);

		out_block3r_con7.allocator()->init(TensorInfo(out_shape_block3r_con7, 1, DataType::F32));

		out_block3r_act5.allocator()->init(TensorInfo(out_shape_block3r_con7, 1, DataType::F32));

		NPYLoader npy34;npy34.open(data_path+"1.layer3.2.conv3.weight.npy");npy34.init_tensor(weights_block3r_con8,DataType::F32);

		NPYLoader npyb34;npyb34.open(data_path+"1.layer3.2.conv3.bias.npy");npyb34.init_tensor(bias_block3r_con8,DataType::F32);

		const TensorShape out_shape_block3r_con8(14, 14, 1024);

		out_block3r_con8.allocator()->init(TensorInfo(out_shape_block3r_con8, 1, DataType::F32));

		TensorShape out_shape_block3_2 = out_shape_block3r_con8;

		out_block3_add2.allocator()->init(TensorInfo(out_shape_block3_2, 1, DataType::F32));

		out_block3_act2.allocator()->init(TensorInfo(out_shape_block3_2, 1, DataType::F32));



		NPYLoader npy35;npy35.open(data_path+"1.layer3.3.conv1.weight.npy");npy35.init_tensor(weights_block3r_con9,DataType::F32);

		NPYLoader npyb35;npyb35.open(data_path+"1.layer3.3.conv1.bias.npy");npyb35.init_tensor(bias_block3r_con9,DataType::F32);

		const TensorShape out_shape_block3r_con9(14, 14, 256);

		out_block3r_con9.allocator()->init(TensorInfo(out_shape_block3r_con9, 1, DataType::F32));

		out_block3r_act6.allocator()->init(TensorInfo(out_shape_block3r_con9, 1, DataType::F32));

		NPYLoader npy36;npy36.open(data_path+"1.layer3.3.conv2.weight.npy");npy36.init_tensor(weights_block3r_con10,DataType::F32);

		NPYLoader npyb36;npyb36.open(data_path+"1.layer3.3.conv2.bias.npy");npyb36.init_tensor(bias_block3r_con10,DataType::F32);

		const TensorShape out_shape_block3r_con10(14, 14, 256);

		out_block3r_con10.allocator()->init(TensorInfo(out_shape_block3r_con10, 1, DataType::F32));

		out_block3r_act7.allocator()->init(TensorInfo(out_shape_block3r_con10, 1, DataType::F32));

		NPYLoader npy37;npy37.open(data_path+"1.layer3.3.conv3.weight.npy");npy37.init_tensor(weights_block3r_con11,DataType::F32);

		NPYLoader npyb37;npyb37.open(data_path+"1.layer3.3.conv3.bias.npy");npyb37.init_tensor(bias_block3r_con11,DataType::F32);

		const TensorShape out_shape_block3r_con11(14, 14, 1024);

		out_block3r_con11.allocator()->init(TensorInfo(out_shape_block3r_con11, 1, DataType::F32));

		TensorShape out_shape_block3_3 = out_shape_block3r_con11;

		out_block3_add3.allocator()->init(TensorInfo(out_shape_block3_3, 1, DataType::F32));

		out_block3_act3.allocator()->init(TensorInfo(out_shape_block3_3, 1, DataType::F32));



		NPYLoader npy38;npy38.open(data_path+"1.layer3.4.conv1.weight.npy");npy38.init_tensor(weights_block3r_con12,DataType::F32);

		NPYLoader npyb38;npyb38.open(data_path+"1.layer3.4.conv1.bias.npy");npyb38.init_tensor(bias_block3r_con12,DataType::F32);

		const TensorShape out_shape_block3r_con12(14, 14, 256);

		out_block3r_con12.allocator()->init(TensorInfo(out_shape_block3r_con12, 1, DataType::F32));

		out_block3r_act8.allocator()->init(TensorInfo(out_shape_block3r_con12, 1, DataType::F32));

		NPYLoader npy39;npy39.open(data_path+"1.layer3.4.conv2.weight.npy");npy39.init_tensor(weights_block3r_con13,DataType::F32);

		NPYLoader npyb39;npyb39.open(data_path+"1.layer3.4.conv2.bias.npy");npyb39.init_tensor(bias_block3r_con13,DataType::F32);

		const TensorShape out_shape_block3r_con13(14, 14, 256);

		out_block3r_con13.allocator()->init(TensorInfo(out_shape_block3r_con13, 1, DataType::F32));

		out_block3r_act9.allocator()->init(TensorInfo(out_shape_block3r_con13, 1, DataType::F32));

		NPYLoader npy40;npy40.open(data_path+"1.layer3.4.conv3.weight.npy");npy40.init_tensor(weights_block3r_con14,DataType::F32);

		NPYLoader npyb40;npyb40.open(data_path+"1.layer3.4.conv3.bias.npy");npyb40.init_tensor(bias_block3r_con14,DataType::F32);

		const TensorShape out_shape_block3r_con14(14, 14, 1024);

		out_block3r_con14.allocator()->init(TensorInfo(out_shape_block3r_con14, 1, DataType::F32));

		TensorShape out_shape_block3_4 = out_shape_block3r_con14;

		out_block3_add4.allocator()->init(TensorInfo(out_shape_block3_4, 1, DataType::F32));

		out_block3_act4.allocator()->init(TensorInfo(out_shape_block3_4, 1, DataType::F32));



		NPYLoader npy41;npy41.open(data_path+"1.layer3.5.conv1.weight.npy");npy41.init_tensor(weights_block3r_con15,DataType::F32);

		NPYLoader npyb41;npyb41.open(data_path+"1.layer3.5.conv1.bias.npy");npyb41.init_tensor(bias_block3r_con15,DataType::F32);

		const TensorShape out_shape_block3r_con15(14, 14,256);

		out_block3r_con15.allocator()->init(TensorInfo(out_shape_block3r_con15, 1, DataType::F32));

		out_block3r_act10.allocator()->init(TensorInfo(out_shape_block3r_con15, 1, DataType::F32));

		NPYLoader npy42;npy42.open(data_path+"1.layer3.5.conv2.weight.npy");npy42.init_tensor(weights_block3r_con16,DataType::F32);

		NPYLoader npyb42;npyb42.open(data_path+"1.layer3.5.conv2.bias.npy");npyb42.init_tensor(bias_block3r_con16,DataType::F32);

		const TensorShape out_shape_block3r_con16(14, 14, 256);

		out_block3r_con16.allocator()->init(TensorInfo(out_shape_block3r_con16, 1, DataType::F32));

		out_block3r_act11.allocator()->init(TensorInfo(out_shape_block3r_con16, 1, DataType::F32));

		NPYLoader npy43;npy43.open(data_path+"1.layer3.5.conv3.weight.npy");npy43.init_tensor(weights_block3r_con17,DataType::F32);

		NPYLoader npyb43;npyb43.open(data_path+"1.layer3.5.conv3.bias.npy");npyb43.init_tensor(bias_block3r_con17,DataType::F32);

		const TensorShape out_shape_block3r_con17(14, 14, 1024);

		out_block3r_con17.allocator()->init(TensorInfo(out_shape_block3r_con17, 1, DataType::F32));

		TensorShape out_shape_block3_5 = out_shape_block3r_con17;

		out_block3_add5.allocator()->init(TensorInfo(out_shape_block3_5, 1, DataType::F32));

		out_block3_act5.allocator()->init(TensorInfo(out_shape_block3_5, 1, DataType::F32));



		NPYLoader npy44;npy44.open(data_path+"1.layer4.0.conv1.weight.npy");npy44.init_tensor(weights_block4r_con0,DataType::F32);

		NPYLoader npyb44;npyb44.open(data_path+"1.layer4.0.conv1.bias.npy");npyb44.init_tensor(bias_block4r_con0,DataType::F32);

		const TensorShape out_shape_block4r_con0(7, 7, 512);

		out_block4r_con0.allocator()->init(TensorInfo(out_shape_block4r_con0, 1, DataType::F32));

		out_block4r_act0.allocator()->init(TensorInfo(out_shape_block4r_con0, 1, DataType::F32));

		NPYLoader npy45;npy45.open(data_path+"1.layer4.0.conv2.weight.npy");npy45.init_tensor(weights_block4r_con1,DataType::F32);

		NPYLoader npyb45;npyb45.open(data_path+"1.layer4.0.conv2.bias.npy");npyb45.init_tensor(bias_block4r_con1,DataType::F32);

		const TensorShape out_shape_block4r_con1(7, 7,512);

		out_block4r_con1.allocator()->init(TensorInfo(out_shape_block4r_con1, 1, DataType::F32));

		out_block4r_act1.allocator()->init(TensorInfo(out_shape_block4r_con1, 1, DataType::F32));

		NPYLoader npy46;npy46.open(data_path+"1.layer4.0.conv3.weight.npy");npy46.init_tensor(weights_block4r_con2,DataType::F32);

		NPYLoader npyb46;npyb46.open(data_path+"1.layer4.0.conv3.bias.npy");npyb46.init_tensor(bias_block4r_con2,DataType::F32);

		const TensorShape out_shape_block4r_con2(7, 7, 2048);

		out_block4r_con2.allocator()->init(TensorInfo(out_shape_block4r_con2, 1, DataType::F32));

		NPYLoader npy47;npy47.open(data_path+"1.layer4.0.downsample.0.weight.npy");npy47.init_tensor(weights_block4l_con0,DataType::F32);

		NPYLoader npyb47;npyb47.open(data_path+"1.layer4.0.downsample.0.bias.npy");npyb47.init_tensor(bias_block4l_con0,DataType::F32);

		const TensorShape out_shape_block4l_con0(7, 7, 2048);

		out_block4l_con0.allocator()->init(TensorInfo(out_shape_block4l_con0, 1, DataType::F32));

		TensorShape out_shape_block4_0 = out_shape_block4r_con2;

		out_block4_add0.allocator()->init(TensorInfo(out_shape_block4_0, 1, DataType::F32));

		out_block4_act0.allocator()->init(TensorInfo(out_shape_block4_0, 1, DataType::F32));



		NPYLoader npy48;npy48.open(data_path+"1.layer4.1.conv1.weight.npy");npy48.init_tensor(weights_block4r_con3,DataType::F32);

		NPYLoader npyb48;npyb48.open(data_path+"1.layer4.1.conv1.bias.npy");npyb48.init_tensor(bias_block4r_con3,DataType::F32);

		const TensorShape out_shape_block4r_con3(7, 7, 512);

		out_block4r_con3.allocator()->init(TensorInfo(out_shape_block4r_con3, 1, DataType::F32));

		out_block4r_act2.allocator()->init(TensorInfo(out_shape_block4r_con3, 1, DataType::F32));

		NPYLoader npy49;npy49.open(data_path+"1.layer4.1.conv2.weight.npy");npy49.init_tensor(weights_block4r_con4,DataType::F32);

		NPYLoader npyb49;npyb49.open(data_path+"1.layer4.1.conv2.bias.npy");npyb49.init_tensor(bias_block4r_con4,DataType::F32);

		const TensorShape out_shape_block4r_con4(7, 7, 512);

		out_block4r_con4.allocator()->init(TensorInfo(out_shape_block4r_con4, 1, DataType::F32));

		out_block4r_act3.allocator()->init(TensorInfo(out_shape_block4r_con4, 1, DataType::F32));

		NPYLoader npy50;npy50.open(data_path+"1.layer4.1.conv3.weight.npy");npy50.init_tensor(weights_block4r_con5,DataType::F32);

		NPYLoader npyb50;npyb50.open(data_path+"1.layer4.1.conv3.bias.npy");npyb50.init_tensor(bias_block4r_con5,DataType::F32);

		const TensorShape out_shape_block4r_con5(7, 7, 2048);

		out_block4r_con5.allocator()->init(TensorInfo(out_shape_block4r_con5, 1, DataType::F32));

		TensorShape out_shape_block4_1 = out_shape_block4r_con5;

		out_block4_add1.allocator()->init(TensorInfo(out_shape_block4_1, 1, DataType::F32));

		out_block4_act1.allocator()->init(TensorInfo(out_shape_block4_1, 1, DataType::F32));



		NPYLoader npy51;npy51.open(data_path+"1.layer4.2.conv1.weight.npy");npy51.init_tensor(weights_block4r_con6,DataType::F32);

		NPYLoader npyb51;npyb51.open(data_path+"1.layer4.2.conv1.bias.npy");npyb51.init_tensor(bias_block4r_con6,DataType::F32);

		const TensorShape out_shape_block4r_con6(7, 7, 512);

		out_block4r_con6.allocator()->init(TensorInfo(out_shape_block4r_con6, 1, DataType::F32));

		out_block4r_act4.allocator()->init(TensorInfo(out_shape_block4r_con6, 1, DataType::F32));

		NPYLoader npy52;npy52.open(data_path+"1.layer4.2.conv2.weight.npy");npy52.init_tensor(weights_block4r_con7,DataType::F32);

		NPYLoader npyb52;npyb52.open(data_path+"1.layer4.2.conv2.bias.npy");npyb52.init_tensor(bias_block4r_con7,DataType::F32);

		const TensorShape out_shape_block4r_con7(7, 7, 512);

		out_block4r_con7.allocator()->init(TensorInfo(out_shape_block4r_con7, 1, DataType::F32));

		out_block4r_act5.allocator()->init(TensorInfo(out_shape_block4r_con7, 1, DataType::F32));

		NPYLoader npy53;npy53.open(data_path+"1.layer4.2.conv3.weight.npy");npy53.init_tensor(weights_block4r_con8,DataType::F32);

		NPYLoader npyb53;npyb53.open(data_path+"1.layer4.2.conv3.bias.npy");npyb53.init_tensor(bias_block4r_con8,DataType::F32);

		const TensorShape out_shape_block4r_con8(7, 7, 2048);

		out_block4r_con8.allocator()->init(TensorInfo(out_shape_block4r_con8, 1, DataType::F32));

		TensorShape out_shape_block4_2 = out_shape_block4r_con8;

		out_block4_add2.allocator()->init(TensorInfo(out_shape_block4_2, 1, DataType::F32));

		out_block4_act2.allocator()->init(TensorInfo(out_shape_block4_2, 1, DataType::F32));



		TensorShape out_shape_pool1 = out_shape_block4_2;

		out_shape_pool1.set(0, out_shape_pool1.x() / 7);

		out_shape_pool1.set(1, out_shape_pool1.y() / 7);

		out_pool1.allocator()->init(TensorInfo(out_shape_pool1, 1, DataType::F32));         

		NPYLoader npy54;npy54.open(data_path+"1.fc.weight2.npy");npy54.init_tensor(weights_con1,DataType::F32);

		NPYLoader npyb54;npyb54.open(data_path+"1.fc.bias.npy");npyb54.init_tensor(bias_con1,DataType::F32);

		const TensorShape out_shape_con1(1, 1, 1000);

		out_con1.allocator()->init(TensorInfo(out_shape_con1, 1, DataType::F32));

		const TensorShape out_shape_flatten(out_shape_con1.x()*out_shape_con1.y()*out_shape_con1.z(),0);                     

		out_flatten.allocator()->init(TensorInfo(out_shape_flatten, 1, DataType::F32));

		const TensorShape out_shape_softmax(out_shape_flatten.x());

		out_softmax.allocator()->init(TensorInfo(out_shape_softmax, 1, DataType::F32));



		con0.configure(&src, &weights_con0,&bias_con0, &out_con0, PadStrideInfo(2, 2, 3, 3));

		act0.configure(&out_con0, &out_act0, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

		pool0.configure(&out_act0, &out_pool0, PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 1, 0, 1, DimensionRoundingType::FLOOR)));

		

		block1r_con0.configure(&out_pool0, &weights_block1r_con0, &bias_block1r_con0, &out_block1r_con0, PadStrideInfo(1, 1, 0, 0));

		block1r_act0.configure(&out_block1r_con0, &out_block1r_act0, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

		block1r_con1.configure(&out_block1r_act0, &weights_block1r_con1, &bias_block1r_con1, &out_block1r_con1, PadStrideInfo(1, 1, 1, 1));

		block1r_act1.configure(&out_block1r_con1, &out_block1r_act1, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

		block1r_con2.configure(&out_block1r_act1, &weights_block1r_con2, &bias_block1r_con2, &out_block1r_con2, PadStrideInfo(1, 1, 0, 0));

		block1l_con0.configure(&out_pool0, &weights_block1l_con0, &bias_block1l_con0, &out_block1l_con0, PadStrideInfo(1, 1, 0, 0));

		block1_add0.configure(&out_block1r_con2, &out_block1l_con0, &out_block1_add0,ConvertPolicy::SATURATE);

		block1_act0.configure(&out_block1_add0, &out_block1_act0, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

		

		block1r_con3.configure(&out_block1_act0, &weights_block1r_con3, &bias_block1r_con3, &out_block1r_con3, PadStrideInfo(1, 1, 0, 0));

		block1r_act2.configure(&out_block1r_con3, &out_block1r_act2, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

		block1r_con4.configure(&out_block1r_act2, &weights_block1r_con4, &bias_block1r_con4, &out_block1r_con4, PadStrideInfo(1, 1, 1, 1));

		block1r_act3.configure(&out_block1r_con4, &out_block1r_act3, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

		block1r_con5.configure(&out_block1r_act3, &weights_block1r_con5, &bias_block1r_con5, &out_block1r_con5, PadStrideInfo(1, 1, 0, 0));

		block1_add1.configure(&out_block1r_con5, &out_block1_act0, &out_block1_add1,ConvertPolicy::SATURATE);

		block1_act1.configure(&out_block1_add1, &out_block1_act1, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

		

		block1r_con6.configure(&out_block1_act1, &weights_block1r_con6, &bias_block1r_con6, &out_block1r_con6, PadStrideInfo(1, 1, 0, 0));

		block1r_act4.configure(&out_block1r_con6, &out_block1r_act4, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

		block1r_con7.configure(&out_block1r_act4, &weights_block1r_con7, &bias_block1r_con7, &out_block1r_con7, PadStrideInfo(1, 1, 1, 1));

		block1r_act5.configure(&out_block1r_con7, &out_block1r_act5, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

		block1r_con8.configure(&out_block1r_act5, &weights_block1r_con8, &bias_block1r_con8, &out_block1r_con8, PadStrideInfo(1, 1, 0, 0));

		block1_add2.configure(&out_block1r_con8, &out_block1_act1, &out_block1_add2,ConvertPolicy::SATURATE);

		block1_act2.configure(&out_block1_add2, &out_block1_act2, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));





		block2r_con0.configure(&out_block1_act2, &weights_block2r_con0, &bias_block2r_con0, &out_block2r_con0, PadStrideInfo(2, 2, 0, 0));

		block2r_act0.configure(&out_block2r_con0, &out_block2r_act0, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

		block2r_con1.configure(&out_block2r_act0, &weights_block2r_con1, &bias_block2r_con1, &out_block2r_con1, PadStrideInfo(1, 1, 1, 1));

		block2r_act1.configure(&out_block2r_con1, &out_block2r_act1, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

		block2r_con2.configure(&out_block2r_act1, &weights_block2r_con2, &bias_block2r_con2, &out_block2r_con2, PadStrideInfo(1, 1, 0, 0));

		block2l_con0.configure(&out_block1_act2, &weights_block2l_con0, &bias_block2l_con0, &out_block2l_con0, PadStrideInfo(2, 2, 0, 0));

		block2_add0.configure(&out_block2r_con2, &out_block2l_con0, &out_block2_add0, ConvertPolicy::SATURATE);

		block2_act0.configure(&out_block2_add0, &out_block2_act0, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

		

		block2r_con3.configure(&out_block2_act0, &weights_block2r_con3, &bias_block2r_con3, &out_block2r_con3, PadStrideInfo(1, 1, 0, 0));

		block2r_act2.configure(&out_block2r_con3, &out_block2r_act2, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

		block2r_con4.configure(&out_block2r_act2, &weights_block2r_con4, &bias_block2r_con4, &out_block2r_con4, PadStrideInfo(1, 1, 1, 1));

		block2r_act3.configure(&out_block2r_con4, &out_block2r_act3, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

		block2r_con5.configure(&out_block2r_act3, &weights_block2r_con5, &bias_block2r_con5, &out_block2r_con5, PadStrideInfo(1, 1, 0, 0));

		block2_add1.configure(&out_block2r_con5, &out_block2_act0, &out_block2_add1, ConvertPolicy::SATURATE);

		block2_act1.configure(&out_block2_add1, &out_block2_act1, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

		

		block2r_con6.configure(&out_block2_act1, &weights_block2r_con6, &bias_block2r_con6, &out_block2r_con6, PadStrideInfo(1, 1, 0, 0));

		block2r_act4.configure(&out_block2r_con6, &out_block2r_act4, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

		block2r_con7.configure(&out_block2r_act4, &weights_block2r_con7, &bias_block2r_con7, &out_block2r_con7, PadStrideInfo(1, 1, 1, 1));

		block2r_act5.configure(&out_block2r_con7, &out_block2r_act5, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

		block2r_con8.configure(&out_block2r_act5, &weights_block2r_con8, &bias_block2r_con8, &out_block2r_con8, PadStrideInfo(1, 1, 0, 0));

		block2_add2.configure(&out_block2r_con8, &out_block2_act1, &out_block2_add2, ConvertPolicy::SATURATE);

		block2_act2.configure(&out_block2_add2, &out_block2_act2, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

		

		block2r_con9.configure(&out_block2_act2, &weights_block2r_con9, &bias_block2r_con9, &out_block2r_con9, PadStrideInfo(1, 1, 0, 0));

		block2r_act6.configure(&out_block2r_con9, &out_block2r_act6, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

		block2r_con10.configure(&out_block2r_act6, &weights_block2r_con10, &bias_block2r_con10, &out_block2r_con10, PadStrideInfo(1, 1, 1, 1));

		block2r_act7.configure(&out_block2r_con10, &out_block2r_act7, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

		block2r_con11.configure(&out_block2r_act7, &weights_block2r_con11, &bias_block2r_con11, &out_block2r_con11, PadStrideInfo(1, 1, 0, 0));

		block2_add3.configure(&out_block2r_con11, &out_block2_act2, &out_block2_add3, ConvertPolicy::SATURATE);

		block2_act3.configure(&out_block2_add3, &out_block2_act3, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

		





		block3r_con0.configure(&out_block2_act3, &weights_block3r_con0, &bias_block3r_con0, &out_block3r_con0, PadStrideInfo(2, 2, 0, 0));

		block3r_act0.configure(&out_block3r_con0, &out_block3r_act0, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

		block3r_con1.configure(&out_block3r_act0, &weights_block3r_con1,  &bias_block3r_con1, &out_block3r_con1, PadStrideInfo(1, 1, 1, 1));

		block3r_act1.configure(&out_block3r_con1, &out_block3r_act1, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

		block3r_con2.configure(&out_block3r_act1, &weights_block3r_con2,  &bias_block3r_con2, &out_block3r_con2, PadStrideInfo(1, 1, 0, 0));

		block3l_con0.configure(&out_block2_act3, &weights_block3l_con0,  &bias_block3l_con0, &out_block3l_con0, PadStrideInfo(2, 2, 0, 0));

		block3_add0.configure(&out_block3r_con2, &out_block3l_con0, &out_block3_add0, ConvertPolicy::SATURATE);

		block3_act0.configure(&out_block3_add0, &out_block3_act0, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));



		block3r_con3.configure(&out_block3_act0, &weights_block3r_con3,  &bias_block3r_con3, &out_block3r_con3, PadStrideInfo(1, 1, 0, 0));

		block3r_act2.configure(&out_block3r_con3, &out_block3r_act2, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

		block3r_con4.configure(&out_block3r_act2, &weights_block3r_con4,  &bias_block3r_con4, &out_block3r_con4, PadStrideInfo(1, 1, 1, 1));

		block3r_act3.configure(&out_block3r_con4, &out_block3r_act3, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

		block3r_con5.configure(&out_block3r_act3, &weights_block3r_con5,  &bias_block3r_con5, &out_block3r_con5, PadStrideInfo(1, 1, 0, 0));

		block3_add1.configure(&out_block3r_con5, &out_block3_act0, &out_block3_add1, ConvertPolicy::SATURATE);

		block3_act1.configure(&out_block3_add1, &out_block3_act1, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        

		block3r_con6.configure(&out_block3_act1, &weights_block3r_con6,  &bias_block3r_con6, &out_block3r_con6, PadStrideInfo(1, 1, 0, 0));

		block3r_act4.configure(&out_block3r_con6, &out_block3r_act4, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

		block3r_con7.configure(&out_block3r_act4, &weights_block3r_con7,  &bias_block3r_con7, &out_block3r_con7, PadStrideInfo(1, 1, 1, 1));

		block3r_act5.configure(&out_block3r_con7, &out_block3r_act5, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

		block3r_con8.configure(&out_block3r_act5, &weights_block3r_con8,  &bias_block3r_con8, &out_block3r_con8, PadStrideInfo(1, 1, 0, 0));

		block3_add2.configure(&out_block3r_con8, &out_block3_act1, &out_block3_add2, ConvertPolicy::SATURATE);

		block3_act2.configure(&out_block3_add2, &out_block3_act2, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

      

		block3r_con9.configure(&out_block3_act2, &weights_block3r_con9,  &bias_block3r_con9, &out_block3r_con9, PadStrideInfo(1, 1, 0, 0));

		block3r_act6.configure(&out_block3r_con9, &out_block3r_act6, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

		block3r_con10.configure(&out_block3r_act6, &weights_block3r_con10,  &bias_block3r_con10, &out_block3r_con10, PadStrideInfo(1, 1, 1, 1));

		block3r_act7.configure(&out_block3r_con10, &out_block3r_act7, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

		block3r_con11.configure(&out_block3r_act7, &weights_block3r_con11,  &bias_block3r_con11, &out_block3r_con11, PadStrideInfo(1, 1, 0, 0));

		block3_add3.configure(&out_block3r_con11, &out_block3_act2, &out_block3_add3, ConvertPolicy::SATURATE);

		block3_act3.configure(&out_block3_add3, &out_block3_act3, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        

		block3r_con12.configure(&out_block3_act3, &weights_block3r_con12,  &bias_block3r_con12, &out_block3r_con12, PadStrideInfo(1, 1, 0, 0));

		block3r_act8.configure(&out_block3r_con12, &out_block3r_act8, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

		block3r_con13.configure(&out_block3r_act8, &weights_block3r_con13,  &bias_block3r_con13, &out_block3r_con13, PadStrideInfo(1, 1, 1, 1));

		block3r_act9.configure(&out_block3r_con13, &out_block3r_act9, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

		block3r_con14.configure(&out_block3r_act9, &weights_block3r_con14,  &bias_block3r_con14, &out_block3r_con14, PadStrideInfo(1, 1, 0, 0));

		block3_add4.configure(&out_block3r_con14, &out_block3_act3, &out_block3_add4, ConvertPolicy::SATURATE);

		block3_act4.configure(&out_block3_add4, &out_block3_act4, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

      

		block3r_con15.configure(&out_block3_act4, &weights_block3r_con15,  &bias_block3r_con15, &out_block3r_con15, PadStrideInfo(1, 1, 0, 0));

		block3r_act10.configure(&out_block3r_con15, &out_block3r_act10, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

		block3r_con16.configure(&out_block3r_act10, &weights_block3r_con16,  &bias_block3r_con16, &out_block3r_con16, PadStrideInfo(1, 1, 1, 1));

		block3r_act11.configure(&out_block3r_con16, &out_block3r_act11, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

		block3r_con17.configure(&out_block3r_act11, &weights_block3r_con17,  &bias_block3r_con17, &out_block3r_con17, PadStrideInfo(1, 1, 0, 0));

		block3_add5.configure(&out_block3r_con17, &out_block3_act4, &out_block3_add5, ConvertPolicy::SATURATE);

		block3_act5.configure(&out_block3_add5, &out_block3_act5, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

		





		block4r_con0.configure(&out_block3_act5, &weights_block4r_con0, &bias_block4r_con0, &out_block4r_con0, PadStrideInfo(2, 2, 0, 0));

		block4r_act0.configure(&out_block4r_con0, &out_block4r_act0, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

		block4r_con1.configure(&out_block4r_act0, &weights_block4r_con1, &bias_block4r_con1, &out_block4r_con1, PadStrideInfo(1, 1, 1, 1));

		block4r_act1.configure(&out_block4r_con1, &out_block4r_act1, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

		block4r_con2.configure(&out_block4r_act1, &weights_block4r_con2, &bias_block4r_con2, &out_block4r_con2, PadStrideInfo(1, 1, 0, 0));

		block4l_con0.configure(&out_block3_act5, &weights_block4l_con0, &bias_block4l_con0, &out_block4l_con0, PadStrideInfo(2, 2, 0, 0));

		block4_add0.configure(&out_block4r_con2, &out_block4l_con0, &out_block4_add0, ConvertPolicy::SATURATE);

		block4_act0.configure(&out_block4_add0, &out_block4_act0, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

     

		block4r_con3.configure(&out_block4_act0, &weights_block4r_con3, &bias_block4r_con3, &out_block4r_con3, PadStrideInfo(1, 1, 0, 0));

		block4r_act2.configure(&out_block4r_con3, &out_block4r_act2, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

		block4r_con4.configure(&out_block4r_act2, &weights_block4r_con4, &bias_block4r_con4, &out_block4r_con4, PadStrideInfo(1, 1, 1, 1));

		block4r_act3.configure(&out_block4r_con4, &out_block4r_act3, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

		block4r_con5.configure(&out_block4r_act3, &weights_block4r_con5, &bias_block4r_con5, &out_block4r_con5, PadStrideInfo(1, 1, 0, 0));

		block4_add1.configure(&out_block4r_con5, &out_block4_act0, &out_block4_add1, ConvertPolicy::SATURATE);

		block4_act1.configure(&out_block4_add1, &out_block4_act1, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

      

		block4r_con6.configure(&out_block4_act1, &weights_block4r_con6, &bias_block4r_con6, &out_block4r_con6, PadStrideInfo(1, 1, 0, 0));

		block4r_act4.configure(&out_block4r_con6, &out_block4r_act4, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

		block4r_con7.configure(&out_block4r_act4, &weights_block4r_con7, &bias_block4r_con7, &out_block4r_con7, PadStrideInfo(1, 1, 1, 1));

		block4r_act5.configure(&out_block4r_con7, &out_block4r_act5, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

		block4r_con8.configure(&out_block4r_act5, &weights_block4r_con8, &bias_block4r_con8, &out_block4r_con8, PadStrideInfo(1, 1, 0, 0));

		block4_add2.configure(&out_block4r_con8, &out_block4_act1, &out_block4_add2,ConvertPolicy::SATURATE);

		block4_act2.configure(&out_block4_add2, &out_block4_act2, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));





	    pool1.configure(&out_block4_act2, &out_pool1, PoolingLayerInfo(PoolingType::AVG, 7, PadStrideInfo(1, 1, 0, 0, 0, 0, DimensionRoundingType::FLOOR)));

		con1.configure(&out_pool1, &weights_con1, &bias_con1, &out_con1, PadStrideInfo(1, 1, 0, 0));

		flatten.configure(&out_con1, &out_flatten);

		softmax.configure(&out_flatten, &out_softmax);



	    out_con0.allocator()->allocate(); 

	    out_act0.allocator()->allocate(); out_pool0.allocator()->allocate();





		out_block1r_con0.allocator()->allocate();  out_block1r_act0.allocator()->allocate();

		out_block1r_con1.allocator()->allocate(); out_block1r_act1.allocator()->allocate();

		out_block1r_con2.allocator()->allocate(); out_block1l_con0.allocator()->allocate(); 

		out_block1_add0.allocator()->allocate(); out_block1_act0.allocator()->allocate();



		out_block1r_con3.allocator()->allocate(); out_block1r_act2.allocator()->allocate();

		out_block1r_con4.allocator()->allocate(); out_block1r_act3.allocator()->allocate();

		out_block1r_con5.allocator()->allocate();

		out_block1_add1.allocator()->allocate(); out_block1_act1.allocator()->allocate();



		out_block1r_con6.allocator()->allocate(); out_block1r_act4.allocator()->allocate();

		out_block1r_con7.allocator()->allocate(); out_block1r_act5.allocator()->allocate();

		out_block1r_con8.allocator()->allocate();  

		out_block1_add2.allocator()->allocate(); out_block1_act2.allocator()->allocate();





		out_block2r_con0.allocator()->allocate(); out_block2r_act0.allocator()->allocate();

		out_block2r_con1.allocator()->allocate(); out_block2r_act1.allocator()->allocate();

		out_block2r_con2.allocator()->allocate(); out_block2l_con0.allocator()->allocate(); 

		out_block2_add0.allocator()->allocate(); out_block2_act0.allocator()->allocate();



		out_block2r_con3.allocator()->allocate(); out_block2r_act2.allocator()->allocate();

		out_block2r_con4.allocator()->allocate();  out_block2r_act3.allocator()->allocate();

		out_block2r_con5.allocator()->allocate(); 

		out_block2_add1.allocator()->allocate(); out_block2_act1.allocator()->allocate();



		out_block2r_con6.allocator()->allocate(); out_block2r_act4.allocator()->allocate();

		out_block2r_con7.allocator()->allocate(); out_block2r_act5.allocator()->allocate();

		out_block2r_con8.allocator()->allocate();

		out_block2_add2.allocator()->allocate(); out_block2_act2.allocator()->allocate();



		out_block2r_con9.allocator()->allocate(); out_block2r_act6.allocator()->allocate();

		out_block2r_con10.allocator()->allocate(); out_block2r_act7.allocator()->allocate();

		out_block2r_con11.allocator()->allocate(); 

		out_block2_add3.allocator()->allocate(); out_block2_act3.allocator()->allocate();





		out_block3r_con0.allocator()->allocate(); out_block3r_act0.allocator()->allocate();

		out_block3r_con1.allocator()->allocate();  out_block3r_act1.allocator()->allocate();

		out_block3r_con2.allocator()->allocate(); out_block3l_con0.allocator()->allocate(); 

		out_block3_add0.allocator()->allocate(); out_block3_act0.allocator()->allocate();



		out_block3r_con3.allocator()->allocate();  out_block3r_act2.allocator()->allocate();

		out_block3r_con4.allocator()->allocate();out_block3r_act3.allocator()->allocate();

		out_block3r_con5.allocator()->allocate(); 

		out_block3_add1.allocator()->allocate(); out_block3_act1.allocator()->allocate();



		out_block3r_con6.allocator()->allocate();out_block3r_act4.allocator()->allocate();

		out_block3r_con7.allocator()->allocate();  out_block3r_act5.allocator()->allocate();

		out_block3r_con8.allocator()->allocate(); 

		out_block3_add2.allocator()->allocate(); out_block3_act2.allocator()->allocate();



		out_block3r_con9.allocator()->allocate(); out_block3r_act6.allocator()->allocate();

		out_block3r_con10.allocator()->allocate(); out_block3r_act7.allocator()->allocate();

		out_block3r_con11.allocator()->allocate();

		out_block3_add3.allocator()->allocate(); out_block3_act3.allocator()->allocate();



		out_block3r_con12.allocator()->allocate();  out_block3r_act8.allocator()->allocate();

		out_block3r_con13.allocator()->allocate(); out_block3r_act9.allocator()->allocate();

		out_block3r_con14.allocator()->allocate(); 

		out_block3_add4.allocator()->allocate(); out_block3_act4.allocator()->allocate();



		out_block3r_con15.allocator()->allocate(); ; out_block3r_act10.allocator()->allocate();

		out_block3r_con16.allocator()->allocate(); out_block3r_act11.allocator()->allocate();

		out_block3r_con17.allocator()->allocate();  

		out_block3_add5.allocator()->allocate(); out_block3_act5.allocator()->allocate();







		out_block4r_con0.allocator()->allocate();  out_block4r_act0.allocator()->allocate();

		out_block4r_con1.allocator()->allocate(); out_block4r_act1.allocator()->allocate();

		out_block4r_con2.allocator()->allocate(); out_block4l_con0.allocator()->allocate(); 

		out_block4_add0.allocator()->allocate(); out_block4_act0.allocator()->allocate();



		out_block4r_con3.allocator()->allocate(); out_block4r_act2.allocator()->allocate();

		out_block4r_con4.allocator()->allocate();  out_block4r_act3.allocator()->allocate();

		out_block4r_con5.allocator()->allocate(); 

		out_block4_add1.allocator()->allocate(); out_block4_act1.allocator()->allocate();



		out_block4r_con6.allocator()->allocate();  out_block4r_act4.allocator()->allocate();

		out_block4r_con7.allocator()->allocate(); out_block4r_act5.allocator()->allocate();

		out_block4r_con8.allocator()->allocate(); 

		out_block4_add2.allocator()->allocate(); out_block4_act2.allocator()->allocate();





        out_pool1.allocator()->allocate(); out_con1.allocator()->allocate(); out_flatten.allocator()->allocate(); out_softmax.allocator()->allocate();





        src.allocator()->allocate(); weights_con0.allocator()->allocate();bias_con0.allocator()->allocate();

		

		weights_block1r_con0.allocator()->allocate(); weights_block1r_con1.allocator()->allocate(); weights_block1r_con2.allocator()->allocate(); 

		weights_block1r_con3.allocator()->allocate(); weights_block1r_con4.allocator()->allocate(); weights_block1r_con5.allocator()->allocate(); 

		weights_block1r_con6.allocator()->allocate(); weights_block1r_con7.allocator()->allocate(); weights_block1r_con8.allocator()->allocate(); 

		weights_block1l_con0.allocator()->allocate();

		

		weights_block2r_con0.allocator()->allocate(); weights_block2r_con1.allocator()->allocate(); weights_block2r_con2.allocator()->allocate();

		weights_block2r_con3.allocator()->allocate(); weights_block2r_con4.allocator()->allocate(); weights_block2r_con5.allocator()->allocate();

		weights_block2r_con6.allocator()->allocate(); weights_block2r_con7.allocator()->allocate(); weights_block2r_con8.allocator()->allocate();

		weights_block2r_con9.allocator()->allocate(); weights_block2r_con10.allocator()->allocate(); weights_block2r_con11.allocator()->allocate();

		weights_block2l_con0.allocator()->allocate(); 

		

		weights_block3r_con0.allocator()->allocate(); weights_block3r_con1.allocator()->allocate(); weights_block3r_con2.allocator()->allocate();

		weights_block3r_con3.allocator()->allocate(); weights_block3r_con4.allocator()->allocate(); weights_block3r_con5.allocator()->allocate();

		weights_block3r_con6.allocator()->allocate(); weights_block3r_con7.allocator()->allocate(); weights_block3r_con8.allocator()->allocate();

		weights_block3r_con9.allocator()->allocate(); weights_block3r_con10.allocator()->allocate(); weights_block3r_con11.allocator()->allocate();

		weights_block3r_con12.allocator()->allocate(); weights_block3r_con13.allocator()->allocate(); weights_block3r_con14.allocator()->allocate();

		weights_block3r_con15.allocator()->allocate(); weights_block3r_con16.allocator()->allocate(); weights_block3r_con17.allocator()->allocate();

		weights_block3l_con0.allocator()->allocate(); 

		

		weights_block4r_con0.allocator()->allocate(); weights_block4r_con1.allocator()->allocate(); weights_block4r_con2.allocator()->allocate();

		weights_block4r_con3.allocator()->allocate(); weights_block4r_con4.allocator()->allocate(); weights_block4r_con5.allocator()->allocate();

		weights_block4r_con6.allocator()->allocate(); weights_block4r_con7.allocator()->allocate(); weights_block4r_con8.allocator()->allocate();

		weights_block4l_con0.allocator()->allocate(); 

		

		bias_block1r_con0.allocator()->allocate(); bias_block1r_con1.allocator()->allocate(); bias_block1r_con2.allocator()->allocate(); 

		bias_block1r_con3.allocator()->allocate(); bias_block1r_con4.allocator()->allocate(); bias_block1r_con5.allocator()->allocate(); 

		bias_block1r_con6.allocator()->allocate(); bias_block1r_con7.allocator()->allocate(); bias_block1r_con8.allocator()->allocate(); 

		bias_block1l_con0.allocator()->allocate();

		

		bias_block2r_con0.allocator()->allocate(); bias_block2r_con1.allocator()->allocate(); bias_block2r_con2.allocator()->allocate();

		bias_block2r_con3.allocator()->allocate(); bias_block2r_con4.allocator()->allocate(); bias_block2r_con5.allocator()->allocate();

		bias_block2r_con6.allocator()->allocate(); bias_block2r_con7.allocator()->allocate(); bias_block2r_con8.allocator()->allocate();

		bias_block2r_con9.allocator()->allocate(); bias_block2r_con10.allocator()->allocate(); bias_block2r_con11.allocator()->allocate();

		bias_block2l_con0.allocator()->allocate(); 

		

		bias_block3r_con0.allocator()->allocate(); bias_block3r_con1.allocator()->allocate(); bias_block3r_con2.allocator()->allocate();

		bias_block3r_con3.allocator()->allocate(); bias_block3r_con4.allocator()->allocate(); bias_block3r_con5.allocator()->allocate();

		bias_block3r_con6.allocator()->allocate(); bias_block3r_con7.allocator()->allocate(); bias_block3r_con8.allocator()->allocate();

		bias_block3r_con9.allocator()->allocate(); bias_block3r_con10.allocator()->allocate(); bias_block3r_con11.allocator()->allocate();

		bias_block3r_con12.allocator()->allocate(); bias_block3r_con13.allocator()->allocate(); bias_block3r_con14.allocator()->allocate();

		bias_block3r_con15.allocator()->allocate(); bias_block3r_con16.allocator()->allocate(); bias_block3r_con17.allocator()->allocate();

		bias_block3l_con0.allocator()->allocate(); 

		

		bias_block4r_con0.allocator()->allocate(); bias_block4r_con1.allocator()->allocate(); bias_block4r_con2.allocator()->allocate();

		bias_block4r_con3.allocator()->allocate(); bias_block4r_con4.allocator()->allocate(); bias_block4r_con5.allocator()->allocate();

		bias_block4r_con6.allocator()->allocate(); bias_block4r_con7.allocator()->allocate(); bias_block4r_con8.allocator()->allocate();

		bias_block4l_con0.allocator()->allocate(); 



		weights_con1.allocator()->allocate(); bias_con1.allocator()->allocate();







		npy0.fill_tensor(src);

		npy1.fill_tensor(weights_con0);

		npy2.fill_tensor(weights_block1r_con0);

		npy3.fill_tensor(weights_block1r_con1);

		npy4.fill_tensor(weights_block1r_con2);

		npy5.fill_tensor(weights_block1l_con0);

		npy6.fill_tensor(weights_block1r_con3);

		npy7.fill_tensor(weights_block1r_con4);

		npy8.fill_tensor(weights_block1r_con5);

		npy9.fill_tensor(weights_block1r_con6);

		npy10.fill_tensor(weights_block1r_con7);

		npy11.fill_tensor(weights_block1r_con8);

		npy12.fill_tensor(weights_block2r_con0);

		npy13.fill_tensor(weights_block2r_con1);

		npy14.fill_tensor(weights_block2r_con2);

		npy15.fill_tensor(weights_block2l_con0);

		npy16.fill_tensor(weights_block2r_con3);

		npy17.fill_tensor(weights_block2r_con4);

		npy18.fill_tensor(weights_block2r_con5);

		npy19.fill_tensor(weights_block2r_con6);

		npy20.fill_tensor(weights_block2r_con7);

		npy21.fill_tensor(weights_block2r_con8);

		npy22.fill_tensor(weights_block2r_con9);

		npy23.fill_tensor(weights_block2r_con10);

		npy24.fill_tensor(weights_block2r_con11);

		npy25.fill_tensor(weights_block3r_con0);

		npy26.fill_tensor(weights_block3r_con1);

		npy27.fill_tensor(weights_block3r_con2);

		npy28.fill_tensor(weights_block3l_con0);

		npy29.fill_tensor(weights_block3r_con3);

		npy30.fill_tensor(weights_block3r_con4);

		npy31.fill_tensor(weights_block3r_con5);

		npy32.fill_tensor(weights_block3r_con6);

		npy33.fill_tensor(weights_block3r_con7);

		npy34.fill_tensor(weights_block3r_con8);

		npy35.fill_tensor(weights_block3r_con9);

		npy36.fill_tensor(weights_block3r_con10);

		npy37.fill_tensor(weights_block3r_con11);

		npy38.fill_tensor(weights_block3r_con12);

		npy39.fill_tensor(weights_block3r_con13);

		npy40.fill_tensor(weights_block3r_con14);

		npy41.fill_tensor(weights_block3r_con15);

		npy42.fill_tensor(weights_block3r_con16);

		npy43.fill_tensor(weights_block3r_con17);

		npy44.fill_tensor(weights_block4r_con0);

		npy45.fill_tensor(weights_block4r_con1);

		npy46.fill_tensor(weights_block4r_con2);

		npy47.fill_tensor(weights_block4l_con0);

		npy48.fill_tensor(weights_block4r_con3);

		npy49.fill_tensor(weights_block4r_con4);

		npy50.fill_tensor(weights_block4r_con5);

		npy51.fill_tensor(weights_block4r_con6);

		npy52.fill_tensor(weights_block4r_con7);

		npy53.fill_tensor(weights_block4r_con8);

		npy54.fill_tensor(weights_con1);

		npyb1.fill_tensor(bias_con0);

		npyb2.fill_tensor(bias_block1r_con0);

		npyb3.fill_tensor(bias_block1r_con1);

		npyb4.fill_tensor(bias_block1r_con2);

		npyb5.fill_tensor(bias_block1l_con0);

		npyb6.fill_tensor(bias_block1r_con3);

		npyb7.fill_tensor(bias_block1r_con4);

		npyb8.fill_tensor(bias_block1r_con5);

		npyb9.fill_tensor(bias_block1r_con6);

		npyb10.fill_tensor(bias_block1r_con7);

		npyb11.fill_tensor(bias_block1r_con8);

		npyb12.fill_tensor(bias_block2r_con0);

		npyb13.fill_tensor(bias_block2r_con1);

		npyb14.fill_tensor(bias_block2r_con2);

		npyb15.fill_tensor(bias_block2l_con0);

		npyb16.fill_tensor(bias_block2r_con3);

		npyb17.fill_tensor(bias_block2r_con4);

		npyb18.fill_tensor(bias_block2r_con5);

		npyb19.fill_tensor(bias_block2r_con6);

		npyb20.fill_tensor(bias_block2r_con7);

		npyb21.fill_tensor(bias_block2r_con8);

		npyb22.fill_tensor(bias_block2r_con9);

		npyb23.fill_tensor(bias_block2r_con10);

		npyb24.fill_tensor(bias_block2r_con11);

		npyb25.fill_tensor(bias_block3r_con0);

		npyb26.fill_tensor(bias_block3r_con1);

		npyb27.fill_tensor(bias_block3r_con2);

		npyb28.fill_tensor(bias_block3l_con0);

		npyb29.fill_tensor(bias_block3r_con3);

		npyb30.fill_tensor(bias_block3r_con4);

		npyb31.fill_tensor(bias_block3r_con5);

		npyb32.fill_tensor(bias_block3r_con6);

		npyb33.fill_tensor(bias_block3r_con7);

		npyb34.fill_tensor(bias_block3r_con8);

		npyb35.fill_tensor(bias_block3r_con9);

		npyb36.fill_tensor(bias_block3r_con10);

		npyb37.fill_tensor(bias_block3r_con11);

		npyb38.fill_tensor(bias_block3r_con12);

		npyb39.fill_tensor(bias_block3r_con13);

		npyb40.fill_tensor(bias_block3r_con14);

		npyb41.fill_tensor(bias_block3r_con15);

		npyb42.fill_tensor(bias_block3r_con16);

		npyb43.fill_tensor(bias_block3r_con17);

		npyb44.fill_tensor(bias_block4r_con0);

		npyb45.fill_tensor(bias_block4r_con1);

		npyb46.fill_tensor(bias_block4r_con2);

		npyb47.fill_tensor(bias_block4l_con0);

		npyb48.fill_tensor(bias_block4r_con3);

		npyb49.fill_tensor(bias_block4r_con4);

		npyb50.fill_tensor(bias_block4r_con5);

		npyb51.fill_tensor(bias_block4r_con6);

		npyb52.fill_tensor(bias_block4r_con7);

		npyb53.fill_tensor(bias_block4r_con8);

		npyb54.fill_tensor(bias_con1);

		is_fortran      = npy0.is_fortran();







		return true;

}

void do_run()override

{

	double conv_layer=0, act_layer=0, pool_layer=0, norm_layer=0, fc_layer=0, other_layer=0;

	double lend01=0,lend02=0,lend03=0,lend04=0;



	double lend111=0,lend112=0,lend113=0,lend114=0,lend115=0,lend116=0,lend117=0,lend118=0,lend119=0,lend1110=0,lend1111=0,lend1112=0;

	double lend121=0,lend122=0,lend123=0,lend124=0,lend125=0,lend126=0,lend127=0,lend128=0,lend129=0,lend1210=0;

	double lend131=0,lend132=0,lend133=0,lend134=0,lend135=0,lend136=0,lend137=0,lend138=0,lend139=0,lend1310=0,lend1311=0;



	double lend211=0,lend212=0,lend213=0,lend214=0,lend215=0,lend216=0,lend217=0,lend218=0,lend219=0,lend2110=0,lend2111=0,lend2112=0;

	double lend221=0,lend222=0,lend223=0,lend224=0,lend225=0,lend226=0,lend227=0,lend228=0,lend229=0,lend2210=0;

	double lend231=0,lend232=0,lend233=0,lend234=0,lend235=0,lend236=0,lend237=0,lend238=0,lend239=0,lend2310=0;

	double lend241=0,lend242=0,lend243=0,lend244=0,lend245=0,lend246=0,lend247=0,lend248=0,lend249=0,lend2410=0,lend2411=0;



	double lend311=0,lend312=0,lend313=0,lend314=0,lend315=0,lend316=0,lend317=0,lend318=0,lend319=0,lend3110=0,lend3111=0,lend3112=0;

	double lend321=0,lend322=0,lend323=0,lend324=0,lend325=0,lend326=0,lend327=0,lend328=0,lend329=0,lend3210=0;

	double lend331=0,lend332=0,lend333=0,lend334=0,lend335=0,lend336=0,lend337=0,lend338=0,lend339=0,lend3310=0;

	double lend341=0,lend342=0,lend343=0,lend344=0,lend345=0,lend346=0,lend347=0,lend348=0,lend349=0,lend3410=0;

	double lend351=0,lend352=0,lend353=0,lend354=0,lend355=0,lend356=0,lend357=0,lend358=0,lend359=0,lend3510=0;

	double lend361=0,lend362=0,lend363=0,lend364=0,lend365=0,lend366=0,lend367=0,lend368=0,lend369=0,lend3610=0,lend3611=0;



	double lend411=0,lend412=0,lend413=0,lend414=0,lend415=0,lend416=0,lend417=0,lend418=0,lend419=0,lend4110=0,lend4111=0,lend4112=0;

	double lend421=0,lend422=0,lend423=0,lend424=0,lend425=0,lend426=0,lend427=0,lend428=0,lend429=0,lend4210=0;

	double lend431=0,lend432=0,lend433=0,lend434=0,lend435=0,lend436=0,lend437=0,lend438=0,lend439=0,lend4310=0;



    double lend11=0, lend12=0, lend13=0, lend14=0; 



	double endtime=0;

	double total_time=0;

	double time=0;

	int cycles=101;



	std::string base_path = "/media/sdcard/ComputeLibrary";

	std::string output_file_path = "/model.csv";

	ofstream out(base_path+output_file_path, ios::out | ios::app);

	out<<"ResNet50 GEMM"<<std::endl;

	for (int i = 0; i < cycles; i++)

	{

		auto start = std::chrono::high_resolution_clock::now();

		con0.run(); auto end01=std::chrono::high_resolution_clock::now();

		auto end02=std::chrono::high_resolution_clock::now();

		act0.run();auto end03=std::chrono::high_resolution_clock::now(); 

		pool0.run();auto end04=std::chrono::high_resolution_clock::now();

	

	

		block1r_con0.run(); auto end111=std::chrono::high_resolution_clock::now();

		auto end112=std::chrono::high_resolution_clock::now();

		block1r_act0.run();auto end113=std::chrono::high_resolution_clock::now();

		block1r_con1.run(); auto end114=std::chrono::high_resolution_clock::now();

		auto end115=std::chrono::high_resolution_clock::now();

		block1r_act1.run();auto end116=std::chrono::high_resolution_clock::now();

		block1r_con2.run(); auto end117=std::chrono::high_resolution_clock::now();

		auto end118=std::chrono::high_resolution_clock::now();

		block1l_con0.run(); auto end119=std::chrono::high_resolution_clock::now();

		auto end1110=std::chrono::high_resolution_clock::now();

		block1_add0.run(); auto end1111=std::chrono::high_resolution_clock::now();

		block1_act0.run();auto end1112=std::chrono::high_resolution_clock::now();



		block1r_con3.run();auto end121=std::chrono::high_resolution_clock::now();

		auto end122=std::chrono::high_resolution_clock::now();

		block1r_act2.run();auto end123=std::chrono::high_resolution_clock::now();

		block1r_con4.run(); auto end124=std::chrono::high_resolution_clock::now();

		auto end125=std::chrono::high_resolution_clock::now();

		block1r_act3.run();auto end126=std::chrono::high_resolution_clock::now();

		block1r_con5.run(); auto end127=std::chrono::high_resolution_clock::now();

		auto end128=std::chrono::high_resolution_clock::now();

		block1_add1.run(); auto end129=std::chrono::high_resolution_clock::now();

		block1_act1.run();auto end1210=std::chrono::high_resolution_clock::now();



		block1r_con6.run(); auto end131=std::chrono::high_resolution_clock::now();

		auto end132=std::chrono::high_resolution_clock::now();

		block1r_act4.run(); auto end133=std::chrono::high_resolution_clock::now();

		block1r_con7.run();  auto end134=std::chrono::high_resolution_clock::now();

		auto end135=std::chrono::high_resolution_clock::now();

		block1r_act5.run(); auto end136=std::chrono::high_resolution_clock::now();

		block1r_con8.run();  auto end137=std::chrono::high_resolution_clock::now();

		auto end138=std::chrono::high_resolution_clock::now();

		auto end139=std::chrono::high_resolution_clock::now();

		block1_add2.run();  auto end1310=std::chrono::high_resolution_clock::now();

		block1_act2.run(); auto end1311=std::chrono::high_resolution_clock::now();





		block2r_con0.run(); auto end211=std::chrono::high_resolution_clock::now();

		auto end212=std::chrono::high_resolution_clock::now();

		block2r_act0.run();auto end213=std::chrono::high_resolution_clock::now();

		block2r_con1.run(); auto end214=std::chrono::high_resolution_clock::now();

		auto end215=std::chrono::high_resolution_clock::now();

		block2r_act1.run();auto end216=std::chrono::high_resolution_clock::now();

		block2r_con2.run(); auto end217=std::chrono::high_resolution_clock::now();

		auto end218=std::chrono::high_resolution_clock::now();

		block2l_con0.run(); auto end219=std::chrono::high_resolution_clock::now();

		auto end2110=std::chrono::high_resolution_clock::now();

		block2_add0.run(); auto end2111=std::chrono::high_resolution_clock::now();

		block2_act0.run();auto end2112=std::chrono::high_resolution_clock::now();



		block2r_con3.run(); auto end221=std::chrono::high_resolution_clock::now();

		auto end222=std::chrono::high_resolution_clock::now();

		block2r_act2.run();auto end223=std::chrono::high_resolution_clock::now();

		block2r_con4.run();auto end224=std::chrono::high_resolution_clock::now();

		auto end225=std::chrono::high_resolution_clock::now();

		block2r_act3.run();auto end226=std::chrono::high_resolution_clock::now();

		block2r_con5.run(); auto end227=std::chrono::high_resolution_clock::now();

		auto end228=std::chrono::high_resolution_clock::now();

		block2_add1.run(); auto end229=std::chrono::high_resolution_clock::now();

		block2_act1.run();auto end2210=std::chrono::high_resolution_clock::now();



		block2r_con6.run(); auto end231=std::chrono::high_resolution_clock::now();

		auto end232=std::chrono::high_resolution_clock::now();

		block2r_act4.run();auto end233=std::chrono::high_resolution_clock::now();

		block2r_con7.run();auto end234=std::chrono::high_resolution_clock::now();

		auto end235=std::chrono::high_resolution_clock::now();

		block2r_act5.run();auto end236=std::chrono::high_resolution_clock::now();

		block2r_con8.run(); auto end237=std::chrono::high_resolution_clock::now();

		auto end238=std::chrono::high_resolution_clock::now();

		block2_add2.run();auto end239=std::chrono::high_resolution_clock::now();

		block2_act2.run();auto end2310=std::chrono::high_resolution_clock::now();



		block2r_con9.run(); auto end241=std::chrono::high_resolution_clock::now();

		auto end242=std::chrono::high_resolution_clock::now();

		block2r_act6.run();auto end243=std::chrono::high_resolution_clock::now();

		block2r_con10.run();auto end244=std::chrono::high_resolution_clock::now();

		auto end245=std::chrono::high_resolution_clock::now();

		block2r_act7.run();auto end246=std::chrono::high_resolution_clock::now();

		block2r_con11.run(); auto end247=std::chrono::high_resolution_clock::now();

		auto end248=std::chrono::high_resolution_clock::now();

		auto end249=std::chrono::high_resolution_clock::now();

		block2_add3.run(); auto end2410=std::chrono::high_resolution_clock::now();

		block2_act3.run();auto end2411=std::chrono::high_resolution_clock::now();





		block3r_con0.run();auto end311=std::chrono::high_resolution_clock::now();

		auto end312=std::chrono::high_resolution_clock::now();

		block3r_act0.run();auto end313=std::chrono::high_resolution_clock::now();

		block3r_con1.run(); auto end314=std::chrono::high_resolution_clock::now();

		auto end315=std::chrono::high_resolution_clock::now();

		block3r_act1.run();auto end316=std::chrono::high_resolution_clock::now();

		block3r_con2.run(); auto end317=std::chrono::high_resolution_clock::now();

		auto end318=std::chrono::high_resolution_clock::now();

		block3l_con0.run(); auto end319=std::chrono::high_resolution_clock::now();

		auto end3110=std::chrono::high_resolution_clock::now();

		block3_add0.run(); auto end3111=std::chrono::high_resolution_clock::now();

		block3_act0.run();auto end3112=std::chrono::high_resolution_clock::now();



		block3r_con3.run(); auto end321=std::chrono::high_resolution_clock::now();

		auto end322=std::chrono::high_resolution_clock::now();

		block3r_act2.run();auto end323=std::chrono::high_resolution_clock::now();

		block3r_con4.run(); auto end324=std::chrono::high_resolution_clock::now();

		auto end325=std::chrono::high_resolution_clock::now();

		block3r_act3.run();auto end326=std::chrono::high_resolution_clock::now();

		block3r_con5.run(); auto end327=std::chrono::high_resolution_clock::now();

		auto end328=std::chrono::high_resolution_clock::now();

		block3_add1.run(); auto end329=std::chrono::high_resolution_clock::now();

		block3_act1.run();auto end3210=std::chrono::high_resolution_clock::now();



		block3r_con6.run(); auto end331=std::chrono::high_resolution_clock::now();

		auto end332=std::chrono::high_resolution_clock::now();

		block3r_act4.run();auto end333=std::chrono::high_resolution_clock::now();

		block3r_con7.run(); auto end334=std::chrono::high_resolution_clock::now();

		auto end335=std::chrono::high_resolution_clock::now();

		block3r_act5.run();auto end336=std::chrono::high_resolution_clock::now();

		block3r_con8.run(); auto end337=std::chrono::high_resolution_clock::now();

		auto end338=std::chrono::high_resolution_clock::now();

		block3_add2.run();auto end339=std::chrono::high_resolution_clock::now();

		block3_act2.run();auto end3310=std::chrono::high_resolution_clock::now();



		block3r_con9.run(); auto end341=std::chrono::high_resolution_clock::now();

		auto end342=std::chrono::high_resolution_clock::now();

		block3r_act6.run();auto end343=std::chrono::high_resolution_clock::now();

		block3r_con10.run(); auto end344=std::chrono::high_resolution_clock::now();

		auto end345=std::chrono::high_resolution_clock::now();

		block3r_act7.run();auto end346=std::chrono::high_resolution_clock::now();

		block3r_con11.run(); auto end347=std::chrono::high_resolution_clock::now();

		auto end348=std::chrono::high_resolution_clock::now();

		block3_add3.run(); auto end349=std::chrono::high_resolution_clock::now();

		block3_act3.run();auto end3410=std::chrono::high_resolution_clock::now();



		block3r_con12.run(); auto end351=std::chrono::high_resolution_clock::now();

		auto end352=std::chrono::high_resolution_clock::now();

		block3r_act8.run();auto end353=std::chrono::high_resolution_clock::now();

		block3r_con13.run(); auto end354=std::chrono::high_resolution_clock::now();

		auto end355=std::chrono::high_resolution_clock::now();

		block3r_act9.run();auto end356=std::chrono::high_resolution_clock::now();

		block3r_con14.run(); auto end357=std::chrono::high_resolution_clock::now();

		auto end358=std::chrono::high_resolution_clock::now();

		block3_add4.run();auto end359=std::chrono::high_resolution_clock::now();

		block3_act4.run();auto end3510=std::chrono::high_resolution_clock::now();



		block3r_con15.run();auto end361=std::chrono::high_resolution_clock::now();

		auto end362=std::chrono::high_resolution_clock::now();

		block3r_act10.run();auto end363=std::chrono::high_resolution_clock::now();

		block3r_con16.run(); auto end364=std::chrono::high_resolution_clock::now();

		auto end365=std::chrono::high_resolution_clock::now();

		block3r_act11.run();auto end366=std::chrono::high_resolution_clock::now();

		block3r_con17.run(); auto end367=std::chrono::high_resolution_clock::now();

		auto end368=std::chrono::high_resolution_clock::now();

		auto end369=std::chrono::high_resolution_clock::now();

		block3_add5.run();auto end3610=std::chrono::high_resolution_clock::now();

		block3_act5.run();auto end3611=std::chrono::high_resolution_clock::now();







		block4r_con0.run(); auto end411=std::chrono::high_resolution_clock::now();

		auto end412=std::chrono::high_resolution_clock::now();

		block4r_act0.run(); auto end413=std::chrono::high_resolution_clock::now();

		block4r_con1.run();  auto end414=std::chrono::high_resolution_clock::now();

		auto end415=std::chrono::high_resolution_clock::now();

		block4r_act1.run(); auto end416=std::chrono::high_resolution_clock::now();

		block4r_con2.run();  auto end417=std::chrono::high_resolution_clock::now();

		auto end418=std::chrono::high_resolution_clock::now();

		block4l_con0.run();  auto end419=std::chrono::high_resolution_clock::now();

		auto end4110=std::chrono::high_resolution_clock::now();

		block4_add0.run();  auto end4111=std::chrono::high_resolution_clock::now();

		block4_act0.run(); auto end4112=std::chrono::high_resolution_clock::now();



		block4r_con3.run(); auto end421=std::chrono::high_resolution_clock::now();

		auto end422=std::chrono::high_resolution_clock::now();

		block4r_act2.run();auto end423=std::chrono::high_resolution_clock::now();

		block4r_con4.run(); auto end424=std::chrono::high_resolution_clock::now();

		auto end425=std::chrono::high_resolution_clock::now();

		block4r_act3.run();auto end426=std::chrono::high_resolution_clock::now();

		block4r_con5.run(); auto end427=std::chrono::high_resolution_clock::now();

		auto end428=std::chrono::high_resolution_clock::now();

		block4_add1.run(); auto end429=std::chrono::high_resolution_clock::now();

		block4_act1.run();auto end4210=std::chrono::high_resolution_clock::now();



		block4r_con6.run(); auto end431=std::chrono::high_resolution_clock::now();

		auto end432=std::chrono::high_resolution_clock::now();

		block4r_act4.run();auto end433=std::chrono::high_resolution_clock::now();

		block4r_con7.run(); auto end434=std::chrono::high_resolution_clock::now();

		auto end435=std::chrono::high_resolution_clock::now();

		block4r_act5.run();auto end436=std::chrono::high_resolution_clock::now();

		block4r_con8.run(); auto end437=std::chrono::high_resolution_clock::now();

		auto end438=std::chrono::high_resolution_clock::now();

		block4_add2.run(); auto end439=std::chrono::high_resolution_clock::now();

		block4_act2.run();auto end4310=std::chrono::high_resolution_clock::now();





		pool1.run();auto end11=std::chrono::high_resolution_clock::now();

		con1.run(); auto end12=std::chrono::high_resolution_clock::now();

		flatten.run();auto end13=std::chrono::high_resolution_clock::now();

		softmax.run();auto end14=std::chrono::high_resolution_clock::now();

		auto end = std::chrono::high_resolution_clock::now();



		if(i>0){

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end01 - start).count();lend01+=time;conv_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end02 - end01).count();lend02+=time;other_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end03 - end01).count();lend03+=time;act_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end04 - end03).count();lend04+=time;pool_layer+=time;





			time = std::chrono::duration_cast<std::chrono::duration<double>>(end111 - end04).count();lend111+=time;conv_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end112 - end111).count();lend112+=time;other_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end113 - end111).count();lend113+=time;act_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end114 - end113).count();lend114+=time;conv_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end115 - end114).count();lend115+=time;other_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end116 - end114).count();lend116+=time;act_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end117 - end116).count();lend117+=time;conv_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end118 - end117).count();lend118+=time;other_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end119 - end117).count();lend119+=time;conv_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end1110 - end119).count();lend1110+=time;other_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end1111 - end119).count();lend1111+=time;other_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end1112 - end1111).count();lend1112+=time;act_layer+=time;



			time = std::chrono::duration_cast<std::chrono::duration<double>>(end121 - end1112).count();lend121+=time;conv_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end122 - end121).count();lend122+=time;other_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end123 - end121).count();lend123+=time;act_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end124 - end123).count();lend124+=time;conv_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end125 - end124).count();lend125+=time;other_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end126 - end124).count();lend126+=time;act_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end127 - end126).count();lend127+=time;conv_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end128 - end127).count();lend128+=time;other_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end129 - end127).count();lend129+=time;other_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end1210 - end129).count();lend1210+=time;act_layer+=time;



			time = std::chrono::duration_cast<std::chrono::duration<double>>(end131 - end1210).count();lend131+=time;conv_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end132 - end131).count();lend132+=time;other_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end133 - end131).count();lend133+=time;act_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end134 - end133).count();lend134+=time;conv_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end135 - end134).count();lend135+=time;other_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end136 - end134).count();lend136+=time;act_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end137 - end136).count();lend137+=time;conv_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end138 - end137).count();lend138+=time;other_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end139 - end137).count();lend139+=time;other_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end1310 - end139).count();lend1310+=time;other_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end1311 - end1310).count();lend1311+=time;act_layer+=time;





			time = std::chrono::duration_cast<std::chrono::duration<double>>(end211 - end1311).count();lend211+=time;conv_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end212 - end211).count();lend212+=time;other_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end213 - end211).count();lend213+=time;act_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end214 - end213).count();lend214+=time;conv_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end215 - end214).count();lend215+=time;other_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end216 - end214).count();lend216+=time;act_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end217 - end216).count();lend217+=time;conv_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end218 - end217).count();lend218+=time;other_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end219 - end217).count();lend219+=time;conv_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end2110 - end219).count();lend2110+=time;other_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end2111 - end219).count();lend2111+=time;other_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end2112 - end2111).count();lend2112+=time;act_layer+=time;



			time = std::chrono::duration_cast<std::chrono::duration<double>>(end221 - end2112).count();lend221+=time;conv_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end222 - end221).count();lend222+=time;other_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end223 - end221).count();lend223+=time;act_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end224 - end223).count();lend224+=time;conv_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end225 - end224).count();lend225+=time;other_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end226 - end224).count();lend226+=time;act_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end227 - end226).count();lend227+=time;conv_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end228 - end227).count();lend228+=time;other_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end229 - end227).count();lend229+=time;other_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end2210 - end229).count();lend2210+=time;act_layer+=time;



			time = std::chrono::duration_cast<std::chrono::duration<double>>(end231 - end2210).count();lend231+=time;conv_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end232 - end231).count();lend232+=time;other_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end233 - end231).count();lend233+=time;act_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end234 - end233).count();lend234+=time;conv_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end235 - end234).count();lend235+=time;other_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end236 - end234).count();lend236+=time;act_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end237 - end236).count();lend237+=time;conv_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end238 - end237).count();lend238+=time;other_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end239 - end237).count();lend239+=time;other_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end2310 - end239).count();lend2310+=time;act_layer+=time;



			time = std::chrono::duration_cast<std::chrono::duration<double>>(end241 - end2310).count();lend241+=time;conv_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end242 - end241).count();lend242+=time;other_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end243 - end241).count();lend243+=time;act_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end244 - end243).count();lend244+=time;conv_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end245 - end244).count();lend245+=time;other_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end246 - end244).count();lend246+=time;act_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end247 - end246).count();lend247+=time;conv_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end248 - end247).count();lend248+=time;other_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end249 - end247).count();lend249+=time;other_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end2410 - end249).count();lend2410+=time;other_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end2411 - end2410).count();lend2411+=time;act_layer+=time;





			time = std::chrono::duration_cast<std::chrono::duration<double>>(end311 - end2410).count();lend311+=time;conv_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end312 - end311).count();lend312+=time;other_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end313 - end311).count();lend313+=time;act_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end314 - end313).count();lend314+=time;conv_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end315 - end314).count();lend315+=time;other_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end316 - end314).count();lend316+=time;act_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end317 - end316).count();lend317+=time;conv_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end318 - end317).count();lend318+=time;other_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end319 - end317).count();lend319+=time;conv_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end3110 - end319).count();lend3110+=time;other_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end3111 - end319).count();lend3111+=time;other_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end3112 - end3111).count();lend3112+=time;act_layer+=time;



			time = std::chrono::duration_cast<std::chrono::duration<double>>(end321 - end3112).count();lend321+=time;conv_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end322 - end321).count();lend322+=time;other_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end323 - end321).count();lend323+=time;act_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end324 - end323).count();lend324+=time;conv_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end325 - end324).count();lend325+=time;other_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end326 - end324).count();lend326+=time;act_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end327 - end326).count();lend327+=time;conv_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end328 - end327).count();lend328+=time;other_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end329 - end327).count();lend329+=time;other_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end3210 - end329).count();lend3210+=time;act_layer+=time;



			time = std::chrono::duration_cast<std::chrono::duration<double>>(end331 - end3210).count();lend331+=time;conv_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end332 - end331).count();lend332+=time;other_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end333 - end331).count();lend333+=time;act_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end334 - end333).count();lend334+=time;conv_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end335 - end334).count();lend335+=time;other_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end336 - end334).count();lend336+=time;act_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end337 - end336).count();lend337+=time;conv_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end338 - end337).count();lend338+=time;other_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end339 - end337).count();lend339+=time;other_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end3310 - end339).count();lend3310+=time;act_layer+=time;



			time = std::chrono::duration_cast<std::chrono::duration<double>>(end341 - end3310).count();lend341+=time;conv_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end342 - end341).count();lend342+=time;other_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end343 - end341).count();lend343+=time;act_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end344 - end343).count();lend344+=time;conv_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end345 - end344).count();lend345+=time;other_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end346 - end344).count();lend346+=time;act_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end347 - end346).count();lend347+=time;conv_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end348 - end347).count();lend348+=time;other_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end349 - end347).count();lend349+=time;other_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end3410 - end349).count();lend3410+=time;act_layer+=time;



			time = std::chrono::duration_cast<std::chrono::duration<double>>(end351 - end3410).count();lend351+=time;conv_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end352 - end351).count();lend352+=time;other_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end353 - end351).count();lend353+=time;act_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end354 - end353).count();lend354+=time;conv_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end355 - end354).count();lend355+=time;other_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end356 - end354).count();lend356+=time;act_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end357 - end356).count();lend357+=time;conv_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end358 - end357).count();lend358+=time;other_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end359 - end357).count();lend359+=time;other_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end3510 - end359).count();lend3510+=time;act_layer+=time;



			time = std::chrono::duration_cast<std::chrono::duration<double>>(end361 - end3510).count();lend361+=time;conv_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end362 - end361).count();lend362+=time;other_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end363 - end361).count();lend363+=time;act_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end364 - end363).count();lend364+=time;conv_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end365 - end364).count();lend365+=time;other_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end366 - end364).count();lend366+=time;act_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end367 - end366).count();lend367+=time;conv_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end368 - end367).count();lend368+=time;other_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end369 - end367).count();lend369+=time;other_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end3610 - end369).count();lend3610+=time;other_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end3611 - end3610).count();lend3611+=time;act_layer+=time;





			time = std::chrono::duration_cast<std::chrono::duration<double>>(end411 - end3611).count();lend411+=time;conv_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end412 - end411).count();lend412+=time;other_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end413 - end411).count();lend413+=time;act_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end414 - end413).count();lend414+=time;conv_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end415 - end414).count();lend415+=time;other_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end416 - end414).count();lend416+=time;act_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end417 - end416).count();lend417+=time;conv_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end418 - end417).count();lend418+=time;other_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end419 - end417).count();lend419+=time;conv_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end4110 - end419).count();lend4110+=time;other_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end4111 - end419).count();lend4111+=time;other_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end4112 - end4111).count();lend4112+=time;act_layer+=time;



			time = std::chrono::duration_cast<std::chrono::duration<double>>(end421 - end4112).count();lend421+=time;conv_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end422 - end421).count();lend422+=time;other_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end423 - end421).count();lend423+=time;act_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end424 - end423).count();lend424+=time;conv_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end425 - end424).count();lend425+=time;other_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end426 - end424).count();lend426+=time;act_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end427 - end426).count();lend427+=time;conv_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end428 - end427).count();lend428+=time;other_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end429 - end427).count();lend429+=time;other_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end4210 - end429).count();lend4210+=time;act_layer+=time;



			time = std::chrono::duration_cast<std::chrono::duration<double>>(end431 - end4210).count();lend431+=time;conv_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end432 - end431).count();lend432+=time;other_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end433 - end431).count();lend433+=time;act_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end434 - end433).count();lend434+=time;conv_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end435 - end434).count();lend435+=time;other_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end436 - end434).count();lend436+=time;act_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end437 - end436).count();lend437+=time;conv_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end438 - end437).count();lend438+=time;other_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end439 - end437).count();lend439+=time;other_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end4310 - end439).count();lend4310+=time;act_layer+=time;





			time = std::chrono::duration_cast<std::chrono::duration<double>>(end11 - end4310).count();lend11+=time;pool_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end12 - end11).count();lend12+=time;fc_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end13 - end12).count();lend13+=time;other_layer+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end14 - end13).count();lend14+=time;other_layer+=time;

			endtime = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();

			if(i>0){

			std::cout<<i<<"---run:"<<std::endl;

			std::cout<<"time="<<endtime*1000<<"ms"<<std::endl;

			out<<"one run time"<<","<<endtime*1000<<std::endl;

			total_time+=endtime;}

			if(i==0){

			std::cout<<"First run:"<<std::endl;

			std::cout<<"time="<<endtime*1000<<"ms"<<std::endl;

			}

		}

	}

		arm_compute::utils::save_to_npy(out_pool1,output_filename,false);

		arm_compute::utils::save_to_npy(out_con1,output_filename1,false);

		arm_compute::utils::save_to_npy(out_block4_add2,output_filename2,false);

		arm_compute::utils::save_to_npy(out_pool1,output_filename3,false);



		out<<"Resnet50"<<std::endl;

		out << "---conv1       " <<","<< lend01 * 1000/(cycles-1) << std::endl;

		out << "---relu1       " <<","<< lend03* 1000/(cycles-1) << std::endl;

		out << "---pooling1    " <<","<< lend04 * 1000/(cycles-1) << std::endl;



		out<<"---layer1      "<<std::endl;

		out<<"  ---0         "<<std::endl;

		out << "   ---conv1    " << ","<< lend111 * 1000/(cycles-1) << std::endl;

		out << "   ---relu1    " << ","<< lend113 * 1000/(cycles-1) << std::endl;

		out << "   ---conv2    " << ","<< lend114 * 1000/(cycles-1) << std::endl;

		out << "   ---relu2    " << ","<< lend116 * 1000/(cycles-1) << std::endl;

		out << "   ---conv3    " << ","<< lend117 * 1000/(cycles-1) << std::endl;

		out << "   ---conv    " << ","<< lend119 * 1000/(cycles-1) << std::endl;

		out << "   ---add      " << ","<< lend1111 * 1000/(cycles-1) << std::endl;

		out << "   ---relu     " << ","<< lend1112 * 1000/(cycles-1) << std::endl;

		

		out<<"  ---1         "<<std::endl;

		out << "   ---conv1    " << ","<< lend121 * 1000/(cycles-1) << std::endl;

		out << "   ---relu1    " << ","<< lend123 * 1000/(cycles-1) << std::endl;

		out << "   ---conv2    " << ","<< lend124 * 1000/(cycles-1) << std::endl;

		out << "   ---relu2    " << ","<< lend126 * 1000/(cycles-1) << std::endl;

		out << "   ---conv3    " << ","<< lend127 * 1000/(cycles-1) << std::endl;

		out << "   ---add      " << ","<< lend129 * 1000/(cycles-1) << std::endl;

		out << "   ---relu     " << ","<< lend1210 * 1000/(cycles-1) << std::endl;



		out<<"  ---2         "<<std::endl;

		out << "   ---conv1    " << ","<< lend131 * 1000/(cycles-1) << std::endl;

		out << "   ---relu1    " << ","<< lend133 * 1000/(cycles-1) << std::endl;

		out << "   ---conv2    " << ","<< lend134 * 1000/(cycles-1) << std::endl;

		out << "   ---relu2    " << ","<< lend136 * 1000/(cycles-1) << std::endl;

		out << "   ---conv3    " << ","<< lend137 * 1000/(cycles-1) << std::endl;

		out << "   ---add      " << ","<< lend1310 * 1000/(cycles-1) << std::endl;

		out << "   ---relu     " << ","<< lend1311 * 1000/(cycles-1) << std::endl;



		out<<"---layer2      "<<std::endl;

		out<<"  ---0         "<<std::endl;

		out << "   ---conv1    " << ","<< lend211 * 1000/(cycles-1) << std::endl;

		out << "   ---relu1    " << ","<< lend213 * 1000/(cycles-1) << std::endl;

		out << "   ---conv2    " << ","<< lend214 * 1000/(cycles-1) << std::endl;

		out << "   ---relu2    " << ","<< lend216 * 1000/(cycles-1) << std::endl;

		out << "   ---conv3    " << ","<< lend217 * 1000/(cycles-1) << std::endl;

		out << "   ---conv    " << ","<< lend219 * 1000/(cycles-1) << std::endl;

		out << "   ---add      " << ","<< lend2111 * 1000/(cycles-1) << std::endl;

		out << "   ---relu     " << ","<< lend2112 * 1000/(cycles-1) << std::endl;

		

		out<<"  ---1         "<<std::endl;

		out << "   ---conv1    " << ","<< lend221 * 1000/(cycles-1) << std::endl;

		out << "   ---relu1    " << ","<< lend223 * 1000/(cycles-1) << std::endl;

		out << "   ---conv2    " << ","<< lend224 * 1000/(cycles-1) << std::endl;

		out << "   ---relu2    " << ","<< lend226 * 1000/(cycles-1) << std::endl;

		out << "   ---conv3    " << ","<< lend227 * 1000/(cycles-1) << std::endl;

		out << "   ---add      " << ","<< lend229 * 1000/(cycles-1) << std::endl;

		out << "   ---relu     " << ","<< lend2210 * 1000/(cycles-1) << std::endl;



		out<<"  ---2         "<<std::endl;

		out << "   ---conv1    " << ","<< lend231* 1000/(cycles-1) << std::endl;

		out << "   ---relu1    " << ","<< lend233 * 1000/(cycles-1) << std::endl;

		out << "   ---conv2    " << ","<< lend234 * 1000/(cycles-1) << std::endl;

		out << "   ---relu2    " << ","<< lend236 * 1000/(cycles-1) << std::endl;

		out << "   ---conv3    " << ","<< lend237 * 1000/(cycles-1) << std::endl;

		out << "   ---add      " << ","<< lend239 * 1000/(cycles-1) << std::endl;

		out << "   ---relu     " << ","<< lend2310 * 1000/(cycles-1) << std::endl;



		out<<"  ---3         "<<std::endl;

		out << "   ---conv1    " << ","<< lend241 * 1000/(cycles-1) << std::endl;

		out << "   ---relu1    " << ","<< lend243 * 1000/(cycles-1) << std::endl;

		out << "   ---conv2    " << ","<< lend244 * 1000/(cycles-1) << std::endl;

		out << "   ---relu2    " << ","<< lend246 * 1000/(cycles-1) << std::endl;

		out << "   ---conv3    " << ","<< lend247 * 1000/(cycles-1) << std::endl;

		out << "   ---add      " << ","<< lend2410 * 1000/(cycles-1) << std::endl;

		out << "   ---relu     " << ","<< lend2411 * 1000/(cycles-1) << std::endl;



		out<<"---layer3      "<<std::endl;

		out<<"  ---0         "<<std::endl;

		out << "   ---conv1    " << ","<< lend311 * 1000/(cycles-1) << std::endl;

		out << "   ---relu1    " << ","<< lend313 * 1000/(cycles-1) << std::endl;

		out << "   ---conv2    " << ","<< lend314 * 1000/(cycles-1) << std::endl;

		out << "   ---relu2    " << ","<< lend316 * 1000/(cycles-1) << std::endl;

		out << "   ---conv3    " << ","<< lend317 * 1000/(cycles-1) << std::endl;

		out << "   ---conv    " << ","<< lend319 * 1000/(cycles-1) << std::endl;

		out << "   ---add      " << ","<< lend3111 * 1000/(cycles-1) << std::endl;

		out << "   ---relu     " << ","<< lend3112 * 1000/(cycles-1) << std::endl;

		

		out<<"  ---1         "<<std::endl;

		out << "   ---conv1    " << ","<< lend321 * 1000/(cycles-1) << std::endl;

		out << "   ---relu1    " << ","<< lend323 * 1000/(cycles-1) << std::endl;

		out << "   ---conv2    " << ","<< lend324 * 1000/(cycles-1) << std::endl;

		out << "   ---relu2    " << ","<< lend326 * 1000/(cycles-1) << std::endl;

		out << "   ---conv3    " << ","<< lend327 * 1000/(cycles-1) << std::endl;

		out << "   ---add      " << ","<< lend329 * 1000/(cycles-1) << std::endl;

		out << "   ---relu     " << ","<< lend3210 * 1000/(cycles-1) << std::endl;



		out<<"  ---2         "<<std::endl;

		out << "   ---conv1    " << ","<< lend331 * 1000/(cycles-1) << std::endl;

		out << "   ---relu1    " << ","<< lend333 * 1000/(cycles-1) << std::endl;

		out << "   ---conv2    " << ","<< lend334 * 1000/(cycles-1) << std::endl;

		out << "   ---relu2    " << ","<< lend336 * 1000/(cycles-1) << std::endl;

		out << "   ---conv3    " << ","<< lend337 * 1000/(cycles-1) << std::endl;

		out << "   ---add      " << ","<< lend339 * 1000/(cycles-1) << std::endl;

		out << "   ---relu     " << ","<< lend3310 * 1000/(cycles-1) << std::endl;



		out<<"  ---3         "<<std::endl;

		out << "   ---conv1    " << ","<< lend341 * 1000/(cycles-1) << std::endl;

		out << "   ---relu1    " << ","<< lend343 * 1000/(cycles-1) << std::endl;

		out << "   ---conv2    " << ","<< lend344 * 1000/(cycles-1) << std::endl;

		out << "   ---relu2    " << ","<< lend346 * 1000/(cycles-1) << std::endl;

		out << "   ---conv3    " << ","<< lend347 * 1000/(cycles-1) << std::endl;

		out << "   ---add      " << ","<< lend349 * 1000/(cycles-1) << std::endl;

		out << "   ---relu     " << ","<< lend3410 * 1000/(cycles-1) << std::endl;



		out<<"  ---4         "<<std::endl;

		out << "   ---conv1    " << ","<< lend351 * 1000/(cycles-1) << std::endl;

		out << "   ---relu1    " << ","<< lend353 * 1000/(cycles-1) << std::endl;

		out << "   ---conv2    " << ","<< lend354 * 1000/(cycles-1) << std::endl;

		out << "   ---relu2    " << ","<< lend356 * 1000/(cycles-1) << std::endl;

		out << "   ---conv3    " << ","<< lend357 * 1000/(cycles-1) << std::endl;

		out << "   ---add      " << ","<< lend359 * 1000/(cycles-1) << std::endl;

		out << "   ---relu     " << ","<< lend3510 * 1000/(cycles-1) << std::endl;



		out<<"  ---5         "<<std::endl;

		out << "   ---conv1    " << ","<< lend361 * 1000/(cycles-1) << std::endl;

		out << "   ---relu1    " << ","<< lend363 * 1000/(cycles-1) << std::endl;

		out << "   ---conv2    " << ","<< lend364 * 1000/(cycles-1) << std::endl;

		out << "   ---relu2    " << ","<< lend366 * 1000/(cycles-1) << std::endl;

		out << "   ---conv3    " << ","<< lend367 * 1000/(cycles-1) << std::endl;

		out << "   ---add      " << ","<< lend3610 * 1000/(cycles-1) << std::endl;

		out << "   ---relu     " << ","<< lend3611 * 1000/(cycles-1) << std::endl;



		out<<"---layer4      "<<std::endl;

		out<<"  ---0         "<<std::endl;

		out << "   ---conv1    " << ","<< lend411 * 1000/(cycles-1) << std::endl;

		out << "   ---relu1    " << ","<< lend413 * 1000/(cycles-1) << std::endl;

		out << "   ---conv2    " << ","<< lend414 * 1000/(cycles-1) << std::endl;

		out << "   ---relu2    " << ","<< lend416 * 1000/(cycles-1) << std::endl;

		out << "   ---conv3    " << ","<< lend417 * 1000/(cycles-1) << std::endl;

		out << "   ---conv    " << ","<< lend419 * 1000/(cycles-1) << std::endl;

		out << "   ---add      " << ","<< lend4111 * 1000/(cycles-1) << std::endl;

		out << "   ---relu     " << ","<< lend4112 * 1000/(cycles-1) << std::endl;

		

		out<<"  ---1         "<<std::endl;

		out << "   ---conv1    " << ","<< lend421 * 1000/(cycles-1) << std::endl;

		out << "   ---relu1    " << ","<< lend423 * 1000/(cycles-1) << std::endl;

		out << "   ---conv2    " << ","<< lend424 * 1000/(cycles-1) << std::endl;

		out << "   ---relu2    " << ","<< lend426 * 1000/(cycles-1) << std::endl;

		out << "   ---conv3    " << ","<< lend427 * 1000/(cycles-1) << std::endl;

		out << "   ---add      " << ","<< lend429 * 1000/(cycles-1) << std::endl;

		out << "   ---relu     " << ","<< lend4210 * 1000/(cycles-1) << std::endl;



		out<<"  ---2         "<<std::endl;

		out << "   ---conv1    " << ","<< lend431 * 1000/(cycles-1) << std::endl;

		out << "   ---relu1    " << ","<< lend433 * 1000/(cycles-1) << std::endl;

		out << "   ---conv2    " << ","<< lend434 * 1000/(cycles-1) << std::endl;

		out << "   ---relu2    " << ","<< lend436 * 1000/(cycles-1) << std::endl;

		out << "   ---conv3    " << ","<< lend437 * 1000/(cycles-1) << std::endl;

		out << "   ---add      " << ","<< lend439 * 1000/(cycles-1) << std::endl;

		out << "   ---relu     " << ","<< lend4310 * 1000/(cycles-1) << std::endl;	



		out << "---pooling     " << ","<< lend11 * 1000/(cycles-1) << std::endl;

		out << "---conv1       " << ","<<lend12 * 1000/(cycles-1) << std::endl;

		out << "---flatten     "<<","<< lend13 * 1000/(cycles-1) << std::endl;

		out << "---softmax   " <<","<< lend14 * 1000/(cycles-1) << std::endl;



		if(cycles>1)

		{

			out<<"avg time="<<","<<total_time*1000/(cycles-1)<<std::endl;

			out<<"conv layers: "<<","<<conv_layer*1000/(cycles-1)<<std::endl;

			out<<"act  layers: "<<","<<act_layer*1000/(cycles-1) <<std::endl;

			out<<"pool layers: "<<","<<pool_layer*1000/(cycles-1)<<std::endl;

			out<<"norm layers: "<<","<<norm_layer*1000/(cycles-1)<<std::endl;

			out<<"fc   layers: "<<","<<fc_layer*1000/(cycles-1)  <<std::endl;

			out<<"other layers: "<<","<<other_layer*1000/(cycles-1)<<std::endl;

		}









			std::cout<<"Resnet50"<<std::endl;

			std::cout << "---conv1       " << "		"<< lend01 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "---relu1       " <<"		"<< lend03* 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "---pooling1    " <<"		"<< lend04 * 1000/(cycles-1) << "ms" << std::endl;



			std::cout<<"---layer1      "<<std::endl;

			std::cout<<"  ---0         "<<std::endl;

			std::cout << "   ---conv1    " << "		"<< lend111 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---relu1    " << "		"<< lend113 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---conv2    " << "		"<< lend114 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---relu2    " << "		"<< lend116 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---conv3    " << "		"<< lend117 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---conv    " << "		"<< lend119 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---add      " << "		"<< lend1111 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---relu     " << "		"<< lend1112 * 1000/(cycles-1) << "ms" << std::endl;

			

			std::cout<<"  ---1         "<<std::endl;

			std::cout << "   ---conv1    " << "		"<< lend121 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---relu1    " << "		"<< lend123 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---conv2    " << "		"<< lend124 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---relu2    " << "		"<< lend126 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---conv3    " << "		"<< lend127 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---add      " << "		"<< lend129 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---relu     " << "		"<< lend1210 * 1000/(cycles-1) << "ms" << std::endl;



			std::cout<<"  ---2         "<<std::endl;

			std::cout << "   ---conv1    " << "		"<< lend131 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---relu1    " << "		"<< lend133 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---conv2    " << "		"<< lend134 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---relu2    " << "		"<< lend136 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---conv3    " << "		"<< lend137 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---add      " << "		"<< lend1310 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---relu     " << "		"<< lend1311 * 1000/(cycles-1) << "ms" << std::endl;



			std::cout<<"---layer2      "<<std::endl;

			std::cout<<"  ---0         "<<std::endl;

			std::cout << "   ---conv1    " << "		"<< lend211 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---relu1    " << "		"<< lend213 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---conv2    " << "		"<< lend214 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---relu2    " << "		"<< lend216 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---conv3    " << "		"<< lend217 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---conv    " << "		"<< lend219 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---add      " << "		"<< lend2111 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---relu     " << "		"<< lend2112 * 1000/(cycles-1) << "ms" << std::endl;

			

			std::cout<<"  ---1         "<<std::endl;

			std::cout << "   ---conv1    " << "		"<< lend221 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---relu1    " << "		"<< lend223 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---conv2    " << "		"<< lend224 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---relu2    " << "		"<< lend226 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---conv3    " << "		"<< lend227 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---add      " << "		"<< lend229 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---relu     " << "		"<< lend2210 * 1000/(cycles-1) << "ms" << std::endl;



			std::cout<<"  ---2         "<<std::endl;

			std::cout << "   ---conv1    " << "		"<< lend231* 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---relu1    " << "		"<< lend233 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---conv2    " << "		"<< lend234 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---relu2    " << "		"<< lend236 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---conv3    " << "		"<< lend237 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---add      " << "		"<< lend239 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---relu     " << "		"<< lend2310 * 1000/(cycles-1) << "ms" << std::endl;



			std::cout<<"  ---3         "<<std::endl;

			std::cout << "   ---conv1    " << "		"<< lend241 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---relu1    " << "		"<< lend243 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---conv2    " << "		"<< lend244 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---relu2    " << "		"<< lend246 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---conv3    " << "		"<< lend247 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---add      " << "		"<< lend2410 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---relu     " << "		"<< lend2411 * 1000/(cycles-1) << "ms" << std::endl;



			std::cout<<"---layer3      "<<std::endl;

			std::cout<<"  ---0         "<<std::endl;

			std::cout << "   ---conv1    " << "		"<< lend311 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---relu1    " << "		"<< lend313 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---conv2    " << "		"<< lend314 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---relu2    " << "		"<< lend316 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---conv3    " << "		"<< lend317 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---conv    " << "		"<< lend319 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---add      " << "		"<< lend3111 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---relu     " << "		"<< lend3112 * 1000/(cycles-1) << "ms" << std::endl;

			

			std::cout<<"  ---1         "<<std::endl;

			std::cout << "   ---conv1    " << "		"<< lend321 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---relu1    " << "		"<< lend323 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---conv2    " << "		"<< lend324 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---relu2    " << "		"<< lend326 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---conv3    " << "		"<< lend327 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---add      " << "		"<< lend329 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---relu     " << "		"<< lend3210 * 1000/(cycles-1) << "ms" << std::endl;



			std::cout<<"  ---2         "<<std::endl;

			std::cout << "   ---conv1    " << "		"<< lend331 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---relu1    " << "		"<< lend333 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---conv2    " << "		"<< lend334 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---relu2    " << "		"<< lend336 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---conv3    " << "		"<< lend337 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---add      " << "		"<< lend339 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---relu     " << "		"<< lend3310 * 1000/(cycles-1) << "ms" << std::endl;



			std::cout<<"  ---3         "<<std::endl;

			std::cout << "   ---conv1    " << "		"<< lend341 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---relu1    " << "		"<< lend343 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---conv2    " << "		"<< lend344 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---relu2    " << "		"<< lend346 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---conv3    " << "		"<< lend347 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---add      " << "		"<< lend349 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---relu     " << "		"<< lend3410 * 1000/(cycles-1) << "ms" << std::endl;



			std::cout<<"  ---4         "<<std::endl;

			std::cout << "   ---conv1    " << "		"<< lend351 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---relu1    " << "		"<< lend353 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---conv2    " << "		"<< lend354 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---relu2    " << "		"<< lend356 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---conv3    " << "		"<< lend357 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---add      " << "		"<< lend359 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---relu     " << "		"<< lend3510 * 1000/(cycles-1) << "ms" << std::endl;



			std::cout<<"  ---5         "<<std::endl;

			std::cout << "   ---conv1    " << "		"<< lend361 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---relu1    " << "		"<< lend363 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---conv2    " << "		"<< lend364 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---relu2    " << "		"<< lend366 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---conv3    " << "		"<< lend367 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---add      " << "		"<< lend3610 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---relu     " << "		"<< lend3611 * 1000/(cycles-1) << "ms" << std::endl;



			std::cout<<"---layer4      "<<std::endl;

			std::cout<<"  ---0         "<<std::endl;

			std::cout << "   ---conv1    " << "		"<< lend411 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---relu1    " << "		"<< lend413 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---conv2    " << "		"<< lend414 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---relu2    " << "		"<< lend416 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---conv3    " << "		"<< lend417 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---conv    " << "		"<< lend419 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---add      " << "		"<< lend4111 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---relu     " << "		"<< lend4112 * 1000/(cycles-1) << "ms" << std::endl;

			

			std::cout<<"  ---1         "<<std::endl;

			std::cout << "   ---conv1    " << "		"<< lend421 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---relu1    " << "		"<< lend423 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---conv2    " << "		"<< lend424 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---relu2    " << "		"<< lend426 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---conv3    " << "		"<< lend427 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---add      " << "		"<< lend429 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---relu     " << "		"<< lend4210 * 1000/(cycles-1) << "ms" << std::endl;



			std::cout<<"  ---2         "<<std::endl;

			std::cout << "   ---conv1    " << "		"<< lend431 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---relu1    " << "		"<< lend433 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---conv2    " << "		"<< lend434 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---relu2    " << "		"<< lend436 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---conv3    " << "		"<< lend437 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---add      " << "		"<< lend439 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---relu     " << "		"<< lend4310 * 1000/(cycles-1) << "ms" << std::endl;	



			std::cout << "---pooling     " << "		"<< lend11 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "---conv1       " << "		"<<lend12 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "---flatten     "<<"		"<< lend13 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "---softmax   " <<"		"<< lend14 * 1000/(cycles-1) << "ms" << std::endl;



			if(cycles>1)

			{

				std::cout<<"avg time="<<total_time*1000/(cycles-1)<<"ms"<<std::endl;

				std::cout<<"conv layers: "<<conv_layer*1000/(cycles-1)<<"ms"<<std::endl;

				std::cout<<"act  layers: "<<act_layer*1000/(cycles-1) <<"ms"<<std::endl;

				std::cout<<"pool layers: "<<pool_layer*1000/(cycles-1)<<"ms"<<std::endl;

				std::cout<<"norm layers: "<<norm_layer*1000/(cycles-1)<<"ms"<<std::endl;

				std::cout<<"fc   layers: "<<fc_layer*1000/(cycles-1)  <<"ms"<<std::endl;

				std::cout<<"other layers: "<<other_layer*1000/(cycles-1)<<"ms"<<std::endl;

			}



}

private:

	bool is_fortran{};

	string output_filename="/media/sdcard/ComputeLibrary/data/resnet50_resnet50_float_output/float_output.npy";

	string output_filename1="/media/sdcard/ComputeLibrary/data/resnet50_resnet50_float_output/float_output1.npy";

	string output_filename2="/media/sdcard/ComputeLibrary/data/resnet50_resnet50_float_output/float_output2.npy";

	string output_filename3="/media/sdcard/ComputeLibrary/data/resnet50_resnet50_float_output/float_output3.npy";



	Tensor src{}; Tensor weights_con0{};Tensor bias_con0{};



	Tensor weights_block1r_con0{}; Tensor weights_block1r_con1{}; Tensor weights_block1r_con2{};

	Tensor weights_block1r_con3{}; Tensor weights_block1r_con4{}; Tensor weights_block1r_con5{};

	Tensor weights_block1r_con6{}; Tensor weights_block1r_con7{}; Tensor weights_block1r_con8{};

	Tensor weights_block1l_con0{}; 



	Tensor weights_block2r_con0{}; Tensor weights_block2r_con1{}; Tensor weights_block2r_con2{};

	Tensor weights_block2r_con3{}; Tensor weights_block2r_con4{}; Tensor weights_block2r_con5{};

	Tensor weights_block2r_con6{}; Tensor weights_block2r_con7{}; Tensor weights_block2r_con8{};

	Tensor weights_block2r_con9{}; Tensor weights_block2r_con10{}; Tensor weights_block2r_con11{};

	Tensor weights_block2l_con0{}; 



	Tensor weights_block3r_con0{}; Tensor weights_block3r_con1{}; Tensor weights_block3r_con2{};

	Tensor weights_block3r_con3{}; Tensor weights_block3r_con4{}; Tensor weights_block3r_con5{};

	Tensor weights_block3r_con6{}; Tensor weights_block3r_con7{}; Tensor weights_block3r_con8{};

	Tensor weights_block3r_con9{}; Tensor weights_block3r_con10{}; Tensor weights_block3r_con11{};

	Tensor weights_block3r_con12{}; Tensor weights_block3r_con13{}; Tensor weights_block3r_con14{};

	Tensor weights_block3r_con15{}; Tensor weights_block3r_con16{}; Tensor weights_block3r_con17{};

	Tensor weights_block3l_con0{}; 



	Tensor weights_block4r_con0{}; Tensor weights_block4r_con1{}; Tensor weights_block4r_con2{};

	Tensor weights_block4r_con3{}; Tensor weights_block4r_con4{}; Tensor weights_block4r_con5{};

	Tensor weights_block4r_con6{}; Tensor weights_block4r_con7{}; Tensor weights_block4r_con8{};

	Tensor weights_block4l_con0{}; 



	Tensor bias_block1r_con0{}; Tensor bias_block1r_con1{}; Tensor bias_block1r_con2{};

	Tensor bias_block1r_con3{}; Tensor bias_block1r_con4{}; Tensor bias_block1r_con5{};

	Tensor bias_block1r_con6{}; Tensor bias_block1r_con7{}; Tensor bias_block1r_con8{};

	Tensor bias_block1l_con0{}; 



	Tensor bias_block2r_con0{}; Tensor bias_block2r_con1{}; Tensor bias_block2r_con2{};

	Tensor bias_block2r_con3{}; Tensor bias_block2r_con4{}; Tensor bias_block2r_con5{};

	Tensor bias_block2r_con6{}; Tensor bias_block2r_con7{}; Tensor bias_block2r_con8{};

	Tensor bias_block2r_con9{}; Tensor bias_block2r_con10{}; Tensor bias_block2r_con11{};

	Tensor bias_block2l_con0{}; 



	Tensor bias_block3r_con0{}; Tensor bias_block3r_con1{}; Tensor bias_block3r_con2{};

	Tensor bias_block3r_con3{}; Tensor bias_block3r_con4{}; Tensor bias_block3r_con5{};

	Tensor bias_block3r_con6{}; Tensor bias_block3r_con7{}; Tensor bias_block3r_con8{};

	Tensor bias_block3r_con9{}; Tensor bias_block3r_con10{}; Tensor bias_block3r_con11{};

	Tensor bias_block3r_con12{}; Tensor bias_block3r_con13{}; Tensor bias_block3r_con14{};

	Tensor bias_block3r_con15{}; Tensor bias_block3r_con16{}; Tensor bias_block3r_con17{};

	Tensor bias_block3l_con0{}; 



	Tensor bias_block4r_con0{}; Tensor bias_block4r_con1{}; Tensor bias_block4r_con2{};

	Tensor bias_block4r_con3{}; Tensor bias_block4r_con4{}; Tensor bias_block4r_con5{};

	Tensor bias_block4r_con6{}; Tensor bias_block4r_con7{}; Tensor bias_block4r_con8{};

	Tensor bias_block4l_con0{}; 



	Tensor weights_con1{}, bias_con1{};





	Tensor out_con0{}; 

	Tensor out_act0{}; Tensor out_pool0{};



	Tensor out_block1r_con0{}; Tensor out_block1r_act0{};

	Tensor out_block1r_con1{};  Tensor out_block1r_act1{};

	Tensor out_block1r_con2{}; Tensor out_block1l_con0{}; 

	Tensor out_block1_add0{}; Tensor out_block1_act0{};



	Tensor out_block1r_con3{}; Tensor out_block1r_act2{};

	Tensor out_block1r_con4{};  Tensor out_block1r_act3{};

	Tensor out_block1r_con5{};

	Tensor out_block1_add1{}; Tensor out_block1_act1{};



	Tensor out_block1r_con6{}; Tensor out_block1r_act4{};

	Tensor out_block1r_con7{};  Tensor out_block1r_act5{};

	Tensor out_block1r_con8{}; 

	Tensor out_block1_add2{}; Tensor out_block1_act2{};



	Tensor out_block2r_con0{}; Tensor out_block2r_act0{};

	Tensor out_block2r_con1{};  Tensor out_block2r_act1{};

	Tensor out_block2r_con2{}; Tensor out_block2l_con0{}; 

	Tensor out_block2_add0{}; Tensor out_block2_act0{};



	Tensor out_block2r_con3{}; Tensor out_block2r_act2{};

	Tensor out_block2r_con4{};  Tensor out_block2r_act3{};

	Tensor out_block2r_con5{};

	Tensor out_block2_add1{}; Tensor out_block2_act1{};



	Tensor out_block2r_con6{}; Tensor out_block2r_act4{};

	Tensor out_block2r_con7{}; Tensor out_block2r_act5{};

	Tensor out_block2r_con8{}; 

	Tensor out_block2_add2{}; Tensor out_block2_act2{};



	Tensor out_block2r_con9{};  Tensor out_block2r_act6{};

	Tensor out_block2r_con10{};Tensor out_block2r_act7{};

	Tensor out_block2r_con11{}; 

	Tensor out_block2_add3{}; Tensor out_block2_act3{};



	Tensor out_block3r_con0{};  Tensor out_block3r_act0{};

	Tensor out_block3r_con1{};Tensor out_block3r_act1{};

	Tensor out_block3r_con2{}; Tensor out_block3l_con0{}; 

	Tensor out_block3_add0{}; Tensor out_block3_act0{};



	Tensor out_block3r_con3{};  Tensor out_block3r_act2{};

	Tensor out_block3r_con4{}; Tensor out_block3r_act3{};

	Tensor out_block3r_con5{}; 

	Tensor out_block3_add1{}; Tensor out_block3_act1{};



	Tensor out_block3r_con6{};  Tensor out_block3r_act4{};

	Tensor out_block3r_con7{};  Tensor out_block3r_act5{};

	Tensor out_block3r_con8{}; 

	Tensor out_block3_add2{}; Tensor out_block3_act2{};



	Tensor out_block3r_con9{}; Tensor out_block3r_act6{};

	Tensor out_block3r_con10{}; Tensor out_block3r_act7{};

	Tensor out_block3r_con11{}; 

	Tensor out_block3_add3{}; Tensor out_block3_act3{};



	Tensor out_block3r_con12{}; Tensor out_block3r_act8{};

	Tensor out_block3r_con13{};  Tensor out_block3r_act9{};

	Tensor out_block3r_con14{};

	Tensor out_block3_add4{}; Tensor out_block3_act4{};



	Tensor out_block3r_con15{}; Tensor out_block3r_act10{};

	Tensor out_block3r_con16{};  Tensor out_block3r_act11{};

	Tensor out_block3r_con17{};  

	Tensor out_block3_add5{}; Tensor out_block3_act5{};





	Tensor out_block4r_con0{};  Tensor out_block4r_act0{};

	Tensor out_block4r_con1{};  Tensor out_block4r_act1{};

	Tensor out_block4r_con2{};  Tensor out_block4l_con0{};

	Tensor out_block4_add0{}; Tensor out_block4_act0{};



	Tensor out_block4r_con3{};  Tensor out_block4r_act2{};

	Tensor out_block4r_con4{};  Tensor out_block4r_act3{};

	Tensor out_block4r_con5{}; 

	Tensor out_block4_add1{}; Tensor out_block4_act1{};



	Tensor out_block4r_con6{};  Tensor out_block4r_act4{};

	Tensor out_block4r_con7{};  Tensor out_block4r_act5{};

	Tensor out_block4r_con8{}; 

	Tensor out_block4_add2{}; Tensor out_block4_act2{};



	Tensor out_pool1{}; Tensor out_con1{}; Tensor out_flatten{}; Tensor out_softmax{};





	NEGEMMConvolutionLayer con0{}; NEActivationLayer act0{}; NEPoolingLayer pool0{};



	NEGEMMConvolutionLayer  block1r_con0{};   NEActivationLayer  block1r_act0{};

	NEGEMMConvolutionLayer  block1r_con1{};  NEActivationLayer  block1r_act1{};

	NEGEMMConvolutionLayer  block1r_con2{};   NEGEMMConvolutionLayer block1l_con0{}; 

	NEArithmeticAddition  block1_add0{}; NEActivationLayer  block1_act0{};



	NEGEMMConvolutionLayer  block1r_con3{}; NEActivationLayer  block1r_act2{};

	NEGEMMConvolutionLayer  block1r_con4{};   NEActivationLayer  block1r_act3{};

	NEGEMMConvolutionLayer  block1r_con5{}; 

	NEArithmeticAddition  block1_add1{}; NEActivationLayer  block1_act1{};



	NEGEMMConvolutionLayer  block1r_con6{}; NEActivationLayer  block1r_act4{};

	NEGEMMConvolutionLayer  block1r_con7{};   NEActivationLayer  block1r_act5{};

	NEGEMMConvolutionLayer  block1r_con8{};  

	NEArithmeticAddition  block1_add2{}; NEActivationLayer  block1_act2{};



	NEGEMMConvolutionLayer  block2r_con0{};  NEActivationLayer  block2r_act0{};

	NEGEMMConvolutionLayer  block2r_con1{};  NEActivationLayer  block2r_act1{};

	NEGEMMConvolutionLayer  block2r_con2{};  NEGEMMConvolutionLayer block2l_con0{}; 

	NEArithmeticAddition  block2_add0{}; NEActivationLayer  block2_act0{};



	NEGEMMConvolutionLayer  block2r_con3{};   NEActivationLayer  block2r_act2{};

	NEGEMMConvolutionLayer  block2r_con4{};  NEActivationLayer  block2r_act3{};

	NEGEMMConvolutionLayer  block2r_con5{}; 

	NEArithmeticAddition  block2_add1{}; NEActivationLayer  block2_act1{};



	NEGEMMConvolutionLayer  block2r_con6{}; NEActivationLayer  block2r_act4{};

	NEGEMMConvolutionLayer  block2r_con7{};  NEActivationLayer  block2r_act5{};

	NEGEMMConvolutionLayer  block2r_con8{}; 

	NEArithmeticAddition  block2_add2{}; NEActivationLayer  block2_act2{};



	NEGEMMConvolutionLayer  block2r_con9{};   NEActivationLayer  block2r_act6{};

	NEGEMMConvolutionLayer  block2r_con10{}; NEActivationLayer  block2r_act7{};

	NEGEMMConvolutionLayer  block2r_con11{};  

	NEArithmeticAddition  block2_add3{}; NEActivationLayer  block2_act3{};



	NEGEMMConvolutionLayer  block3r_con0{};  NEActivationLayer  block3r_act0{};

	NEGEMMConvolutionLayer  block3r_con1{};  NEActivationLayer  block3r_act1{};

	NEGEMMConvolutionLayer  block3r_con2{};  NEGEMMConvolutionLayer block3l_con0{}; 

	NEArithmeticAddition  block3_add0{}; NEActivationLayer  block3_act0{};



	NEGEMMConvolutionLayer  block3r_con3{}; NEActivationLayer  block3r_act2{};

	NEGEMMConvolutionLayer  block3r_con4{};   NEActivationLayer  block3r_act3{};

	NEGEMMConvolutionLayer  block3r_con5{}; 

	NEArithmeticAddition  block3_add1{}; NEActivationLayer  block3_act1{};



	NEGEMMConvolutionLayer  block3r_con6{};NEActivationLayer  block3r_act4{};

	NEGEMMConvolutionLayer  block3r_con7{};  NEActivationLayer  block3r_act5{};

	NEGEMMConvolutionLayer  block3r_con8{};  

	NEArithmeticAddition  block3_add2{}; NEActivationLayer  block3_act2{};



	NEGEMMConvolutionLayer  block3r_con9{};  NEActivationLayer  block3r_act6{};

	NEGEMMConvolutionLayer  block3r_con10{}; NEActivationLayer  block3r_act7{};

	NEGEMMConvolutionLayer  block3r_con11{}; 

	NEArithmeticAddition  block3_add3{}; NEActivationLayer  block3_act3{};



	NEGEMMConvolutionLayer  block3r_con12{};  NEActivationLayer  block3r_act8{};

	NEGEMMConvolutionLayer  block3r_con13{}; NEActivationLayer  block3r_act9{};

	NEGEMMConvolutionLayer  block3r_con14{}; 

	NEArithmeticAddition  block3_add4{}; NEActivationLayer  block3_act4{};



	NEGEMMConvolutionLayer  block3r_con15{};  NEActivationLayer  block3r_act10{};

	NEGEMMConvolutionLayer  block3r_con16{};   NEActivationLayer  block3r_act11{};

	NEGEMMConvolutionLayer  block3r_con17{}; 

	NEArithmeticAddition  block3_add5{}; NEActivationLayer  block3_act5{};



	NEGEMMConvolutionLayer  block4r_con0{};  NEActivationLayer  block4r_act0{};

	NEGEMMConvolutionLayer  block4r_con1{}; NEActivationLayer  block4r_act1{};

	NEGEMMConvolutionLayer  block4r_con2{}; NEGEMMConvolutionLayer block4l_con0{}; 

	NEArithmeticAddition  block4_add0{}; NEActivationLayer  block4_act0{};



	NEGEMMConvolutionLayer  block4r_con3{}; NEActivationLayer  block4r_act2{};

	NEGEMMConvolutionLayer  block4r_con4{}; NEActivationLayer  block4r_act3{};

	NEGEMMConvolutionLayer  block4r_con5{};

	NEArithmeticAddition  block4_add1{}; NEActivationLayer  block4_act1{};



	NEGEMMConvolutionLayer  block4r_con6{};  NEActivationLayer  block4r_act4{};

	NEGEMMConvolutionLayer  block4r_con7{};NEActivationLayer  block4r_act5{};

	NEGEMMConvolutionLayer  block4r_con8{}; 

	NEArithmeticAddition  block4_add2{}; NEActivationLayer  block4_act2{};







	NEPoolingLayer pool1{}; NEGEMMConvolutionLayer con1{}; NEFlattenLayer flatten{}; NESoftmaxLayer softmax{};

	

	/*ConvertPolicy A{};*/



};

int main(int argc, char **argv)

{

	/*

	cpu_set_t cpuset;

	CPU_ZERO(&cpuset);

    

	CPU_SET(0, &cpuset);

    CPU_SET(1, &cpuset);

    CPU_SET(2, &cpuset);

    CPU_SET(3, &cpuset);

	

	



    CPU_SET(4, &cpuset);

	

    CPU_SET(5, &cpuset);

    CPU_SET(6, &cpuset);

    CPU_SET(7, &cpuset);

	

	int e = sched_setaffinity(getpid(), sizeof(cpuset), &cpuset);

	if(e !=0) {

		std::cout << "Error in setting sched_setaffinity \n";

	}

	CPPScheduler::get().set_num_threads(1);

	*/

	return utils::run_example<NEONRESNETExample>(argc, argv);

}


