/*#define _GNU_SOURCE*/

#include <stdio.h>

#include <stdlib.h>

#include <sched.h>

#include <unistd.h>

#include <sys/times.h>



#include "arm_compute/core/Types.h"

#include "arm_compute/runtime/CL/CLFunctions.h"

#include "arm_compute/runtime/NEON/NEFunctions.h"

#include "arm_compute/runtime/NEON/NEScheduler.h"

#include "arm_compute/runtime/Allocator.h"

#include "utils/ImageLoader.h"

#include "utils/Utils.h"

#include <ctime>

#include <cstdlib>





using namespace arm_compute;

using namespace utils;



class NEONRESNETExample : public Example

{

public:



	void add_kernel_time(std::tuple<double, double, double, double, double> t)

	{

		im2col_kernel_time+=std::get<0>(t);

		interleave_kernel_time+=std::get<1>(t);

		matrix_multiply_kernel_time+=std::get<2>(t);

		mmlast_kernel_time+=std::get<3>(t);

		col2im_kernel_time+=std::get<4>(t);

	}



	bool do_setup(int argc, char **argv) override

	{

		string data_path="/media/sdcard/ComputeLibrary/data/neon_resnet50/";

		NPYLoader npy_input;npy_input.open(data_path+"input.npy");npy_input.init_tensor2(src,DataType:: S8);

		/*first conv-batch-act-pooling*/

		NPYLoader npy0_q;npy0_q.open(Q_table_datapath+Q_table_name[0]);npy0_q.init_tensor2(Q_table_con0,DataType:: S16);

		NPYLoader npy0_wt;npy0_wt.open(WT_buffer_datapath+WT_buffer_name[0]);npy0_wt.init_tensor2(WT_buffer_con0,DataType::U16);

		NPYLoader npy0_b;npy0_b.open(bias_datapath+bias_name[0]);npy0_b.init_tensor2(bias_con0,DataType:: S8);

		WeightsInfo weights_con0(false,7,7,64,false);

		const TensorShape out_shape_con0(112, 112, 64);

		out_con0.allocator()->init(TensorInfo(out_shape_con0, 1, DataType:: S8));

		out_act0.allocator()->init(TensorInfo(out_shape_con0,1,DataType:: F32));

		TensorShape out_shape_pool0 = out_shape_con0;

		out_shape_pool0.set(0, out_shape_pool0.x() / 2);

		out_shape_pool0.set(1, out_shape_pool0.y() / 2);

		out_pool0.allocator()->init(TensorInfo(out_shape_pool0, 1, DataType:: F32));

	

		/* block start          */

		/* block1 */

		/* conv-batch-act*/

		NPYLoader npy1_q;npy1_q.open(Q_table_datapath+Q_table_name[1]);npy1_q.init_tensor2(Q_table_block1r_con0,DataType:: S16);

		NPYLoader npy1_wt;npy1_wt.open(WT_buffer_datapath+WT_buffer_name[1]);npy1_wt.init_tensor2(WT_buffer_block1r_con0,DataType::U16);

		NPYLoader npy1_b;npy1_b.open(bias_datapath+bias_name[1]);npy1_b.init_tensor2(bias_block1r_con0,DataType:: S8);

		WeightsInfo weights_block1r_con0(false,1,1,64,false);

		const TensorShape out_shape_block1r_con0(56, 56, 64);

		out_block1r_con0.allocator()->init(TensorInfo(out_shape_block1r_con0, 1, DataType:: S8));

		out_block1r_act0.allocator()->init(TensorInfo(out_shape_block1r_con0, 1, DataType:: F32));

   		/*conv-batch-act	  */

		NPYLoader npy2_q;npy2_q.open(Q_table_datapath+Q_table_name[2]);npy2_q.init_tensor2(Q_table_block1r_con1,DataType:: S16);

		NPYLoader npy2_wt;npy2_wt.open(WT_buffer_datapath+WT_buffer_name[2]);npy2_wt.init_tensor2(WT_buffer_block1r_con1,DataType::U16);

		NPYLoader npy2_b;npy2_b.open(bias_datapath+bias_name[2]);npy2_b.init_tensor2(bias_block1r_con1,DataType:: S8);

		WeightsInfo weights_block1r_con1(false,3,3,64,false);

		const TensorShape out_shape_block1r_con1(56, 56, 64);

		out_block1r_con1.allocator()->init(TensorInfo(out_shape_block1r_con1, 1, DataType:: S8));

		out_block1r_act1.allocator()->init(TensorInfo(out_shape_block1r_con1, 1, DataType:: F32));

  		 /*conv-batch     */

		NPYLoader npy3_q;npy3_q.open(Q_table_datapath+Q_table_name[3]);npy3_q.init_tensor2(Q_table_block1r_con2,DataType:: S16);

		NPYLoader npy3_wt;npy3_wt.open(WT_buffer_datapath+WT_buffer_name[3]);npy3_wt.init_tensor2(WT_buffer_block1r_con2,DataType::U16);

		NPYLoader npy3_b;npy3_b.open(bias_datapath+bias_name[3]);npy3_b.init_tensor2(bias_block1r_con2,DataType:: S8);

		WeightsInfo weights_block1r_con2(false,1,1,256,false);

		const TensorShape out_shape_block1r_con2(56, 56,256);

		out_block1r_con2.allocator()->init(TensorInfo(out_shape_block1r_con2, 1, DataType:: S8));

   		/*conv-batch*/

		NPYLoader npy4_q;npy4_q.open(Q_table_datapath+Q_table_name[4]);npy4_q.init_tensor2(Q_table_block1l_con0,DataType:: S16);

		NPYLoader npy4_wt;npy4_wt.open(WT_buffer_datapath+WT_buffer_name[4]);npy4_wt.init_tensor2(WT_buffer_block1l_con0,DataType::U16);

		NPYLoader npy4_b;npy4_b.open(bias_datapath+bias_name[4]);npy4_b.init_tensor2(bias_block1l_con0,DataType:: S8);

		WeightsInfo weights_block1l_con0(false,1,1,256,false);

		const TensorShape out_shape_block1l_con0(56, 56, 256);

		out_block1l_con0.allocator()->init(TensorInfo(out_shape_block1l_con0, 1, DataType:: S8));

		/*    //add-act*/

		TensorShape out_shape_block1_0 = out_shape_block1r_con2;

		out_block1_add0.allocator()->init(TensorInfo(out_shape_block1_0, 1, DataType:: S8));

		out_block1_act0.allocator()->init(TensorInfo(out_shape_block1_0, 1, DataType:: F32));

   		/*conv-batch-act*/

		NPYLoader npy5_q;npy5_q.open(Q_table_datapath+Q_table_name[5]);npy5_q.init_tensor2(Q_table_block1r_con3,DataType:: S16);

		NPYLoader npy5_wt;npy5_wt.open(WT_buffer_datapath+WT_buffer_name[5]);npy5_wt.init_tensor2(WT_buffer_block1r_con3,DataType::U16);

		NPYLoader npy5_b;npy5_b.open(bias_datapath+bias_name[5]);npy5_b.init_tensor2(bias_block1r_con3,DataType:: S8);

		WeightsInfo weights_block1r_con3(false,1,1,64,false);

		const TensorShape out_shape_block1r_con3(56, 56,64);

		out_block1r_con3.allocator()->init(TensorInfo(out_shape_block1r_con3, 1, DataType:: S8));

		out_block1r_act2.allocator()->init(TensorInfo(out_shape_block1r_con3, 1, DataType:: F32));

   		/*conv-batch-act*/

		NPYLoader npy6_q;npy6_q.open(Q_table_datapath+Q_table_name[6]);npy6_q.init_tensor2(Q_table_block1r_con4,DataType:: S16);

		NPYLoader npy6_wt;npy6_wt.open(WT_buffer_datapath+WT_buffer_name[6]);npy6_wt.init_tensor2(WT_buffer_block1r_con4,DataType::U16);

		NPYLoader npy6_b;npy6_b.open(bias_datapath+bias_name[6]);npy6_b.init_tensor2(bias_block1r_con4,DataType:: S8);

		WeightsInfo weights_block1r_con4(false,3,3,64,false);

		const TensorShape out_shape_block1r_con4(56, 56, 64);

		out_block1r_con4.allocator()->init(TensorInfo(out_shape_block1r_con4, 1, DataType:: S8));

		out_block1r_act3.allocator()->init(TensorInfo(out_shape_block1r_con4, 1, DataType:: F32));

   		/*conv-batch*/

		NPYLoader npy7_q;npy7_q.open(Q_table_datapath+Q_table_name[7]);npy7_q.init_tensor2(Q_table_block1r_con5,DataType:: S16);

		NPYLoader npy7_wt;npy7_wt.open(WT_buffer_datapath+WT_buffer_name[7]);npy7_wt.init_tensor2(WT_buffer_block1r_con5,DataType::U16);

		NPYLoader npy7_b;npy7_b.open(bias_datapath+bias_name[7]);npy7_b.init_tensor2(bias_block1r_con5,DataType:: S8);

		WeightsInfo weights_block1r_con5(false,1,1,256,false);

		const TensorShape out_shape_block1r_con5(56, 56,256);

		out_block1r_con5.allocator()->init(TensorInfo(out_shape_block1r_con5, 1, DataType:: S8));

   		/*add-act*/

		TensorShape out_shape_block1_1 = out_shape_block1r_con5;

		out_block1_add1.allocator()->init(TensorInfo(out_shape_block1_1, 1, DataType:: S8));

		out_block1_act1.allocator()->init(TensorInfo(out_shape_block1_1, 1, DataType:: F32));

  		 /*conv-batch-act*/

		NPYLoader npy8_q;npy8_q.open(Q_table_datapath+Q_table_name[8]);npy8_q.init_tensor2(Q_table_block1r_con6,DataType:: S16);

		NPYLoader npy8_wt;npy8_wt.open(WT_buffer_datapath+WT_buffer_name[8]);npy8_wt.init_tensor2(WT_buffer_block1r_con6,DataType::U16);

		NPYLoader npy8_b;npy8_b.open(bias_datapath+bias_name[8]);npy8_b.init_tensor2(bias_block1r_con6,DataType:: S8);

		WeightsInfo weights_block1r_con6(false,1,1,64,false);

		const TensorShape out_shape_block1r_con6(56, 56, 64);

		out_block1r_con6.allocator()->init(TensorInfo(out_shape_block1r_con6, 1, DataType:: S8));

		out_block1r_act4.allocator()->init(TensorInfo(out_shape_block1r_con6, 1, DataType:: F32));

  		 /*conv-batch-act*/

		NPYLoader npy9_q;npy9_q.open(Q_table_datapath+Q_table_name[9]);npy9_q.init_tensor2(Q_table_block1r_con7,DataType:: S16);

		NPYLoader npy9_wt;npy9_wt.open(WT_buffer_datapath+WT_buffer_name[9]);npy9_wt.init_tensor2(WT_buffer_block1r_con7,DataType::U16);

		NPYLoader npy9_b;npy9_b.open(bias_datapath+bias_name[9]);npy9_b.init_tensor2(bias_block1r_con7,DataType:: S8);

		WeightsInfo weights_block1r_con7(false,3,3,64,false);

		const TensorShape out_shape_block1r_con7(56, 56,64);

		out_block1r_con7.allocator()->init(TensorInfo(out_shape_block1r_con7, 1, DataType:: S8));

		out_block1r_act5.allocator()->init(TensorInfo(out_shape_block1r_con7, 1, DataType:: F32));

   		/*conv-batch*/

		NPYLoader npy10_q;npy10_q.open(Q_table_datapath+Q_table_name[10]);npy10_q.init_tensor2(Q_table_block1r_con8,DataType:: S16);

		NPYLoader npy10_wt;npy10_wt.open(WT_buffer_datapath+WT_buffer_name[10]);npy10_wt.init_tensor2(WT_buffer_block1r_con8,DataType::U16);

		NPYLoader npy10_b;npy10_b.open(bias_datapath+bias_name[10]);npy10_b.init_tensor2(bias_block1r_con8,DataType:: S8);

		WeightsInfo weights_block1r_con8(false,1,1,256,false);

		const TensorShape out_shape_block1r_con8(56, 56,256);

		out_block1r_con8.allocator()->init(TensorInfo(out_shape_block1r_con8, 1, DataType:: S8));

		/*add-act*/

		TensorShape out_shape_block1_2 = out_shape_block1r_con8;

		out_block1_add2.allocator()->init(TensorInfo(out_shape_block1_2, 1, DataType:: S8));

		out_block1_act2.allocator()->init(TensorInfo(out_shape_block1_2, 1, DataType:: F32));



		

		/*block2*/

   		/*conv-batch-act*/

        NPYLoader npy11_q;npy11_q.open(Q_table_datapath+Q_table_name[11]);npy11_q.init_tensor2(Q_table_block2r_con0,DataType:: S16);

		NPYLoader npy11_wt;npy11_wt.open(WT_buffer_datapath+WT_buffer_name[11]);npy11_wt.init_tensor2(WT_buffer_block2r_con0,DataType::U16);

		NPYLoader npy11_b;npy11_b.open(bias_datapath+bias_name[11]);npy11_b.init_tensor2(bias_block2r_con0,DataType:: S8);

		WeightsInfo weights_block2r_con0(false,1,1,128,false);

		const TensorShape out_shape_block2r_con0(28, 28, 128);

		out_block2r_con0.allocator()->init(TensorInfo(out_shape_block2r_con0, 1, DataType:: S8));

		out_block2r_act0.allocator()->init(TensorInfo(out_shape_block2r_con0, 1, DataType:: F32));

   		/*conv-batch-act*/

		NPYLoader npy12_q;npy12_q.open(Q_table_datapath+Q_table_name[12]);npy12_q.init_tensor2(Q_table_block2r_con1,DataType:: S16);

		NPYLoader npy12_wt;npy12_wt.open(WT_buffer_datapath+WT_buffer_name[12]);npy12_wt.init_tensor2(WT_buffer_block2r_con1,DataType::U16);

		NPYLoader npy12_b;npy12_b.open(bias_datapath+bias_name[12]);npy12_b.init_tensor2(bias_block2r_con1,DataType:: S8);

		WeightsInfo weights_block2r_con1(false,3,3,128,false);

		const TensorShape out_shape_block2r_con1(28, 28, 128);

		out_block2r_con1.allocator()->init(TensorInfo(out_shape_block2r_con1, 1, DataType:: S8));

		out_block2r_act1.allocator()->init(TensorInfo(out_shape_block2r_con1, 1, DataType:: F32));

   		/*conv-batch*/

		NPYLoader npy13_q;npy13_q.open(Q_table_datapath+Q_table_name[13]);npy13_q.init_tensor2(Q_table_block2r_con2,DataType:: S16);

		NPYLoader npy13_wt;npy13_wt.open(WT_buffer_datapath+WT_buffer_name[13]);npy13_wt.init_tensor2(WT_buffer_block2r_con2,DataType::U16);

		NPYLoader npy13_b;npy13_b.open(bias_datapath+bias_name[13]);npy13_b.init_tensor2(bias_block2r_con2,DataType:: S8);

		WeightsInfo weights_block2r_con2(false,1,1,512,false);

		const TensorShape out_shape_block2r_con2(28, 28, 512);

		out_block2r_con2.allocator()->init(TensorInfo(out_shape_block2r_con2, 1, DataType:: S8));

   		/*conv-batch*/

		NPYLoader npy14_q;npy14_q.open(Q_table_datapath+Q_table_name[14]);npy14_q.init_tensor2(Q_table_block2l_con0,DataType:: S16);

		NPYLoader npy14_wt;npy14_wt.open(WT_buffer_datapath+WT_buffer_name[14]);npy14_wt.init_tensor2(WT_buffer_block2l_con0,DataType::U16);

		NPYLoader npy14_b;npy14_b.open(bias_datapath+bias_name[14]);npy14_b.init_tensor2(bias_block2l_con0,DataType:: S8);

		WeightsInfo weights_block2l_con0(false,1,1,512,false);

		const TensorShape out_shape_block2l_con0(28, 28, 512);

		out_block2l_con0.allocator()->init(TensorInfo(out_shape_block2l_con0, 1, DataType:: S8));

   		/*add-act*/

		TensorShape out_shape_block2_0 = out_shape_block2r_con2;

		out_block2_add0.allocator()->init(TensorInfo(out_shape_block2_0, 1, DataType:: S8));

		out_block2_act0.allocator()->init(TensorInfo(out_shape_block2_0, 1, DataType:: F32));

   		/*conv-batch-act*/

        NPYLoader npy15_q;npy15_q.open(Q_table_datapath+Q_table_name[15]);npy15_q.init_tensor2(Q_table_block2r_con3,DataType:: S16);

		NPYLoader npy15_wt;npy15_wt.open(WT_buffer_datapath+WT_buffer_name[15]);npy15_wt.init_tensor2(WT_buffer_block2r_con3,DataType::U16);

		NPYLoader npy15_b;npy15_b.open(bias_datapath+bias_name[15]);npy15_b.init_tensor2(bias_block2r_con3,DataType:: S8);

		WeightsInfo weights_block2r_con3(false,1,1,128,false);

		const TensorShape out_shape_block2r_con3(28, 28, 128);

		out_block2r_con3.allocator()->init(TensorInfo(out_shape_block2r_con3, 1, DataType:: S8));

		out_block2r_act2.allocator()->init(TensorInfo(out_shape_block2r_con3, 1, DataType:: F32));

  		 /*conv-batch-act*/

		NPYLoader npy16_q;npy16_q.open(Q_table_datapath+Q_table_name[16]);npy16_q.init_tensor2(Q_table_block2r_con4,DataType:: S16);

		NPYLoader npy16_wt;npy16_wt.open(WT_buffer_datapath+WT_buffer_name[16]);npy16_wt.init_tensor2(WT_buffer_block2r_con4,DataType::U16);

		NPYLoader npy16_b;npy16_b.open(bias_datapath+bias_name[16]);npy16_b.init_tensor2(bias_block2r_con4,DataType:: S8);

		WeightsInfo weights_block2r_con4(false,3,3,128,false);

		const TensorShape out_shape_block2r_con4(28, 28,128);

		out_block2r_con4.allocator()->init(TensorInfo(out_shape_block2r_con4, 1, DataType:: S8));

		out_block2r_act3.allocator()->init(TensorInfo(out_shape_block2r_con4, 1, DataType:: F32));

 		/*conv-batch*/

		NPYLoader npy17_q;npy17_q.open(Q_table_datapath+Q_table_name[17]);npy17_q.init_tensor2(Q_table_block2r_con5,DataType:: S16);

		NPYLoader npy17_wt;npy17_wt.open(WT_buffer_datapath+WT_buffer_name[17]);npy17_wt.init_tensor2(WT_buffer_block2r_con5,DataType::U16);

		NPYLoader npy17_b;npy17_b.open(bias_datapath+bias_name[17]);npy17_b.init_tensor2(bias_block2r_con5,DataType:: S8);

		WeightsInfo weights_block2r_con5(false,1,1,512,false);

		const TensorShape out_shape_block2r_con5(28, 28,512);

		out_block2r_con5.allocator()->init(TensorInfo(out_shape_block2r_con5, 1, DataType:: S8));

  		 /*add-act*/

		TensorShape out_shape_block2_1 = out_shape_block2r_con5;

		out_block2_add1.allocator()->init(TensorInfo(out_shape_block2_1, 1, DataType:: S8));

		out_block2_act1.allocator()->init(TensorInfo(out_shape_block2_1, 1, DataType:: F32));

  		 /*conv-batch-act*/

        NPYLoader npy18_q;npy18_q.open(Q_table_datapath+Q_table_name[18]);npy18_q.init_tensor2(Q_table_block2r_con6,DataType:: S16);

		NPYLoader npy18_wt;npy18_wt.open(WT_buffer_datapath+WT_buffer_name[18]);npy18_wt.init_tensor2(WT_buffer_block2r_con6,DataType::U16);

		NPYLoader npy18_b;npy18_b.open(bias_datapath+bias_name[18]);npy18_b.init_tensor2(bias_block2r_con6,DataType:: S8);

		WeightsInfo weights_block2r_con6(false,1,1,128,false);

		const TensorShape out_shape_block2r_con6(28, 28, 128);

		out_block2r_con6.allocator()->init(TensorInfo(out_shape_block2r_con6, 1, DataType:: S8));

		out_block2r_act4.allocator()->init(TensorInfo(out_shape_block2r_con6, 1, DataType:: F32));

  		 /*conv-batch-act*/

		NPYLoader npy19_q;npy19_q.open(Q_table_datapath+Q_table_name[19]);npy19_q.init_tensor2(Q_table_block2r_con7,DataType:: S16);

		NPYLoader npy19_wt;npy19_wt.open(WT_buffer_datapath+WT_buffer_name[19]);npy19_wt.init_tensor2(WT_buffer_block2r_con7,DataType::U16);

		NPYLoader npy19_b;npy19_b.open(bias_datapath+bias_name[19]);npy19_b.init_tensor2(bias_block2r_con7,DataType:: S8);

		WeightsInfo weights_block2r_con7(false,3,3,128,false);

		const TensorShape out_shape_block2r_con7(28, 28, 128);

		out_block2r_con7.allocator()->init(TensorInfo(out_shape_block2r_con7, 1, DataType:: S8));

		out_block2r_act5.allocator()->init(TensorInfo(out_shape_block2r_con7, 1, DataType:: F32));

   		/*conv-batch*/

		NPYLoader npy20_q;npy20_q.open(Q_table_datapath+Q_table_name[20]);npy20_q.init_tensor2(Q_table_block2r_con8,DataType:: S16);

		NPYLoader npy20_wt;npy20_wt.open(WT_buffer_datapath+WT_buffer_name[20]);npy20_wt.init_tensor2(WT_buffer_block2r_con8,DataType::U16);

		NPYLoader npy20_b;npy20_b.open(bias_datapath+bias_name[20]);npy20_b.init_tensor2(bias_block2r_con8,DataType:: S8);

		WeightsInfo weights_block2r_con8(false,1,1,512,false);

		const TensorShape out_shape_block2r_con8(28, 28, 512);

		out_block2r_con8.allocator()->init(TensorInfo(out_shape_block2r_con8, 1, DataType:: S8));

   		/*add-act*/

		TensorShape out_shape_block2_2 = out_shape_block2r_con8;

		out_block2_add2.allocator()->init(TensorInfo(out_shape_block2_2, 1, DataType:: S8));

		out_block2_act2.allocator()->init(TensorInfo(out_shape_block2_2, 1, DataType:: F32));

   		/*conv-batch-act*/

        NPYLoader npy21_q;npy21_q.open(Q_table_datapath+Q_table_name[21]);npy21_q.init_tensor2(Q_table_block2r_con9,DataType:: S16);

		NPYLoader npy21_wt;npy21_wt.open(WT_buffer_datapath+WT_buffer_name[21]);npy21_wt.init_tensor2(WT_buffer_block2r_con9,DataType::U16);

		NPYLoader npy21_b;npy21_b.open(bias_datapath+bias_name[21]);npy21_b.init_tensor2(bias_block2r_con9,DataType:: S8);

		WeightsInfo weights_block2r_con9(false,1,1,128,false);

		const TensorShape out_shape_block2r_con9(28, 28,128);

		out_block2r_con9.allocator()->init(TensorInfo(out_shape_block2r_con9, 1, DataType:: S8));

		out_block2r_act6.allocator()->init(TensorInfo(out_shape_block2r_con9, 1, DataType:: F32));

   		/*conv-batch-act*/

		NPYLoader npy22_q;npy22_q.open(Q_table_datapath+Q_table_name[22]);npy22_q.init_tensor2(Q_table_block2r_con10,DataType:: S16);

		NPYLoader npy22_wt;npy22_wt.open(WT_buffer_datapath+WT_buffer_name[22]);npy22_wt.init_tensor2(WT_buffer_block2r_con10,DataType::U16);

		NPYLoader npy22_b;npy22_b.open(bias_datapath+bias_name[22]);npy22_b.init_tensor2(bias_block2r_con10,DataType:: S8);

		WeightsInfo weights_block2r_con10(false,3,3,128,false);

		const TensorShape out_shape_block2r_con10(28, 28, 128);

		out_block2r_con10.allocator()->init(TensorInfo(out_shape_block2r_con10, 1, DataType:: S8));

		out_block2r_act7.allocator()->init(TensorInfo(out_shape_block2r_con10, 1, DataType:: F32));

   		/*conv-batch*/

		NPYLoader npy23_q;npy23_q.open(Q_table_datapath+Q_table_name[23]);npy23_q.init_tensor2(Q_table_block2r_con11,DataType:: S16);

		NPYLoader npy23_wt;npy23_wt.open(WT_buffer_datapath+WT_buffer_name[23]);npy23_wt.init_tensor2(WT_buffer_block2r_con11,DataType::U16);

		NPYLoader npy23_b;npy23_b.open(bias_datapath+bias_name[23]);npy23_b.init_tensor2(bias_block2r_con11,DataType:: S8);

		WeightsInfo weights_block2r_con11(false,1,1,512,false);

		const TensorShape out_shape_block2r_con11(28, 28, 512);

		out_block2r_con11.allocator()->init(TensorInfo(out_shape_block2r_con11, 1, DataType:: S8));

  		 /*add-act*/

		TensorShape out_shape_block2_3 = out_shape_block2r_con11;

		out_block2_add3.allocator()->init(TensorInfo(out_shape_block2_3, 1, DataType:: S8));

		out_block2_act3.allocator()->init(TensorInfo(out_shape_block2_3, 1, DataType:: F32));



		/*block3*/

  		 /*conv-batch-act*/

        NPYLoader npy24_q;npy24_q.open(Q_table_datapath+Q_table_name[24]);npy24_q.init_tensor2(Q_table_block3r_con0,DataType:: S16);

		NPYLoader npy24_wt;npy24_wt.open(WT_buffer_datapath+WT_buffer_name[24]);npy24_wt.init_tensor2(WT_buffer_block3r_con0,DataType::U16);

		NPYLoader npy24_b;npy24_b.open(bias_datapath+bias_name[24]);npy24_b.init_tensor2(bias_block3r_con0,DataType:: S8);

		WeightsInfo weights_block3r_con0(false,1,1,256,false);

		const TensorShape out_shape_block3r_con0(14, 14, 256);

		out_block3r_con0.allocator()->init(TensorInfo(out_shape_block3r_con0, 1, DataType:: S8));

		out_block3r_act0.allocator()->init(TensorInfo(out_shape_block3r_con0, 1, DataType:: F32));

		/*conv-batch-act*/

		NPYLoader npy25_q;npy25_q.open(Q_table_datapath+Q_table_name[25]);npy25_q.init_tensor2(Q_table_block3r_con1,DataType:: S16);

		NPYLoader npy25_wt;npy25_wt.open(WT_buffer_datapath+WT_buffer_name[25]);npy25_wt.init_tensor2(WT_buffer_block3r_con1,DataType::U16);

		NPYLoader npy25_b;npy25_b.open(bias_datapath+bias_name[25]);npy25_b.init_tensor2(bias_block3r_con1,DataType:: S8);

		WeightsInfo weights_block3r_con1(false,3,3,256,false);

		const TensorShape out_shape_block3r_con1(14, 14, 256);

		out_block3r_con1.allocator()->init(TensorInfo(out_shape_block3r_con1, 1, DataType:: S8));

		out_block3r_act1.allocator()->init(TensorInfo(out_shape_block3r_con1, 1, DataType:: F32));

   		/*conv-batch		*/

		NPYLoader npy26_q;npy26_q.open(Q_table_datapath+Q_table_name[26]);npy26_q.init_tensor2(Q_table_block3r_con2,DataType:: S16);

		NPYLoader npy26_wt;npy26_wt.open(WT_buffer_datapath+WT_buffer_name[26]);npy26_wt.init_tensor2(WT_buffer_block3r_con2,DataType::U16);

		NPYLoader npy26_b;npy26_b.open(bias_datapath+bias_name[26]);npy26_b.init_tensor2(bias_block3r_con2,DataType:: S8);

		WeightsInfo weights_block3r_con2(false,1,1,1024,false);

		const TensorShape out_shape_block3r_con2(14, 14, 1024);

		out_block3r_con2.allocator()->init(TensorInfo(out_shape_block3r_con2, 1, DataType:: S8));

  		 /*conv-batch*/

		NPYLoader npy27_q;npy27_q.open(Q_table_datapath+Q_table_name[27]);npy27_q.init_tensor2(Q_table_block3l_con0,DataType:: S16);

		NPYLoader npy27_wt;npy27_wt.open(WT_buffer_datapath+WT_buffer_name[27]);npy27_wt.init_tensor2(WT_buffer_block3l_con0,DataType::U16);

		NPYLoader npy27_b;npy27_b.open(bias_datapath+bias_name[27]);npy27_b.init_tensor2(bias_block3l_con0,DataType:: S8);

		WeightsInfo weights_block3l_con0(false,1,1,1024,false);

		const TensorShape out_shape_block3l_con0(14, 14, 1024);

		out_block3l_con0.allocator()->init(TensorInfo(out_shape_block3l_con0, 1, DataType:: S8));

   		/*add-act*/

		TensorShape out_shape_block3_0 = out_shape_block3r_con2;

		out_block3_add0.allocator()->init(TensorInfo(out_shape_block3_0, 1, DataType:: S8));

		out_block3_act0.allocator()->init(TensorInfo(out_shape_block3_0, 1, DataType:: F32));

   		/*conv-batch-act*/

		NPYLoader npy28_q;npy28_q.open(Q_table_datapath+Q_table_name[28]);npy28_q.init_tensor2(Q_table_block3r_con3,DataType:: S16);

		NPYLoader npy28_wt;npy28_wt.open(WT_buffer_datapath+WT_buffer_name[28]);npy28_wt.init_tensor2(WT_buffer_block3r_con3,DataType::U16);

		NPYLoader npy28_b;npy28_b.open(bias_datapath+bias_name[28]);npy28_b.init_tensor2(bias_block3r_con3,DataType:: S8);

		WeightsInfo weights_block3r_con3(false,1,1,256,false);

		const TensorShape out_shape_block3r_con3(14, 14,256);

		out_block3r_con3.allocator()->init(TensorInfo(out_shape_block3r_con3, 1, DataType:: S8));

		out_block3r_act2.allocator()->init(TensorInfo(out_shape_block3r_con3, 1, DataType:: F32));

  		 /*conv-batch-act		*/

		NPYLoader npy29_q;npy29_q.open(Q_table_datapath+Q_table_name[29]);npy29_q.init_tensor2(Q_table_block3r_con4,DataType:: S16);

		NPYLoader npy29_wt;npy29_wt.open(WT_buffer_datapath+WT_buffer_name[29]);npy29_wt.init_tensor2(WT_buffer_block3r_con4,DataType::U16);

		NPYLoader npy29_b;npy29_b.open(bias_datapath+bias_name[29]);npy29_b.init_tensor2(bias_block3r_con4,DataType:: S8);

		WeightsInfo weights_block3r_con4(false,3,3,256,false);

		const TensorShape out_shape_block3r_con4(14, 14,256);

		out_block3r_con4.allocator()->init(TensorInfo(out_shape_block3r_con4, 1, DataType:: S8));

		out_block3r_act3.allocator()->init(TensorInfo(out_shape_block3r_con4, 1, DataType:: F32));

   		/*conv-batch		*/

		NPYLoader npy30_q;npy30_q.open(Q_table_datapath+Q_table_name[30]);npy30_q.init_tensor2(Q_table_block3r_con5,DataType:: S16);

		NPYLoader npy30_wt;npy30_wt.open(WT_buffer_datapath+WT_buffer_name[30]);npy30_wt.init_tensor2(WT_buffer_block3r_con5,DataType::U16);

		NPYLoader npy30_b;npy30_b.open(bias_datapath+bias_name[30]);npy30_b.init_tensor2(bias_block3r_con5,DataType:: S8);

		WeightsInfo weights_block3r_con5(false,1,1,1024,false);

		const TensorShape out_shape_block3r_con5(14, 14, 1024);

		out_block3r_con5.allocator()->init(TensorInfo(out_shape_block3r_con5, 1, DataType:: S8));

   		/*add-act		*/

		TensorShape out_shape_block3_1 = out_shape_block3r_con5;

		out_block3_add1.allocator()->init(TensorInfo(out_shape_block3_1, 1, DataType:: S8));

		out_block3_act1.allocator()->init(TensorInfo(out_shape_block3_1, 1, DataType:: F32));

   		/*conv-batch-act*/

		NPYLoader npy31_q;npy31_q.open(Q_table_datapath+Q_table_name[31]);npy31_q.init_tensor2(Q_table_block3r_con6,DataType:: S16);

		NPYLoader npy31_wt;npy31_wt.open(WT_buffer_datapath+WT_buffer_name[31]);npy31_wt.init_tensor2(WT_buffer_block3r_con6,DataType::U16);

		NPYLoader npy31_b;npy31_b.open(bias_datapath+bias_name[31]);npy31_b.init_tensor2(bias_block3r_con6,DataType:: S8);

		WeightsInfo weights_block3r_con6(false,1,1,256,false);

		const TensorShape out_shape_block3r_con6(14, 14, 256);

		out_block3r_con6.allocator()->init(TensorInfo(out_shape_block3r_con6, 1, DataType:: S8));

		out_block3r_act4.allocator()->init(TensorInfo(out_shape_block3r_con6, 1, DataType:: F32));

  		 /*conv-batch-act		*/

		NPYLoader npy32_q;npy32_q.open(Q_table_datapath+Q_table_name[32]);npy32_q.init_tensor2(Q_table_block3r_con7,DataType:: S16);

		NPYLoader npy32_wt;npy32_wt.open(WT_buffer_datapath+WT_buffer_name[32]);npy32_wt.init_tensor2(WT_buffer_block3r_con7,DataType::U16);

		NPYLoader npy32_b;npy32_b.open(bias_datapath+bias_name[32]);npy32_b.init_tensor2(bias_block3r_con7,DataType:: S8);

		WeightsInfo weights_block3r_con7(false,3,3,256,false);

		const TensorShape out_shape_block3r_con7(14, 14, 256);

		out_block3r_con7.allocator()->init(TensorInfo(out_shape_block3r_con7, 1, DataType:: S8));

		out_block3r_act5.allocator()->init(TensorInfo(out_shape_block3r_con7, 1, DataType:: F32));

   		/*conv-batch		*/

		NPYLoader npy33_q;npy33_q.open(Q_table_datapath+Q_table_name[33]);npy33_q.init_tensor2(Q_table_block3r_con8,DataType:: S16);

		NPYLoader npy33_wt;npy33_wt.open(WT_buffer_datapath+WT_buffer_name[33]);npy33_wt.init_tensor2(WT_buffer_block3r_con8,DataType::U16);

		NPYLoader npy33_b;npy33_b.open(bias_datapath+bias_name[33]);npy33_b.init_tensor2(bias_block3r_con8,DataType:: S8);

		WeightsInfo weights_block3r_con8(false,1,1,1024,false);

		const TensorShape out_shape_block3r_con8(14, 14, 1024);

		out_block3r_con8.allocator()->init(TensorInfo(out_shape_block3r_con8, 1, DataType:: S8));

   		/*add-act		*/

		TensorShape out_shape_block3_2 = out_shape_block3r_con8;

		out_block3_add2.allocator()->init(TensorInfo(out_shape_block3_2, 1, DataType:: S8));

		out_block3_act2.allocator()->init(TensorInfo(out_shape_block3_2, 1, DataType:: F32));

  		 /*conv-batch-act*/

		NPYLoader npy34_q;npy34_q.open(Q_table_datapath+Q_table_name[34]);npy34_q.init_tensor2(Q_table_block3r_con9,DataType:: S16);

		NPYLoader npy34_wt;npy34_wt.open(WT_buffer_datapath+WT_buffer_name[34]);npy34_wt.init_tensor2(WT_buffer_block3r_con9,DataType::U16);

		NPYLoader npy34_b;npy34_b.open(bias_datapath+bias_name[34]);npy34_b.init_tensor2(bias_block3r_con9,DataType:: S8);

		WeightsInfo weights_block3r_con9(false,1,1,256,false);

		const TensorShape out_shape_block3r_con9(14, 14, 256);

		out_block3r_con9.allocator()->init(TensorInfo(out_shape_block3r_con9, 1, DataType:: S8));

		out_block3r_act6.allocator()->init(TensorInfo(out_shape_block3r_con9, 1, DataType:: F32));

  		 /*conv-batch-act		*/

		NPYLoader npy35_q;npy35_q.open(Q_table_datapath+Q_table_name[35]);npy35_q.init_tensor2(Q_table_block3r_con10,DataType:: S16);

		NPYLoader npy35_wt;npy35_wt.open(WT_buffer_datapath+WT_buffer_name[35]);npy35_wt.init_tensor2(WT_buffer_block3r_con10,DataType::U16);

		NPYLoader npy35_b;npy35_b.open(bias_datapath+bias_name[35]);npy35_b.init_tensor2(bias_block3r_con10,DataType:: S8);

		WeightsInfo weights_block3r_con10(false,3,3,256,false);

		const TensorShape out_shape_block3r_con10(14, 14, 256);

		out_block3r_con10.allocator()->init(TensorInfo(out_shape_block3r_con10, 1, DataType:: S8));

		out_block3r_act7.allocator()->init(TensorInfo(out_shape_block3r_con10, 1, DataType:: F32));

   		/*conv-batch		*/

		NPYLoader npy36_q;npy36_q.open(Q_table_datapath+Q_table_name[36]);npy36_q.init_tensor2(Q_table_block3r_con11,DataType:: S16);

		NPYLoader npy36_wt;npy36_wt.open(WT_buffer_datapath+WT_buffer_name[36]);npy36_wt.init_tensor2(WT_buffer_block3r_con11,DataType::U16);

		NPYLoader npy36_b;npy36_b.open(bias_datapath+bias_name[36]);npy36_b.init_tensor2(bias_block3r_con11,DataType:: S8);

		WeightsInfo weights_block3r_con11(false,1,1,1024,false);

		const TensorShape out_shape_block3r_con11(14, 14, 1024);

		out_block3r_con11.allocator()->init(TensorInfo(out_shape_block3r_con11, 1, DataType:: S8));

   		/*add-act		*/

		TensorShape out_shape_block3_3 = out_shape_block3r_con11;

		out_block3_add3.allocator()->init(TensorInfo(out_shape_block3_3, 1, DataType:: S8));

		out_block3_act3.allocator()->init(TensorInfo(out_shape_block3_3, 1, DataType:: F32));

   		/*conv-batch-act*/

		NPYLoader npy37_q;npy37_q.open(Q_table_datapath+Q_table_name[37]);npy37_q.init_tensor2(Q_table_block3r_con12,DataType:: S16);

		NPYLoader npy37_wt;npy37_wt.open(WT_buffer_datapath+WT_buffer_name[37]);npy37_wt.init_tensor2(WT_buffer_block3r_con12,DataType::U16);

		NPYLoader npy37_b;npy37_b.open(bias_datapath+bias_name[37]);npy37_b.init_tensor2(bias_block3r_con12,DataType:: S8);

		WeightsInfo weights_block3r_con12(false,1,1,256,false);

		const TensorShape out_shape_block3r_con12(14, 14, 256);

		out_block3r_con12.allocator()->init(TensorInfo(out_shape_block3r_con12, 1, DataType:: S8));

		out_block3r_act8.allocator()->init(TensorInfo(out_shape_block3r_con12, 1, DataType:: F32));

   		/*conv-batch-act		*/

		NPYLoader npy38_q;npy38_q.open(Q_table_datapath+Q_table_name[38]);npy38_q.init_tensor2(Q_table_block3r_con13,DataType:: S16);

		NPYLoader npy38_wt;npy38_wt.open(WT_buffer_datapath+WT_buffer_name[38]);npy38_wt.init_tensor2(WT_buffer_block3r_con13,DataType::U16);

		NPYLoader npy38_b;npy38_b.open(bias_datapath+bias_name[38]);npy38_b.init_tensor2(bias_block3r_con13,DataType:: S8);

		WeightsInfo weights_block3r_con13(false,3,3,256,false);

		const TensorShape out_shape_block3r_con13(14, 14, 256);

		out_block3r_con13.allocator()->init(TensorInfo(out_shape_block3r_con13, 1, DataType:: S8));

		out_block3r_act9.allocator()->init(TensorInfo(out_shape_block3r_con13, 1, DataType:: F32));

   		/*conv-batch		*/

		NPYLoader npy39_q;npy39_q.open(Q_table_datapath+Q_table_name[39]);npy39_q.init_tensor2(Q_table_block3r_con14,DataType:: S16);

		NPYLoader npy39_wt;npy39_wt.open(WT_buffer_datapath+WT_buffer_name[39]);npy39_wt.init_tensor2(WT_buffer_block3r_con14,DataType::U16);

		NPYLoader npy39_b;npy39_b.open(bias_datapath+bias_name[39]);npy39_b.init_tensor2(bias_block3r_con14,DataType:: S8);

		WeightsInfo weights_block3r_con14(false,1,1,1024,false);

		const TensorShape out_shape_block3r_con14(14, 14, 1024);

		out_block3r_con14.allocator()->init(TensorInfo(out_shape_block3r_con14, 1, DataType:: S8));

  		 /*add-act		*/

		TensorShape out_shape_block3_4 = out_shape_block3r_con14;

		out_block3_add4.allocator()->init(TensorInfo(out_shape_block3_4, 1, DataType:: S8));

		out_block3_act4.allocator()->init(TensorInfo(out_shape_block3_4, 1, DataType:: F32));

   		/*conv-batch-act*/

		NPYLoader npy40_q;npy40_q.open(Q_table_datapath+Q_table_name[40]);npy40_q.init_tensor2(Q_table_block3r_con15,DataType:: S16);

		NPYLoader npy40_wt;npy40_wt.open(WT_buffer_datapath+WT_buffer_name[40]);npy40_wt.init_tensor2(WT_buffer_block3r_con15,DataType::U16);

		NPYLoader npy40_b;npy40_b.open(bias_datapath+bias_name[40]);npy40_b.init_tensor2(bias_block3r_con15,DataType:: S8);

		WeightsInfo weights_block3r_con15(false,1,1,256,false);

		const TensorShape out_shape_block3r_con15(14, 14,256);

		out_block3r_con15.allocator()->init(TensorInfo(out_shape_block3r_con15, 1, DataType:: S8));

		out_block3r_act10.allocator()->init(TensorInfo(out_shape_block3r_con15, 1, DataType:: F32));

   		/*conv-batch-act		*/

		NPYLoader npy41_q;npy41_q.open(Q_table_datapath+Q_table_name[41]);npy41_q.init_tensor2(Q_table_block3r_con16,DataType:: S16);

		NPYLoader npy41_wt;npy41_wt.open(WT_buffer_datapath+WT_buffer_name[41]);npy41_wt.init_tensor2(WT_buffer_block3r_con16,DataType::U16);

		NPYLoader npy41_b;npy41_b.open(bias_datapath+bias_name[41]);npy41_b.init_tensor2(bias_block3r_con16,DataType:: S8);

		WeightsInfo weights_block3r_con16(false,3,3,256,false);

		const TensorShape out_shape_block3r_con16(14, 14, 256);

		out_block3r_con16.allocator()->init(TensorInfo(out_shape_block3r_con16, 1, DataType:: S8));

		out_block3r_act11.allocator()->init(TensorInfo(out_shape_block3r_con16, 1, DataType:: F32));

   		/*conv-batch	*/

		NPYLoader npy42_q;npy42_q.open(Q_table_datapath+Q_table_name[42]);npy42_q.init_tensor2(Q_table_block3r_con17,DataType:: S16);

		NPYLoader npy42_wt;npy42_wt.open(WT_buffer_datapath+WT_buffer_name[42]);npy42_wt.init_tensor2(WT_buffer_block3r_con17,DataType::U16);

		NPYLoader npy42_b;npy42_b.open(bias_datapath+bias_name[42]);npy42_b.init_tensor2(bias_block3r_con17,DataType:: S8);

		WeightsInfo weights_block3r_con17(false,1,1,1024,false);

		const TensorShape out_shape_block3r_con17(14, 14, 1024);

		out_block3r_con17.allocator()->init(TensorInfo(out_shape_block3r_con17, 1, DataType:: S8));

		/*add-act		*/

		TensorShape out_shape_block3_5 = out_shape_block3r_con17;

		out_block3_add5.allocator()->init(TensorInfo(out_shape_block3_5, 1, DataType:: S8));

		out_block3_act5.allocator()->init(TensorInfo(out_shape_block3_5, 1, DataType:: F32));



		/*block4*/

   		/*conv-batch-act*/

		NPYLoader npy43_q;npy43_q.open(Q_table_datapath+Q_table_name[43]);npy43_q.init_tensor2(Q_table_block4r_con0,DataType:: S16);

		NPYLoader npy43_wt;npy43_wt.open(WT_buffer_datapath+WT_buffer_name[43]);npy43_wt.init_tensor2(WT_buffer_block4r_con0,DataType::U16);

		NPYLoader npy43_b;npy43_b.open(bias_datapath+bias_name[43]);npy43_b.init_tensor2(bias_block4r_con0,DataType:: S8);

		WeightsInfo weights_block4r_con0(false,1,1,512,false);

		const TensorShape out_shape_block4r_con0(7, 7, 512);

		out_block4r_con0.allocator()->init(TensorInfo(out_shape_block4r_con0, 1, DataType:: S8));

		out_block4r_act0.allocator()->init(TensorInfo(out_shape_block4r_con0, 1, DataType:: F32));

  		 /*conv-batch-act*/

		NPYLoader npy44_q;npy44_q.open(Q_table_datapath+Q_table_name[44]);npy44_q.init_tensor2(Q_table_block4r_con1,DataType:: S16);

		NPYLoader npy44_wt;npy44_wt.open(WT_buffer_datapath+WT_buffer_name[44]);npy44_wt.init_tensor2(WT_buffer_block4r_con1,DataType::U16);

		NPYLoader npy44_b;npy44_b.open(bias_datapath+bias_name[44]);npy44_b.init_tensor2(bias_block4r_con1,DataType:: S8);

		WeightsInfo weights_block4r_con1(false,3,3,512,false);

		const TensorShape out_shape_block4r_con1(7, 7,512);

		out_block4r_con1.allocator()->init(TensorInfo(out_shape_block4r_con1, 1, DataType:: S8));

		out_block4r_act1.allocator()->init(TensorInfo(out_shape_block4r_con1, 1, DataType:: F32));

  		 /*conv-batch*/

		NPYLoader npy45_q;npy45_q.open(Q_table_datapath+Q_table_name[45]);npy45_q.init_tensor2(Q_table_block4r_con2,DataType:: S16);

		NPYLoader npy45_wt;npy45_wt.open(WT_buffer_datapath+WT_buffer_name[45]);npy45_wt.init_tensor2(WT_buffer_block4r_con2,DataType::U16);

		NPYLoader npy45_b;npy45_b.open(bias_datapath+bias_name[45]);npy45_b.init_tensor2(bias_block4r_con2,DataType:: S8);

		WeightsInfo weights_block4r_con2(false,1,1,2048,false);

		const TensorShape out_shape_block4r_con2(7, 7, 2048);

		out_block4r_con2.allocator()->init(TensorInfo(out_shape_block4r_con2, 1, DataType:: S8));

  		/*conv-batch*/

		NPYLoader npy46_q;npy46_q.open(Q_table_datapath+Q_table_name[46]);npy46_q.init_tensor2(Q_table_block4l_con0,DataType:: S16);

		NPYLoader npy46_wt;npy46_wt.open(WT_buffer_datapath+WT_buffer_name[46]);npy46_wt.init_tensor2(WT_buffer_block4l_con0,DataType::U16);

		NPYLoader npy46_b;npy46_b.open(bias_datapath+bias_name[46]);npy46_b.init_tensor2(bias_block4l_con0,DataType:: S8);

		WeightsInfo weights_block4l_con0(false,1,1,2048,false);

		const TensorShape out_shape_block4l_con0(7, 7, 2048);

		out_block4l_con0.allocator()->init(TensorInfo(out_shape_block4l_con0, 1, DataType:: S8));

   		/*add-act*/

		TensorShape out_shape_block4_0 = out_shape_block4r_con2;

		out_block4_add0.allocator()->init(TensorInfo(out_shape_block4_0, 1, DataType:: S8));

		out_block4_act0.allocator()->init(TensorInfo(out_shape_block4_0, 1, DataType:: F32));

   		/*conv-batch-act*/

		NPYLoader npy47_q;npy47_q.open(Q_table_datapath+Q_table_name[47]);npy47_q.init_tensor2(Q_table_block4r_con3,DataType:: S16);

		NPYLoader npy47_wt;npy47_wt.open(WT_buffer_datapath+WT_buffer_name[47]);npy47_wt.init_tensor2(WT_buffer_block4r_con3,DataType::U16);

		NPYLoader npy47_b;npy47_b.open(bias_datapath+bias_name[47]);npy47_b.init_tensor2(bias_block4r_con3,DataType:: S8);

		WeightsInfo weights_block4r_con3(false,1,1,512,false);

		const TensorShape out_shape_block4r_con3(7, 7, 512);

		out_block4r_con3.allocator()->init(TensorInfo(out_shape_block4r_con3, 1, DataType:: S8));

		out_block4r_act2.allocator()->init(TensorInfo(out_shape_block4r_con3, 1, DataType:: F32));

   		/*conv-batch-act		*/

		NPYLoader npy48_q;npy48_q.open(Q_table_datapath+Q_table_name[48]);npy48_q.init_tensor2(Q_table_block4r_con4,DataType:: S16);

		NPYLoader npy48_wt;npy48_wt.open(WT_buffer_datapath+WT_buffer_name[48]);npy48_wt.init_tensor2(WT_buffer_block4r_con4,DataType::U16);

		NPYLoader npy48_b;npy48_b.open(bias_datapath+bias_name[48]);npy48_b.init_tensor2(bias_block4r_con4,DataType:: S8);

		WeightsInfo weights_block4r_con4(false,3,3,512,false);

		const TensorShape out_shape_block4r_con4(7, 7, 512);

		out_block4r_con4.allocator()->init(TensorInfo(out_shape_block4r_con4, 1, DataType:: S8));

		out_block4r_act3.allocator()->init(TensorInfo(out_shape_block4r_con4, 1, DataType:: F32));

   		/*conv-batch		*/

		NPYLoader npy49_q;npy49_q.open(Q_table_datapath+Q_table_name[49]);npy49_q.init_tensor2(Q_table_block4r_con5,DataType:: S16);

		NPYLoader npy49_wt;npy49_wt.open(WT_buffer_datapath+WT_buffer_name[49]);npy49_wt.init_tensor2(WT_buffer_block4r_con5,DataType::U16);

		NPYLoader npy49_b;npy49_b.open(bias_datapath+bias_name[49]);npy49_b.init_tensor2(bias_block4r_con5,DataType:: S8);

		WeightsInfo weights_block4r_con5(false,1,1,2048,false);

		const TensorShape out_shape_block4r_con5(7, 7, 2048);

		out_block4r_con5.allocator()->init(TensorInfo(out_shape_block4r_con5, 1, DataType:: S8));

		/*add-act*/

		TensorShape out_shape_block4_1 = out_shape_block4r_con5;

		out_block4_add1.allocator()->init(TensorInfo(out_shape_block4_1, 1, DataType:: S8));

		out_block4_act1.allocator()->init(TensorInfo(out_shape_block4_1, 1, DataType:: F32));

  		 /*conv-batch-act*/

		NPYLoader npy50_q;npy50_q.open(Q_table_datapath+Q_table_name[50]);npy50_q.init_tensor2(Q_table_block4r_con6,DataType:: S16);

		NPYLoader npy50_wt;npy50_wt.open(WT_buffer_datapath+WT_buffer_name[50]);npy50_wt.init_tensor2(WT_buffer_block4r_con6,DataType::U16);

		NPYLoader npy50_b;npy50_b.open(bias_datapath+bias_name[50]);npy50_b.init_tensor2(bias_block4r_con6,DataType:: S8);

		WeightsInfo weights_block4r_con6(false,1,1,512,false);

		const TensorShape out_shape_block4r_con6(7, 7, 512);

		out_block4r_con6.allocator()->init(TensorInfo(out_shape_block4r_con6, 1, DataType:: S8));

		out_block4r_act4.allocator()->init(TensorInfo(out_shape_block4r_con6, 1, DataType:: F32));

   		/*conv-batch-act	*/

		NPYLoader npy51_q;npy51_q.open(Q_table_datapath+Q_table_name[51]);npy51_q.init_tensor2(Q_table_block4r_con7,DataType:: S16);

		NPYLoader npy51_wt;npy51_wt.open(WT_buffer_datapath+WT_buffer_name[51]);npy51_wt.init_tensor2(WT_buffer_block4r_con7,DataType::U16);

		NPYLoader npy51_b;npy51_b.open(bias_datapath+bias_name[51]);npy51_b.init_tensor2(bias_block4r_con7,DataType:: S8);

		WeightsInfo weights_block4r_con7(false,3,3,512,false);

		const TensorShape out_shape_block4r_con7(7, 7, 512);

		out_block4r_con7.allocator()->init(TensorInfo(out_shape_block4r_con7, 1, DataType:: S8));

		out_block4r_act5.allocator()->init(TensorInfo(out_shape_block4r_con7, 1, DataType:: F32));

   		/*conv-batch		*/

		NPYLoader npy52_q;npy52_q.open(Q_table_datapath+Q_table_name[52]);npy52_q.init_tensor2(Q_table_block4r_con8,DataType:: S16);

		NPYLoader npy52_wt;npy52_wt.open(WT_buffer_datapath+WT_buffer_name[52]);npy52_wt.init_tensor2(WT_buffer_block4r_con8,DataType::U16);

		NPYLoader npy52_b;npy52_b.open(bias_datapath+bias_name[52]);npy52_b.init_tensor2(bias_block4r_con8,DataType:: S8);

		WeightsInfo weights_block4r_con8(false,1,1,2048,false);

		const TensorShape out_shape_block4r_con8(7, 7, 2048);

		out_block4r_con8.allocator()->init(TensorInfo(out_shape_block4r_con8, 1, DataType:: S8));

		/*add-act	*/

		TensorShape out_shape_block4_2 = out_shape_block4r_con8;

		out_block4_add2.allocator()->init(TensorInfo(out_shape_block4_2, 1, DataType:: S8));

		out_block4_act2.allocator()->init(TensorInfo(out_shape_block4_2, 1, DataType:: F32));

		/* block end  */

       

		/*last pooling-conv-flatten-softmax*/

		TensorShape out_shape_pool1 = out_shape_block4_2;

		out_shape_pool1.set(0, out_shape_pool1.x() / 7);

		out_shape_pool1.set(1, out_shape_pool1.y() / 7);

		out_pool1.allocator()->init(TensorInfo(out_shape_pool1, 1, DataType:: F32));      

		/*NPYLoader npy100;npy100.open(data_path+"input_fc.npy");npy100.init_tensor2(input_fc,DataType:: S8);*/

		NPYLoader npy53_q;npy53_q.open(Q_table_datapath+Q_table_name[53]);npy53_q.init_tensor2(Q_table_con1,DataType:: S16);

		NPYLoader npy53_wt;npy53_wt.open(WT_buffer_datapath+WT_buffer_name[53]);npy53_wt.init_tensor2(WT_buffer_con1,DataType::U16);

		NPYLoader npy53_b;npy53_b.open(bias_datapath+bias_name[53]);npy53_b.init_tensor2(bias_con1,DataType:: S8);

		WeightsInfo weights_con1(false,32,4,1000,false);

		const TensorShape out_shape_con1(1, 1, 1000);

		out_con1.allocator()->init(TensorInfo(out_shape_con1, 1, DataType:: S8));

		const TensorShape out_shape_flatten(out_shape_con1.x()*out_shape_con1.y()*out_shape_con1.z(),0);                     

		out_flatten.allocator()->init(TensorInfo(out_shape_flatten, 1, DataType:: F32));

		const TensorShape out_shape_softmax(out_shape_flatten.x());

		out_softmax.allocator()->init(TensorInfo(out_shape_softmax, 1, DataType:: F32));

		/*last end*/



		/*configure start*/

		/*first start*/

		con0.configure(&src, &Q_table_con0,&WT_buffer_con0,&bias_con0, &out_con0, PadStrideInfo(2, 2, 3, 3),weights_con0,precision[0],1);

		lconv0sf.configure(&out_con0,&conv0sf);

		act0.configure(&conv0sf, &out_act0, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

		pool0.configure(&out_act0, &out_pool0, PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 1, 0, 1, DimensionRoundingType::FLOOR)));

		lpool0fs.configure(&out_pool0,&pool0fs);

		/*first end*/

		/*block start*/

		/* block1*/

		block1r_con0.configure(&pool0fs, &Q_table_block1r_con0, &WT_buffer_block1r_con0,&bias_block1r_con0, &out_block1r_con0, PadStrideInfo(1, 1, 0, 0),weights_block1r_con0,precision[1],1);

		lb1rconv0sf.configure(&out_block1r_con0,&b1rconv0sf);

		block1r_act0.configure(&b1rconv0sf, &out_block1r_act0, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

		lb1ract0fs.configure(&out_block1r_act0,&b1ract0fs);

		block1r_con1.configure(&b1ract0fs, &Q_table_block1r_con1, &WT_buffer_block1r_con1,&bias_block1r_con1, &out_block1r_con1, PadStrideInfo(1, 1, 1, 1),weights_block1r_con1,precision[2],1);

		lb1rconv1sf.configure(&out_block1r_con1,&b1rconv1sf);

		block1r_act1.configure(&b1rconv1sf, &out_block1r_act1, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

		lb1ract1fs.configure(&out_block1r_act1,&b1ract1fs);

		block1r_con2.configure(&b1ract1fs, &Q_table_block1r_con2, &WT_buffer_block1r_con2,&bias_block1r_con2, &out_block1r_con2, PadStrideInfo(1, 1, 0, 0),weights_block1r_con2,precision[3],1);

		block1l_con0.configure(&pool0fs, &Q_table_block1l_con0, &WT_buffer_block1l_con0,&bias_block1l_con0, &out_block1l_con0, PadStrideInfo(1, 1, 0, 0),weights_block1l_con0,precision[4],1);

		block1_add0.configure(&out_block1r_con2, &out_block1l_con0, &out_block1_add0,fp[0]);

		lb1add0sf.configure(&out_block1_add0,&b1add0sf);

		block1_act0.configure(&b1add0sf, &out_block1_act0, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

	 

		lb1act0fs.configure(&out_block1_act0, &b1act0fs);

		block1r_con3.configure(&b1act0fs, &Q_table_block1r_con3, &WT_buffer_block1r_con3,&bias_block1r_con3, &out_block1r_con3, PadStrideInfo(1, 1, 0, 0),weights_block1r_con3,precision[5],1);

		lb1rconv3sf.configure(&out_block1r_con3,&b1rconv3sf);

		block1r_act2.configure(&b1rconv3sf, &out_block1r_act2, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

		lb1ract2fs.configure(&out_block1r_act2,&b1ract2fs);

		block1r_con4.configure(&b1ract2fs, &Q_table_block1r_con4, &WT_buffer_block1r_con4,&bias_block1r_con4, &out_block1r_con4, PadStrideInfo(1, 1, 1, 1),weights_block1r_con4,precision[6],1);

		lb1rconv4sf.configure(&out_block1r_con4,&b1rconv4sf);

		block1r_act3.configure(&b1rconv4sf, &out_block1r_act3, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

		lb1ract3fs.configure(&out_block1r_act3,&b1ract3fs);

		block1r_con5.configure(&b1ract3fs, &Q_table_block1r_con5, &WT_buffer_block1r_con5,&bias_block1r_con5, &out_block1r_con5, PadStrideInfo(1, 1, 0, 0),weights_block1r_con5,precision[7],1);

		block1_add1.configure(&out_block1r_con5, &b1act0fs, &out_block1_add1,fp[1]);

		lb1add1sf.configure(&out_block1_add1,&b1add1sf);

		block1_act1.configure(&b1add1sf, &out_block1_act1, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

		

		lb1act1fs.configure(&out_block1_act1,&b1act1fs);

		block1r_con6.configure(&b1act1fs, &Q_table_block1r_con6, &WT_buffer_block1r_con6,&bias_block1r_con6, &out_block1r_con6, PadStrideInfo(1, 1, 0, 0),weights_block1r_con6,precision[8],1);

		lb1rconv6sf.configure(&out_block1r_con6,&b1rconv6sf);

		block1r_act4.configure(&b1rconv6sf, &out_block1r_act4, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

		lb1ract4fs.configure(&out_block1r_act4,&b1ract4fs);

		block1r_con7.configure(&b1ract4fs, &Q_table_block1r_con7, &WT_buffer_block1r_con7,&bias_block1r_con7, &out_block1r_con7, PadStrideInfo(1, 1, 1, 1),weights_block1r_con7,precision[9],1);

		lb1rconv7sf.configure(&out_block1r_con7,&b1rconv7sf);

		block1r_act5.configure(&b1rconv7sf, &out_block1r_act5, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

		lb1ract5fs.configure(&out_block1r_act5,&b1ract5fs);

		block1r_con8.configure(&b1ract5fs, &Q_table_block1r_con8, &WT_buffer_block1r_con8,&bias_block1r_con8, &out_block1r_con8, PadStrideInfo(1, 1, 0, 0),weights_block1r_con8,precision[10],1);

		block1_add2.configure(&out_block1r_con8, &b1act1fs, &out_block1_add2,fp[2]);

		lb1add2sf.configure(&out_block1_add2,&b1add2sf);

		block1_act2.configure(&b1add2sf, &out_block1_act2, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

		/*end block1*/

		/*block2*/

		lb1act2fs.configure(&out_block1_act2,&b1act2fs);

		block2r_con0.configure(&b1act2fs, &Q_table_block2r_con0, &WT_buffer_block2r_con0,&bias_block2r_con0, &out_block2r_con0, PadStrideInfo(2, 2, 0, 0),weights_block2r_con0,precision[11],1);

		lb2rconv0sf.configure(&out_block2r_con0,&b2rconv0sf);

		block2r_act0.configure(&b2rconv0sf, &out_block2r_act0, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

		lb2ract0fs.configure(&out_block2r_act0,&b2ract0fs);

		block2r_con1.configure(&b2ract0fs, &Q_table_block2r_con1, &WT_buffer_block2r_con1,&bias_block2r_con1, &out_block2r_con1, PadStrideInfo(1, 1, 1, 1),weights_block2r_con1,precision[12],1);

		lb2rconv1sf.configure(&out_block2r_con1,&b2rconv1sf);

		block2r_act1.configure(&b2rconv1sf, &out_block2r_act1, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

		lb2ract1fs.configure(&out_block2r_act1, &b2ract1fs);

		block2r_con2.configure(&b2ract1fs, &Q_table_block2r_con2, &WT_buffer_block2r_con2,&bias_block2r_con2, &out_block2r_con2, PadStrideInfo(1, 1, 0, 0),weights_block2r_con2,precision[13],1);

		block2l_con0.configure(&b1act2fs, &Q_table_block2l_con0, &WT_buffer_block2l_con0,&bias_block2l_con0, &out_block2l_con0, PadStrideInfo(2, 2, 0, 0),weights_block2l_con0,precision[14],1);

		block2_add0.configure(&out_block2r_con2, &out_block2l_con0, &out_block2_add0, fp[3]);

		lb2add0sf.configure(&out_block2_add0,&b2add0sf);

		block2_act0.configure(&b2add0sf, &out_block2_act0, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

		

		lb2act0fs.configure(&out_block2_act0,&b2act0fs);

		block2r_con3.configure(&b2act0fs, &Q_table_block2r_con3, &WT_buffer_block2r_con3,&bias_block2r_con3, &out_block2r_con3, PadStrideInfo(1, 1, 0, 0),weights_block2r_con3,precision[15],1);

		lb2rconv3sf.configure(&out_block2r_con3,&b2rconv3sf);

		block2r_act2.configure(&b2rconv3sf, &out_block2r_act2, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

		lb2ract2fs.configure(&out_block2r_act2,&b2ract2fs);

		block2r_con4.configure(&b2ract2fs, &Q_table_block2r_con4, &WT_buffer_block2r_con4,&bias_block2r_con4, &out_block2r_con4, PadStrideInfo(1, 1, 1, 1),weights_block2r_con4,precision[16],1);

		lb2rconv4sf.configure(&out_block2r_con4,&b2rconv4sf);

		block2r_act3.configure(&b2rconv4sf, &out_block2r_act3, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

		lb2ract3fs.configure(&out_block2r_act3,&b2ract3fs);

		block2r_con5.configure(&b2ract3fs, &Q_table_block2r_con5, &WT_buffer_block2r_con5,&bias_block2r_con5, &out_block2r_con5, PadStrideInfo(1, 1, 0, 0),weights_block2r_con5,precision[17],1);

		block2_add1.configure(&out_block2r_con5, &b2act0fs, &out_block2_add1, fp[4]);

		lb2add1sf.configure(&out_block2_add1,&b2add1sf);

		block2_act1.configure(&b2add1sf, &out_block2_act1, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

		

		lb2act1fs.configure(&out_block2_act1,&b2act1fs);

		block2r_con6.configure(&b2act1fs, &Q_table_block2r_con6, &WT_buffer_block2r_con6,&bias_block2r_con6, &out_block2r_con6, PadStrideInfo(1, 1, 0, 0),weights_block2r_con6,precision[18],1);

		lb2rconv6sf.configure(&out_block2r_con6,&b2rconv6sf);

		block2r_act4.configure(&b2rconv6sf, &out_block2r_act4, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

		lb2ract4fs.configure(&out_block2r_act4,&b2ract4fs);

		block2r_con7.configure(&b2ract4fs, &Q_table_block2r_con7, &WT_buffer_block2r_con7,&bias_block2r_con7, &out_block2r_con7, PadStrideInfo(1, 1, 1, 1),weights_block2r_con7,precision[19],1);

		lb2rconv7sf.configure(&out_block2r_con7,&b2rconv7sf);

		block2r_act5.configure(&b2rconv7sf, &out_block2r_act5, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

		lb2ract5fs.configure(&out_block2r_act5,&b2ract5fs);

		block2r_con8.configure(&b2ract5fs, &Q_table_block2r_con8, &WT_buffer_block2r_con8,&bias_block2r_con8, &out_block2r_con8, PadStrideInfo(1, 1, 0, 0),weights_block2r_con8,precision[20],1);

		block2_add2.configure(&out_block2r_con8, &b2act1fs, &out_block2_add2, fp[5]);

		lb2add2sf.configure(&out_block2_add2,&b2add2sf);

		block2_act2.configure(&b2add2sf, &out_block2_act2, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

		

		lb2act2fs.configure(&out_block2_act2,&b2act2fs);

		block2r_con9.configure(&b2act2fs, &Q_table_block2r_con9, &WT_buffer_block2r_con9,&bias_block2r_con9, &out_block2r_con9, PadStrideInfo(1, 1, 0, 0),weights_block2r_con9,precision[21],1);

		lb2rconv9sf.configure(&out_block2r_con9,&b2rconv9sf);

		block2r_act6.configure(&b2rconv9sf, &out_block2r_act6, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

		lb2ract6fs.configure(&out_block2r_act6,&b2ract6fs);

		block2r_con10.configure(&b2ract6fs, &Q_table_block2r_con10, &WT_buffer_block2r_con10,&bias_block2r_con10, &out_block2r_con10, PadStrideInfo(1, 1, 1, 1),weights_block2r_con10,precision[22],1);

		lb2rconv10sf.configure(&out_block2r_con10,&b2rconv10sf);

		block2r_act7.configure(&b2rconv10sf, &out_block2r_act7, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

		lb2ract7fs.configure(&out_block2r_act7,&b2ract7fs);

		block2r_con11.configure(&b2ract7fs, &Q_table_block2r_con11, &WT_buffer_block2r_con11,&bias_block2r_con11, &out_block2r_con11, PadStrideInfo(1, 1, 0, 0),weights_block2r_con11,precision[23],1);

		block2_add3.configure(&out_block2r_con11, &b2act2fs, &out_block2_add3, fp[6]);

		lb2add3sf.configure(&out_block2_add3,&b2add3sf);

		block2_act3.configure(&b2add3sf, &out_block2_act3, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

		

		/*end block2*/

		/*block3*/

		lb2act3fs.configure(&out_block2_act3,&b2act3fs);

		block3r_con0.configure(&b2act3fs, &Q_table_block3r_con0, &WT_buffer_block3r_con0,&bias_block3r_con0, &out_block3r_con0, PadStrideInfo(2, 2, 0, 0),weights_block3r_con0,precision[24],1);

		lb3rconv0sf.configure(&out_block3r_con0,&b3rconv0sf);

		block3r_act0.configure(&b3rconv0sf, &out_block3r_act0, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

		lb3ract0fs.configure(&out_block3r_act0,&b3ract0fs);

		block3r_con1.configure(&b3ract0fs, &Q_table_block3r_con1,  &WT_buffer_block3r_con1,&bias_block3r_con1, &out_block3r_con1, PadStrideInfo(1, 1, 1, 1),weights_block3r_con1,precision[25],1);

		lb3rconv1sf.configure(&out_block3r_con1,&b3rconv1sf);

		block3r_act1.configure(&b3rconv1sf, &out_block3r_act1, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

		lb3ract1fs.configure(&out_block3r_act1,&b3ract1fs);

		block3r_con2.configure(&b3ract1fs, &Q_table_block3r_con2,  &WT_buffer_block3r_con2,&bias_block3r_con2, &out_block3r_con2, PadStrideInfo(1, 1, 0, 0),weights_block3r_con2,precision[26],1);

		block3l_con0.configure(&b2act3fs, &Q_table_block3l_con0,  &WT_buffer_block3l_con0,&bias_block3l_con0, &out_block3l_con0, PadStrideInfo(2, 2, 0, 0),weights_block3l_con0,precision[27],1);

		block3_add0.configure(&out_block3r_con2, &out_block3l_con0, &out_block3_add0, fp[7]);

		lb3add0sf.configure(&out_block3_add0,&b3add0sf);

		block3_act0.configure(&b3add0sf, &out_block3_act0, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));



		lb3act0fs.configure(&out_block3_act0,&b3act0fs);

		block3r_con3.configure(&b3act0fs, &Q_table_block3r_con3,  &WT_buffer_block3r_con3,&bias_block3r_con3, &out_block3r_con3, PadStrideInfo(1, 1, 0, 0),weights_block3r_con3,precision[28],1);

		lb3rconv3sf.configure(&out_block3r_con3,&b3rconv3sf);

		block3r_act2.configure(&b3rconv3sf, &out_block3r_act2, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

		lb3ract2fs.configure(&out_block3r_act2,&b3ract2fs);

		block3r_con4.configure(&b3ract2fs, &Q_table_block3r_con4,  &WT_buffer_block3r_con4,&bias_block3r_con4, &out_block3r_con4, PadStrideInfo(1, 1, 1, 1),weights_block3r_con4,precision[29],1);

		lb3rconv4sf.configure(&out_block3r_con4,&b3rconv4sf);

		block3r_act3.configure(&b3rconv4sf, &out_block3r_act3, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

		lb3ract3fs.configure(&out_block3r_act3,&b3ract3fs);

		block3r_con5.configure(&b3ract3fs, &Q_table_block3r_con5,  &WT_buffer_block3r_con5,&bias_block3r_con5, &out_block3r_con5, PadStrideInfo(1, 1, 0, 0),weights_block3r_con5,precision[30],1);

		block3_add1.configure(&out_block3r_con5, &b3act0fs, &out_block3_add1, fp[8]);

		lb3add1sf.configure(&out_block3_add1,&b3add1sf);

		block3_act1.configure(&b3add1sf, &out_block3_act1, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        

		lb3act1fs.configure(&out_block3_act1,&b3act1fs);

		block3r_con6.configure(&b3act1fs, &Q_table_block3r_con6,  &WT_buffer_block3r_con6,&bias_block3r_con6, &out_block3r_con6, PadStrideInfo(1, 1, 0, 0),weights_block3r_con6,precision[31],1);

		lb3rconv6sf.configure(&out_block3r_con6,&b3rconv6sf);

		block3r_act4.configure(&b3rconv6sf, &out_block3r_act4, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

		lb3ract4fs.configure(&out_block3r_act4,&b3ract4fs);

		block3r_con7.configure(&b3ract4fs, &Q_table_block3r_con7,  &WT_buffer_block3r_con7,&bias_block3r_con7, &out_block3r_con7, PadStrideInfo(1, 1, 1, 1),weights_block3r_con7,precision[32],1);

		lb3rconv7sf.configure(&out_block3r_con7,&b3rconv7sf);

		block3r_act5.configure(&b3rconv7sf, &out_block3r_act5, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

		lb3ract5fs.configure(&out_block3r_act5,&b3ract5fs);

		block3r_con8.configure(&b3ract5fs, &Q_table_block3r_con8,  &WT_buffer_block3r_con8,&bias_block3r_con8, &out_block3r_con8, PadStrideInfo(1, 1, 0, 0),weights_block3r_con8,precision[33],1);

		block3_add2.configure(&out_block3r_con8, &b3act1fs, &out_block3_add2, fp[9]);

		lb3add2sf.configure(&out_block3_add2,&b3add2sf);

		block3_act2.configure(&b3add2sf, &out_block3_act2, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

      

	  	lb3act2fs.configure(&out_block3_act2,&b3act2fs);

		block3r_con9.configure(&b3act2fs, &Q_table_block3r_con9,  &WT_buffer_block3r_con9,&bias_block3r_con9, &out_block3r_con9, PadStrideInfo(1, 1, 0, 0),weights_block3r_con9,precision[34],1);

		lb3rconv9sf.configure(&out_block3r_con9,&b3rconv9sf);

		block3r_act6.configure(&b3rconv9sf, &out_block3r_act6, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

		lb3ract6fs.configure(&out_block3r_act6,&b3ract6fs);

		block3r_con10.configure(&b3ract6fs, &Q_table_block3r_con10,  &WT_buffer_block3r_con10,&bias_block3r_con10, &out_block3r_con10, PadStrideInfo(1, 1, 1, 1),weights_block3r_con10,precision[35],1);

		lb3rconv10sf.configure(&out_block3r_con10,&b3rconv10sf);

		block3r_act7.configure(&b3rconv10sf, &out_block3r_act7, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

		lb3ract7fs.configure(&out_block3r_act7,&b3ract7fs);

		block3r_con11.configure(&b3ract7fs, &Q_table_block3r_con11,  &WT_buffer_block3r_con11,&bias_block3r_con11, &out_block3r_con11, PadStrideInfo(1, 1, 0, 0),weights_block3r_con11,precision[36],1);

		block3_add3.configure(&out_block3r_con11, &b3act2fs, &out_block3_add3, fp[10]);

		lb3add3sf.configure(&out_block3_add3,&b3add3sf);

		block3_act3.configure(&b3add3sf, &out_block3_act3, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        

		lb3act3fs.configure(&out_block3_act3,&b3act3fs);

		block3r_con12.configure(&b3act3fs, &Q_table_block3r_con12,  &WT_buffer_block3r_con12,&bias_block3r_con12, &out_block3r_con12, PadStrideInfo(1, 1, 0, 0),weights_block3r_con12,precision[37],1);

		lb3rconv12sf.configure(&out_block3r_con12,&b3rconv12sf);

		block3r_act8.configure(&b3rconv12sf, &out_block3r_act8, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

		lb3ract8fs.configure(&out_block3r_act8,&b3ract8fs);

		block3r_con13.configure(&b3ract8fs, &Q_table_block3r_con13,  &WT_buffer_block3r_con13,&bias_block3r_con13, &out_block3r_con13, PadStrideInfo(1, 1, 1, 1),weights_block3r_con13,precision[38],1);

		lb3rconv13sf.configure(&out_block3r_con13,&b3rconv13sf);

		block3r_act9.configure(&b3rconv13sf, &out_block3r_act9, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

		lb3ract9fs.configure(&out_block3r_act9,&b3ract9fs);

		block3r_con14.configure(&b3ract9fs, &Q_table_block3r_con14,  &WT_buffer_block3r_con14,&bias_block3r_con14, &out_block3r_con14, PadStrideInfo(1, 1, 0, 0),weights_block3r_con14,precision[39],1);

		block3_add4.configure(&out_block3r_con14, &b3act3fs, &out_block3_add4, fp[11]);

		lb3add4sf.configure(&out_block3_add4,&b3add4sf);

		block3_act4.configure(&b3add4sf, &out_block3_act4, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

      

	  	lb3act4fs.configure(&out_block3_act4,&b3act4fs);

		block3r_con15.configure(&b3act4fs, &Q_table_block3r_con15,  &WT_buffer_block3r_con15,&bias_block3r_con15, &out_block3r_con15, PadStrideInfo(1, 1, 0, 0),weights_block3r_con15,precision[40],1);

		lb3rconv15sf.configure(&out_block3r_con15,&b3rconv15sf);

		block3r_act10.configure(&b3rconv15sf, &out_block3r_act10, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

		lb3ract10fs.configure(&out_block3r_act10,&b3ract10fs);

		block3r_con16.configure(&b3ract10fs, &Q_table_block3r_con16,  &WT_buffer_block3r_con16,&bias_block3r_con16, &out_block3r_con16, PadStrideInfo(1, 1, 1, 1),weights_block3r_con16,precision[41],1);

		lb3rconv16sf.configure(&out_block3r_con16,&b3rconv16sf);

		block3r_act11.configure(&b3rconv16sf, &out_block3r_act11, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

		lb3ract11fs.configure(&out_block3r_act11,&b3ract11fs);

		block3r_con17.configure(&b3ract11fs, &Q_table_block3r_con17,  &WT_buffer_block3r_con17,&bias_block3r_con17, &out_block3r_con17, PadStrideInfo(1, 1, 0, 0),weights_block3r_con17,precision[42],1);

		block3_add5.configure(&out_block3r_con17, &b3act4fs, &out_block3_add5, fp[12]);

		lb3add5sf.configure(&out_block3_add5,&b3add5sf);

		block3_act5.configure(&b3add5sf, &out_block3_act5, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

		

				

		/*end block3*/

		/*block4*/

		lb3act5fs.configure(&out_block3_act5,&b3act5fs);

		block4r_con0.configure(&b3act5fs, &Q_table_block4r_con0, &WT_buffer_block4r_con0,&bias_block4r_con0, &out_block4r_con0, PadStrideInfo(2, 2, 0, 0),weights_block4r_con0,precision[43],1);

		lb4rconv0sf.configure(&out_block4r_con0,&b4rconv0sf);

		block4r_act0.configure(&b4rconv0sf, &out_block4r_act0, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

		lb4ract0fs.configure(&out_block4r_act0,&b4ract0fs);

		block4r_con1.configure(&b4ract0fs, &Q_table_block4r_con1, &WT_buffer_block4r_con1,&bias_block4r_con1, &out_block4r_con1, PadStrideInfo(1, 1, 1, 1),weights_block4r_con1,precision[44],1);

		lb4rconv1sf.configure(&out_block4r_con1,&b4rconv1sf);

		block4r_act1.configure(&b4rconv1sf, &out_block4r_act1, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

		lb4ract1fs.configure(&out_block4r_act1,&b4ract1fs);

		block4r_con2.configure(&b4ract1fs, &Q_table_block4r_con2, &WT_buffer_block4r_con2,&bias_block4r_con2, &out_block4r_con2, PadStrideInfo(1, 1, 0, 0),weights_block4r_con2,precision[45],1);

		block4l_con0.configure(&b3act5fs, &Q_table_block4l_con0, &WT_buffer_block4l_con0,&bias_block4l_con0, &out_block4l_con0, PadStrideInfo(2, 2, 0, 0),weights_block4l_con0,precision[46],1);

		block4_add0.configure(&out_block4r_con2, &out_block4l_con0, &out_block4_add0, fp[13]);

		lb4add0sf.configure(&out_block4_add0,&b4add0sf);

		block4_act0.configure(&b4add0sf, &out_block4_act0, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

     

	 	lb4act0fs.configure(&out_block4_act0,&b4act0fs);

		block4r_con3.configure(&b4act0fs, &Q_table_block4r_con3, &WT_buffer_block4r_con3,&bias_block4r_con3, &out_block4r_con3, PadStrideInfo(1, 1, 0, 0),weights_block4r_con3,precision[47],1);

		lb4rconv3sf.configure(&out_block4r_con3,&b4rconv3sf);

		block4r_act2.configure(&b4rconv3sf, &out_block4r_act2, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

		lb4ract2fs.configure(&out_block4r_act2,&b4ract2fs);

		block4r_con4.configure(&b4ract2fs, &Q_table_block4r_con4, &WT_buffer_block4r_con4,&bias_block4r_con4, &out_block4r_con4, PadStrideInfo(1, 1, 1, 1),weights_block4r_con4,precision[48],1);

		lb4rconv4sf.configure(&out_block4r_con4,&b4rconv4sf);

		block4r_act3.configure(&b4rconv4sf, &out_block4r_act3, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

		lb4ract3fs.configure(&out_block4r_act3,&b4ract3fs);

		block4r_con5.configure(&b4ract3fs, &Q_table_block4r_con5, &WT_buffer_block4r_con5,&bias_block4r_con5, &out_block4r_con5, PadStrideInfo(1, 1, 0, 0),weights_block4r_con5,precision[49],1);

		block4_add1.configure(&out_block4r_con5, &b4act0fs, &out_block4_add1, fp[14]);

		lb4add1sf.configure(&out_block4_add1,&b4add1sf);

		block4_act1.configure(&b4add1sf, &out_block4_act1, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

      

	  	lb4act1fs.configure(&out_block4_act1,&b4act1fs);

		block4r_con6.configure(&b4act1fs, &Q_table_block4r_con6, &WT_buffer_block4r_con6,&bias_block4r_con6, &out_block4r_con6, PadStrideInfo(1, 1, 0, 0),weights_block4r_con6,precision[50],1);

		lb4rconv6sf.configure(&out_block4r_con6,&b4rconv6sf);

		block4r_act4.configure(&b4rconv6sf, &out_block4r_act4, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

		lb4ract4fs.configure(&out_block4r_act4,&b4ract4fs);

		block4r_con7.configure(&b4ract4fs, &Q_table_block4r_con7, &WT_buffer_block4r_con7,&bias_block4r_con7, &out_block4r_con7, PadStrideInfo(1, 1, 1, 1),weights_block4r_con7,precision[51],1);

		lb4rconv7sf.configure(&out_block4r_con7,&b4rconv7sf);

		block4r_act5.configure(&b4rconv7sf, &out_block4r_act5, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

		lb4ract5fs.configure(&out_block4r_act5,&b4ract5fs);

		block4r_con8.configure(&b4ract5fs, &Q_table_block4r_con8, &WT_buffer_block4r_con8,&bias_block4r_con8, &out_block4r_con8, PadStrideInfo(1, 1, 0, 0),weights_block4r_con8,precision[52],1);

		block4_add2.configure(&out_block4r_con8, &b4act1fs, &out_block4_add2,fp[15]);

		lb4add2sf.configure(&out_block4_add2,&b4add2sf);

		block4_act2.configure(&b4add2sf, &out_block4_act2, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

		/*end block4*/

		/*block end*/

		/*last start*/

	    pool1.configure(&out_block4_act2, &out_pool1, PoolingLayerInfo(PoolingType::AVG, 7, PadStrideInfo(1, 1, 0, 0, 0, 0, DimensionRoundingType::FLOOR)));

		lpool1fs.configure(&out_pool1,&pool1fs);

		TensorShape shape(32,4,16);

		resize.configure(&pool1fs, &out_resize, shape);

		con1.configure(&out_resize, &Q_table_con1, &WT_buffer_con1,&bias_con1, &out_con1, PadStrideInfo(1, 1, 0, 0),weights_con1,precision[53],1);

		lconv1sf.configure(&out_con1,&conv1sf);

		flatten.configure(&conv1sf, &out_flatten);

		softmax.configure(&out_flatten, &out_softmax);

		/*last end*/

		/*configure end*/





		/*allocate start*/

		/*first allocate*/

	    out_con0.allocator()->allocate(); 

	    out_act0.allocator()->allocate(); out_pool0.allocator()->allocate();

		/*first allocate end*/

		/*block allocate*/

		/*block1*/

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

		/*end block1*/

		/*block2*/

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

		/*end block2*/

		/*block3*/

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



		/*end block3*/

		/*block4*/

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

		/*end block4*/

		/*block allocate end*/

		/*last allocate*/

        out_pool1.allocator()->allocate(); out_con1.allocator()->allocate(); out_flatten.allocator()->allocate(); out_softmax.allocator()->allocate();

		/*last allocate end*/

        src.allocator()->allocate(); Q_table_con0.allocator()->allocate(); WT_buffer_con0.allocator()->allocate();bias_con0.allocator()->allocate();

		

		Q_table_block1r_con0.allocator()->allocate(); Q_table_block1r_con1.allocator()->allocate(); Q_table_block1r_con2.allocator()->allocate(); 

		Q_table_block1r_con3.allocator()->allocate(); Q_table_block1r_con4.allocator()->allocate(); Q_table_block1r_con5.allocator()->allocate(); 

		Q_table_block1r_con6.allocator()->allocate(); Q_table_block1r_con7.allocator()->allocate(); Q_table_block1r_con8.allocator()->allocate(); 

		Q_table_block1l_con0.allocator()->allocate();

		WT_buffer_block1r_con0.allocator()->allocate(); WT_buffer_block1r_con1.allocator()->allocate(); WT_buffer_block1r_con2.allocator()->allocate(); 

		WT_buffer_block1r_con3.allocator()->allocate(); WT_buffer_block1r_con4.allocator()->allocate(); WT_buffer_block1r_con5.allocator()->allocate(); 

		WT_buffer_block1r_con6.allocator()->allocate(); WT_buffer_block1r_con7.allocator()->allocate(); WT_buffer_block1r_con8.allocator()->allocate(); 

		WT_buffer_block1l_con0.allocator()->allocate();

		

		Q_table_block2r_con0.allocator()->allocate(); Q_table_block2r_con1.allocator()->allocate(); Q_table_block2r_con2.allocator()->allocate();

		Q_table_block2r_con3.allocator()->allocate(); Q_table_block2r_con4.allocator()->allocate(); Q_table_block2r_con5.allocator()->allocate();

		Q_table_block2r_con6.allocator()->allocate(); Q_table_block2r_con7.allocator()->allocate(); Q_table_block2r_con8.allocator()->allocate();

		Q_table_block2r_con9.allocator()->allocate(); Q_table_block2r_con10.allocator()->allocate(); Q_table_block2r_con11.allocator()->allocate();

		Q_table_block2l_con0.allocator()->allocate(); 

		WT_buffer_block2r_con0.allocator()->allocate(); WT_buffer_block2r_con1.allocator()->allocate(); WT_buffer_block2r_con2.allocator()->allocate();

		WT_buffer_block2r_con3.allocator()->allocate(); WT_buffer_block2r_con4.allocator()->allocate(); WT_buffer_block2r_con5.allocator()->allocate();

		WT_buffer_block2r_con6.allocator()->allocate(); WT_buffer_block2r_con7.allocator()->allocate(); WT_buffer_block2r_con8.allocator()->allocate();

		WT_buffer_block2r_con9.allocator()->allocate(); WT_buffer_block2r_con10.allocator()->allocate(); WT_buffer_block2r_con11.allocator()->allocate();

		WT_buffer_block2l_con0.allocator()->allocate(); 

		

		Q_table_block3r_con0.allocator()->allocate(); Q_table_block3r_con1.allocator()->allocate(); Q_table_block3r_con2.allocator()->allocate();

		Q_table_block3r_con3.allocator()->allocate(); Q_table_block3r_con4.allocator()->allocate(); Q_table_block3r_con5.allocator()->allocate();

		Q_table_block3r_con6.allocator()->allocate(); Q_table_block3r_con7.allocator()->allocate(); Q_table_block3r_con8.allocator()->allocate();

		Q_table_block3r_con9.allocator()->allocate(); Q_table_block3r_con10.allocator()->allocate(); Q_table_block3r_con11.allocator()->allocate();

		Q_table_block3r_con12.allocator()->allocate(); Q_table_block3r_con13.allocator()->allocate(); Q_table_block3r_con14.allocator()->allocate();

		Q_table_block3r_con15.allocator()->allocate(); Q_table_block3r_con16.allocator()->allocate(); Q_table_block3r_con17.allocator()->allocate();

		Q_table_block3l_con0.allocator()->allocate(); 

		WT_buffer_block3r_con0.allocator()->allocate(); WT_buffer_block3r_con1.allocator()->allocate(); WT_buffer_block3r_con2.allocator()->allocate();

		WT_buffer_block3r_con3.allocator()->allocate(); WT_buffer_block3r_con4.allocator()->allocate(); WT_buffer_block3r_con5.allocator()->allocate();

		WT_buffer_block3r_con6.allocator()->allocate(); WT_buffer_block3r_con7.allocator()->allocate(); WT_buffer_block3r_con8.allocator()->allocate();

		WT_buffer_block3r_con9.allocator()->allocate(); WT_buffer_block3r_con10.allocator()->allocate(); WT_buffer_block3r_con11.allocator()->allocate();

		WT_buffer_block3r_con12.allocator()->allocate(); WT_buffer_block3r_con13.allocator()->allocate(); WT_buffer_block3r_con14.allocator()->allocate();

		WT_buffer_block3r_con15.allocator()->allocate(); WT_buffer_block3r_con16.allocator()->allocate(); WT_buffer_block3r_con17.allocator()->allocate();

		WT_buffer_block3l_con0.allocator()->allocate(); 

		

		Q_table_block4r_con0.allocator()->allocate(); Q_table_block4r_con1.allocator()->allocate(); Q_table_block4r_con2.allocator()->allocate();

		Q_table_block4r_con3.allocator()->allocate(); Q_table_block4r_con4.allocator()->allocate(); Q_table_block4r_con5.allocator()->allocate();

		Q_table_block4r_con6.allocator()->allocate(); Q_table_block4r_con7.allocator()->allocate(); Q_table_block4r_con8.allocator()->allocate();

		Q_table_block4l_con0.allocator()->allocate(); 

		WT_buffer_block4r_con0.allocator()->allocate(); WT_buffer_block4r_con1.allocator()->allocate(); WT_buffer_block4r_con2.allocator()->allocate();

		WT_buffer_block4r_con3.allocator()->allocate(); WT_buffer_block4r_con4.allocator()->allocate(); WT_buffer_block4r_con5.allocator()->allocate();

		WT_buffer_block4r_con6.allocator()->allocate(); WT_buffer_block4r_con7.allocator()->allocate(); WT_buffer_block4r_con8.allocator()->allocate();

		WT_buffer_block4l_con0.allocator()->allocate(); 

		

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



		/*input_fc.allocator()->allocate();*/

		Q_table_con1.allocator()->allocate(); WT_buffer_con1.allocator()->allocate(); bias_con1.allocator()->allocate();



		/*type change tensor allocate*/

		 conv0sf.allocator()->allocate();  pool0fs.allocator()->allocate();



		 b1rconv0sf.allocator()->allocate();  b1ract0fs.allocator()->allocate();  b1rconv1sf.allocator()->allocate();  b1ract1fs.allocator()->allocate(); b1add0sf.allocator()->allocate();

		 b1act0fs.allocator()->allocate();  b1rconv3sf.allocator()->allocate();  b1ract2fs.allocator()->allocate();  b1rconv4sf.allocator()->allocate();  b1ract3fs.allocator()->allocate();  b1add1sf.allocator()->allocate();

		 b1act1fs.allocator()->allocate();  b1rconv6sf.allocator()->allocate();  b1ract4fs.allocator()->allocate();  b1rconv7sf.allocator()->allocate();  b1ract5fs.allocator()->allocate(); b1add2sf.allocator()->allocate();



		 b1act2fs.allocator()->allocate(); b2rconv0sf.allocator()->allocate();  b2ract0fs.allocator()->allocate();  b2rconv1sf.allocator()->allocate();  b2ract1fs.allocator()->allocate();  b2add0sf.allocator()->allocate();

		 b2act0fs.allocator()->allocate();  b2rconv3sf.allocator()->allocate();  b2ract2fs.allocator()->allocate();  b2rconv4sf.allocator()->allocate();  b2ract3fs.allocator()->allocate();  b2add1sf.allocator()->allocate();

		 b2act1fs.allocator()->allocate();  b2rconv6sf.allocator()->allocate();  b2ract4fs.allocator()->allocate();  b2rconv7sf.allocator()->allocate();  b2ract5fs.allocator()->allocate();  b2add2sf.allocator()->allocate();

		 b2act2fs.allocator()->allocate();  b2rconv9sf.allocator()->allocate();  b2ract6fs.allocator()->allocate();  b2rconv10sf.allocator()->allocate();  b2ract7fs.allocator()->allocate(); b2add3sf.allocator()->allocate();



		 b2act3fs.allocator()->allocate(); b3rconv0sf.allocator()->allocate();  b3ract0fs.allocator()->allocate();  b3rconv1sf.allocator()->allocate();  b3ract1fs.allocator()->allocate();  b3add0sf.allocator()->allocate();

		 b3act0fs.allocator()->allocate();  b3rconv3sf.allocator()->allocate();  b3ract2fs.allocator()->allocate();  b3rconv4sf.allocator()->allocate();  b3ract3fs.allocator()->allocate();  b3add1sf.allocator()->allocate();

		 b3act1fs.allocator()->allocate();  b3rconv6sf.allocator()->allocate();  b3ract4fs.allocator()->allocate();  b3rconv7sf.allocator()->allocate();  b3ract5fs.allocator()->allocate(); b3add2sf.allocator()->allocate();

		 b3act2fs.allocator()->allocate();  b3rconv9sf.allocator()->allocate();  b3ract6fs.allocator()->allocate();  b3rconv10sf.allocator()->allocate();  b3ract7fs.allocator()->allocate();  b3add3sf.allocator()->allocate();

		 b3act3fs.allocator()->allocate();  b3rconv12sf.allocator()->allocate();  b3ract8fs.allocator()->allocate();  b3rconv13sf.allocator()->allocate();  b3ract9fs.allocator()->allocate();  b3add4sf.allocator()->allocate();

		 b3act4fs.allocator()->allocate();  b3rconv15sf.allocator()->allocate();  b3ract10fs.allocator()->allocate();  b3rconv16sf.allocator()->allocate();  b3ract11fs.allocator()->allocate();  b3add5sf.allocator()->allocate();



		 b3act5fs.allocator()->allocate(); b4rconv0sf.allocator()->allocate();  b4ract0fs.allocator()->allocate();  b4rconv1sf.allocator()->allocate();  b4ract1fs.allocator()->allocate();  b4add0sf.allocator()->allocate();

		 b4act0fs.allocator()->allocate();  b4rconv3sf.allocator()->allocate();  b4ract2fs.allocator()->allocate();  b4rconv4sf.allocator()->allocate();  b4ract3fs.allocator()->allocate();  b4add1sf.allocator()->allocate();

		 b4act1fs.allocator()->allocate();  b4rconv6sf.allocator()->allocate();  b4ract4fs.allocator()->allocate();  b4rconv7sf.allocator()->allocate();  b4ract5fs.allocator()->allocate(); b4add2sf.allocator()->allocate();



		 pool1fs.allocator()->allocate();  conv1sf.allocator()->allocate();



		 out_resize.allocator()->allocate();







		/*Fill tensor*/

		npy_input.fill_tensor2(src);



            npy0_q.fill_tensor2(Q_table_con0);

            npy1_q.fill_tensor2(Q_table_block1r_con0);

            npy2_q.fill_tensor2(Q_table_block1r_con1);

            npy3_q.fill_tensor2(Q_table_block1r_con2);

            npy4_q.fill_tensor2(Q_table_block1l_con0);

            npy5_q.fill_tensor2(Q_table_block1r_con3);

            npy6_q.fill_tensor2(Q_table_block1r_con4);

            npy7_q.fill_tensor2(Q_table_block1r_con5);

            npy8_q.fill_tensor2(Q_table_block1r_con6);

            npy9_q.fill_tensor2(Q_table_block1r_con7);

            npy10_q.fill_tensor2(Q_table_block1r_con8);

            npy11_q.fill_tensor2(Q_table_block2r_con0);

            npy12_q.fill_tensor2(Q_table_block2r_con1);

            npy13_q.fill_tensor2(Q_table_block2r_con2);

            npy14_q.fill_tensor2(Q_table_block2l_con0);

            npy15_q.fill_tensor2(Q_table_block2r_con3);

            npy16_q.fill_tensor2(Q_table_block2r_con4);

            npy17_q.fill_tensor2(Q_table_block2r_con5);

            npy18_q.fill_tensor2(Q_table_block2r_con6);

            npy19_q.fill_tensor2(Q_table_block2r_con7);

            npy20_q.fill_tensor2(Q_table_block2r_con8);

            npy21_q.fill_tensor2(Q_table_block2r_con9);

            npy22_q.fill_tensor2(Q_table_block2r_con10);

            npy23_q.fill_tensor2(Q_table_block2r_con11);

            npy24_q.fill_tensor2(Q_table_block3r_con0);

            npy25_q.fill_tensor2(Q_table_block3r_con1);

            npy26_q.fill_tensor2(Q_table_block3r_con2);

            npy27_q.fill_tensor2(Q_table_block3l_con0);

            npy28_q.fill_tensor2(Q_table_block3r_con3);

            npy29_q.fill_tensor2(Q_table_block3r_con4);

            npy30_q.fill_tensor2(Q_table_block3r_con5);

            npy31_q.fill_tensor2(Q_table_block3r_con6);

            npy32_q.fill_tensor2(Q_table_block3r_con7);

            npy33_q.fill_tensor2(Q_table_block3r_con8);

            npy34_q.fill_tensor2(Q_table_block3r_con9);

            npy35_q.fill_tensor2(Q_table_block3r_con10);

            npy36_q.fill_tensor2(Q_table_block3r_con11);

            npy37_q.fill_tensor2(Q_table_block3r_con12);

            npy38_q.fill_tensor2(Q_table_block3r_con13);

            npy39_q.fill_tensor2(Q_table_block3r_con14);

            npy40_q.fill_tensor2(Q_table_block3r_con15);

            npy41_q.fill_tensor2(Q_table_block3r_con16);

            npy42_q.fill_tensor2(Q_table_block3r_con17);

            npy43_q.fill_tensor2(Q_table_block4r_con0);

            npy44_q.fill_tensor2(Q_table_block4r_con1);

            npy45_q.fill_tensor2(Q_table_block4r_con2);

            npy46_q.fill_tensor2(Q_table_block4l_con0);

            npy47_q.fill_tensor2(Q_table_block4r_con3);

            npy48_q.fill_tensor2(Q_table_block4r_con4);

            npy49_q.fill_tensor2(Q_table_block4r_con5);

            npy50_q.fill_tensor2(Q_table_block4r_con6);

            npy51_q.fill_tensor2(Q_table_block4r_con7);

            npy52_q.fill_tensor2(Q_table_block4r_con8);

            npy53_q.fill_tensor2(Q_table_con1);





            npy0_wt.fill_tensor2(WT_buffer_con0);

            npy1_wt.fill_tensor2(WT_buffer_block1r_con0);

            npy2_wt.fill_tensor2(WT_buffer_block1r_con1);

            npy3_wt.fill_tensor2(WT_buffer_block1r_con2);

            npy4_wt.fill_tensor2(WT_buffer_block1l_con0);

            npy5_wt.fill_tensor2(WT_buffer_block1r_con3);

            npy6_wt.fill_tensor2(WT_buffer_block1r_con4);

            npy7_wt.fill_tensor2(WT_buffer_block1r_con5);

            npy8_wt.fill_tensor2(WT_buffer_block1r_con6);

            npy9_wt.fill_tensor2(WT_buffer_block1r_con7);

            npy10_wt.fill_tensor2(WT_buffer_block1r_con8);

            npy11_wt.fill_tensor2(WT_buffer_block2r_con0);

            npy12_wt.fill_tensor2(WT_buffer_block2r_con1);

            npy13_wt.fill_tensor2(WT_buffer_block2r_con2);

            npy14_wt.fill_tensor2(WT_buffer_block2l_con0);

            npy15_wt.fill_tensor2(WT_buffer_block2r_con3);

            npy16_wt.fill_tensor2(WT_buffer_block2r_con4);

            npy17_wt.fill_tensor2(WT_buffer_block2r_con5);

            npy18_wt.fill_tensor2(WT_buffer_block2r_con6);

            npy19_wt.fill_tensor2(WT_buffer_block2r_con7);

            npy20_wt.fill_tensor2(WT_buffer_block2r_con8);

            npy21_wt.fill_tensor2(WT_buffer_block2r_con9);

            npy22_wt.fill_tensor2(WT_buffer_block2r_con10);

            npy23_wt.fill_tensor2(WT_buffer_block2r_con11);

            npy24_wt.fill_tensor2(WT_buffer_block3r_con0);

            npy25_wt.fill_tensor2(WT_buffer_block3r_con1);

            npy26_wt.fill_tensor2(WT_buffer_block3r_con2);

            npy27_wt.fill_tensor2(WT_buffer_block3l_con0);

            npy28_wt.fill_tensor2(WT_buffer_block3r_con3);

            npy29_wt.fill_tensor2(WT_buffer_block3r_con4);

            npy30_wt.fill_tensor2(WT_buffer_block3r_con5);

            npy31_wt.fill_tensor2(WT_buffer_block3r_con6);

            npy32_wt.fill_tensor2(WT_buffer_block3r_con7);

            npy33_wt.fill_tensor2(WT_buffer_block3r_con8);

            npy34_wt.fill_tensor2(WT_buffer_block3r_con9);

            npy35_wt.fill_tensor2(WT_buffer_block3r_con10);

            npy36_wt.fill_tensor2(WT_buffer_block3r_con11);

            npy37_wt.fill_tensor2(WT_buffer_block3r_con12);

            npy38_wt.fill_tensor2(WT_buffer_block3r_con13);

            npy39_wt.fill_tensor2(WT_buffer_block3r_con14);

            npy40_wt.fill_tensor2(WT_buffer_block3r_con15);

            npy41_wt.fill_tensor2(WT_buffer_block3r_con16);

            npy42_wt.fill_tensor2(WT_buffer_block3r_con17);

            npy43_wt.fill_tensor2(WT_buffer_block4r_con0);

            npy44_wt.fill_tensor2(WT_buffer_block4r_con1);

            npy45_wt.fill_tensor2(WT_buffer_block4r_con2);

            npy46_wt.fill_tensor2(WT_buffer_block4l_con0);

            npy47_wt.fill_tensor2(WT_buffer_block4r_con3);

            npy48_wt.fill_tensor2(WT_buffer_block4r_con4);

            npy49_wt.fill_tensor2(WT_buffer_block4r_con5);

            npy50_wt.fill_tensor2(WT_buffer_block4r_con6);

            npy51_wt.fill_tensor2(WT_buffer_block4r_con7);

            npy52_wt.fill_tensor2(WT_buffer_block4r_con8);

            npy53_wt.fill_tensor2(WT_buffer_con1);

			



            npy0_b.fill_tensor2(bias_con0);

            npy1_b.fill_tensor2(bias_block1r_con0);

            npy2_b.fill_tensor2(bias_block1r_con1);

            npy3_b.fill_tensor2(bias_block1r_con2);

            npy4_b.fill_tensor2(bias_block1l_con0);

            npy5_b.fill_tensor2(bias_block1r_con3);

            npy6_b.fill_tensor2(bias_block1r_con4);

            npy7_b.fill_tensor2(bias_block1r_con5);

            npy8_b.fill_tensor2(bias_block1r_con6);

            npy9_b.fill_tensor2(bias_block1r_con7);

            npy10_b.fill_tensor2(bias_block1r_con8);

            npy11_b.fill_tensor2(bias_block2r_con0);

            npy12_b.fill_tensor2(bias_block2r_con1);

            npy13_b.fill_tensor2(bias_block2r_con2);

            npy14_b.fill_tensor2(bias_block2l_con0);

            npy15_b.fill_tensor2(bias_block2r_con3);

            npy16_b.fill_tensor2(bias_block2r_con4);

            npy17_b.fill_tensor2(bias_block2r_con5);

            npy18_b.fill_tensor2(bias_block2r_con6);

            npy19_b.fill_tensor2(bias_block2r_con7);

            npy20_b.fill_tensor2(bias_block2r_con8);

            npy21_b.fill_tensor2(bias_block2r_con9);

            npy22_b.fill_tensor2(bias_block2r_con10);

            npy23_b.fill_tensor2(bias_block2r_con11);

            npy24_b.fill_tensor2(bias_block3r_con0);

            npy25_b.fill_tensor2(bias_block3r_con1);

            npy26_b.fill_tensor2(bias_block3r_con2);

            npy27_b.fill_tensor2(bias_block3l_con0);

            npy28_b.fill_tensor2(bias_block3r_con3);

            npy29_b.fill_tensor2(bias_block3r_con4);

            npy30_b.fill_tensor2(bias_block3r_con5);

            npy31_b.fill_tensor2(bias_block3r_con6);

            npy32_b.fill_tensor2(bias_block3r_con7);

            npy33_b.fill_tensor2(bias_block3r_con8);

            npy34_b.fill_tensor2(bias_block3r_con9);

            npy35_b.fill_tensor2(bias_block3r_con10);

            npy36_b.fill_tensor2(bias_block3r_con11);

            npy37_b.fill_tensor2(bias_block3r_con12);

            npy38_b.fill_tensor2(bias_block3r_con13);

            npy39_b.fill_tensor2(bias_block3r_con14);

            npy40_b.fill_tensor2(bias_block3r_con15);

            npy41_b.fill_tensor2(bias_block3r_con16);

            npy42_b.fill_tensor2(bias_block3r_con17);

            npy43_b.fill_tensor2(bias_block4r_con0);

            npy44_b.fill_tensor2(bias_block4r_con1);

            npy45_b.fill_tensor2(bias_block4r_con2);

            npy46_b.fill_tensor2(bias_block4l_con0);

            npy47_b.fill_tensor2(bias_block4r_con3);

            npy48_b.fill_tensor2(bias_block4r_con4);

            npy49_b.fill_tensor2(bias_block4r_con5);

            npy50_b.fill_tensor2(bias_block4r_con6);

            npy51_b.fill_tensor2(bias_block4r_con7);

            npy52_b.fill_tensor2(bias_block4r_con8);

            npy53_b.fill_tensor2(bias_con1);



		/*npy100.fill_tensor2(input_fc);*/

		is_fortran      = npy_input.is_fortran();



		/*allocate end*/

		return true;

	}/*end of do_setup*/

void do_run()override

{

	/*defination*/

	double lend01=0,lend02=0,lend03=0;



	double lend111=0,lend112=0,lend113=0,lend114=0,lend115=0,lend116=0,lend117=0,lend118=0;

	double lend121=0,lend122=0,lend123=0,lend124=0,lend125=0,lend126=0,lend127=0;

	double lend131=0,lend132=0,lend133=0,lend134=0,lend135=0,lend136=0,lend137=0;



	double lend211=0,lend212=0,lend213=0,lend214=0,lend215=0,lend216=0,lend217=0,lend218=0;

	double lend221=0,lend222=0,lend223=0,lend224=0,lend225=0,lend226=0,lend227=0;

	double lend231=0,lend232=0,lend233=0,lend234=0,lend235=0,lend236=0,lend237=0;

	double lend241=0,lend242=0,lend243=0,lend244=0,lend245=0,lend246=0,lend247=0;



	double lend311=0,lend312=0,lend313=0,lend314=0,lend315=0,lend316=0,lend317=0,lend318=0;

	double lend321=0,lend322=0,lend323=0,lend324=0,lend325=0,lend326=0,lend327=0;

	double lend331=0,lend332=0,lend333=0,lend334=0,lend335=0,lend336=0,lend337=0;

	double lend341=0,lend342=0,lend343=0,lend344=0,lend345=0,lend346=0,lend347=0;

	double lend351=0,lend352=0,lend353=0,lend354=0,lend355=0,lend356=0,lend357=0;

	double lend361=0,lend362=0,lend363=0,lend364=0,lend365=0,lend366=0,lend367=0;



	double lend411=0,lend412=0,lend413=0,lend414=0,lend415=0,lend416=0,lend417=0,lend418=0;

	double lend421=0,lend422=0,lend423=0,lend424=0,lend425=0,lend426=0,lend427=0;

	double lend431=0,lend432=0,lend433=0,lend434=0,lend435=0,lend436=0,lend437=0;



    double lend11=0, lend12=0, lend13=0, lend14=0; 



	double total_time=0;

	double time=0;

	int cycles=101;



	std::string base_path = "/media/sdcard/ComputeLibrary";

	std::string output_file_path = "/model.csv";

	ofstream out(base_path+output_file_path, ios::out | ios::app);

	out<<"ResNet50 ABM"<<std::endl;

	for (int i = 0; i < cycles; i++)

	{

		auto start = std::chrono::high_resolution_clock::now();



		con0.run(); auto end01=std::chrono::high_resolution_clock::now();;

		lconv0sf.run();auto end02=std::chrono::high_resolution_clock::now();;

		act0.run();auto end03=std::chrono::high_resolution_clock::now();; 

		pool0.run();auto end04=std::chrono::high_resolution_clock::now();;

		lpool0fs.run();auto end05=std::chrono::high_resolution_clock::now();;



		block1r_con0.run(); auto end111=std::chrono::high_resolution_clock::now();;

		lb1rconv0sf.run();auto end112=std::chrono::high_resolution_clock::now();;

		block1r_act0.run();auto end113=std::chrono::high_resolution_clock::now();;

		lb1ract0fs.run();auto end114=std::chrono::high_resolution_clock::now();;

		block1r_con1.run(); auto end115=std::chrono::high_resolution_clock::now();;

		lb1rconv1sf.run();auto end116=std::chrono::high_resolution_clock::now();;

		block1r_act1.run();auto end117=std::chrono::high_resolution_clock::now();;

		lb1ract1fs.run();auto end118=std::chrono::high_resolution_clock::now();;

		block1r_con2.run(); auto end119=std::chrono::high_resolution_clock::now();;

		block1l_con0.run(); auto end1110=std::chrono::high_resolution_clock::now();;

		block1_add0.run(); auto end1111=std::chrono::high_resolution_clock::now();;

		lb1add0sf.run();auto end1112=std::chrono::high_resolution_clock::now();;

		block1_act0.run();auto end1113=std::chrono::high_resolution_clock::now();;

		lb1act0fs.run();auto end121=std::chrono::high_resolution_clock::now();;



		block1r_con3.run();auto end122=std::chrono::high_resolution_clock::now();;

		lb1rconv3sf.run();auto end123=std::chrono::high_resolution_clock::now();;

		block1r_act2.run();auto end124=std::chrono::high_resolution_clock::now();;

		lb1ract2fs.run();auto end125=std::chrono::high_resolution_clock::now();;

		block1r_con4.run(); auto end126=std::chrono::high_resolution_clock::now();;

		lb1rconv4sf.run();auto end127=std::chrono::high_resolution_clock::now();;

		block1r_act3.run();auto end128=std::chrono::high_resolution_clock::now();;

		lb1ract3fs.run();auto end129=std::chrono::high_resolution_clock::now();;

		block1r_con5.run(); auto end1210=std::chrono::high_resolution_clock::now();;

		block1_add1.run(); auto end1211=std::chrono::high_resolution_clock::now();;

		lb1add1sf.run();auto end1212=std::chrono::high_resolution_clock::now();;

		block1_act1.run();auto end1213=std::chrono::high_resolution_clock::now();;

		lb1act1fs.run();auto end131=std::chrono::high_resolution_clock::now();;



		block1r_con6.run(); auto end132=std::chrono::high_resolution_clock::now();;

		lb1rconv6sf.run();auto end133=std::chrono::high_resolution_clock::now();;

		block1r_act4.run(); auto end134=std::chrono::high_resolution_clock::now();;

		lb1ract4fs.run();auto end135=std::chrono::high_resolution_clock::now();;

		block1r_con7.run();  auto end136=std::chrono::high_resolution_clock::now();;

		lb1rconv7sf.run();auto end137=std::chrono::high_resolution_clock::now();;

		block1r_act5.run(); auto end138=std::chrono::high_resolution_clock::now();;

		lb1ract5fs.run();auto end139=std::chrono::high_resolution_clock::now();;

		block1r_con8.run();  auto end1310=std::chrono::high_resolution_clock::now();;

		block1_add2.run();  auto end1311=std::chrono::high_resolution_clock::now();;

		lb1add2sf.run();auto end1312=std::chrono::high_resolution_clock::now();;

		block1_act2.run(); auto end1313=std::chrono::high_resolution_clock::now();;

		lb1act2fs.run();auto end211=std::chrono::high_resolution_clock::now();;



		block2r_con0.run(); auto end212=std::chrono::high_resolution_clock::now();;

		lb2rconv0sf.run();auto end213=std::chrono::high_resolution_clock::now();;

		block2r_act0.run();auto end214=std::chrono::high_resolution_clock::now();;

		lb2ract0fs.run();auto end215=std::chrono::high_resolution_clock::now();;

		block2r_con1.run(); auto end216=std::chrono::high_resolution_clock::now();;

		lb2rconv1sf.run();auto end217=std::chrono::high_resolution_clock::now();;

		block2r_act1.run();auto end218=std::chrono::high_resolution_clock::now();;

		lb2ract1fs.run();auto end219=std::chrono::high_resolution_clock::now();;

		block2r_con2.run(); auto end2110=std::chrono::high_resolution_clock::now();;

		block2l_con0.run(); auto end2111=std::chrono::high_resolution_clock::now();;

		block2_add0.run(); auto end2112=std::chrono::high_resolution_clock::now();;

		lb2add0sf.run();auto end2113=std::chrono::high_resolution_clock::now();;

		block2_act0.run();auto end2114=std::chrono::high_resolution_clock::now();;

		lb2act0fs.run();auto end221=std::chrono::high_resolution_clock::now();;



		block2r_con3.run(); auto end222=std::chrono::high_resolution_clock::now();;

		lb2rconv3sf.run();auto end223=std::chrono::high_resolution_clock::now();;

		block2r_act2.run();auto end224=std::chrono::high_resolution_clock::now();;

		lb2ract2fs.run();auto end225=std::chrono::high_resolution_clock::now();;

		block2r_con4.run();auto end226=std::chrono::high_resolution_clock::now();;

		lb2rconv4sf.run();auto end227=std::chrono::high_resolution_clock::now();;

		block2r_act3.run();auto end228=std::chrono::high_resolution_clock::now();;

		lb2ract3fs.run();auto end229=std::chrono::high_resolution_clock::now();;

		block2r_con5.run(); auto end2210=std::chrono::high_resolution_clock::now();;

		block2_add1.run(); auto end2211=std::chrono::high_resolution_clock::now();;

		lb2add1sf.run();auto end2212=std::chrono::high_resolution_clock::now();;

		block2_act1.run();auto end2213=std::chrono::high_resolution_clock::now();;

		lb2act1fs.run();auto end231=std::chrono::high_resolution_clock::now();;



		block2r_con6.run(); auto end232=std::chrono::high_resolution_clock::now();;

		lb2rconv6sf.run();auto end233=std::chrono::high_resolution_clock::now();;

		block2r_act4.run();auto end234=std::chrono::high_resolution_clock::now();;

		lb2ract4fs.run();auto end235=std::chrono::high_resolution_clock::now();;

		block2r_con7.run();auto end236=std::chrono::high_resolution_clock::now();;

		lb2rconv7sf.run();auto end237=std::chrono::high_resolution_clock::now();;

		block2r_act5.run();auto end238=std::chrono::high_resolution_clock::now();;

		lb2ract5fs.run();auto end239=std::chrono::high_resolution_clock::now();;

		block2r_con8.run(); auto end2310=std::chrono::high_resolution_clock::now();;

		block2_add2.run();auto end2311=std::chrono::high_resolution_clock::now();;

		lb2add2sf.run();auto end2312=std::chrono::high_resolution_clock::now();;

		block2_act2.run();auto end2313=std::chrono::high_resolution_clock::now();;

		lb2act2fs.run();auto end241=std::chrono::high_resolution_clock::now();;



		block2r_con9.run(); auto end242=std::chrono::high_resolution_clock::now();;

		lb2rconv9sf.run();auto end243=std::chrono::high_resolution_clock::now();;

		block2r_act6.run();auto end244=std::chrono::high_resolution_clock::now();;

		lb2ract6fs.run();auto end245=std::chrono::high_resolution_clock::now();;

		block2r_con10.run();auto end246=std::chrono::high_resolution_clock::now();;

		lb2rconv10sf.run();auto end247=std::chrono::high_resolution_clock::now();;

		block2r_act7.run();auto end248=std::chrono::high_resolution_clock::now();;

		lb2ract7fs.run();auto end249=std::chrono::high_resolution_clock::now();;

		block2r_con11.run(); auto end2410=std::chrono::high_resolution_clock::now();;

		block2_add3.run(); auto end2411=std::chrono::high_resolution_clock::now();;

		lb2add3sf.run();auto end2412=std::chrono::high_resolution_clock::now();;

		block2_act3.run();auto end2413=std::chrono::high_resolution_clock::now();;

		lb2act3fs.run();auto end311=std::chrono::high_resolution_clock::now();;



		block3r_con0.run();auto end312=std::chrono::high_resolution_clock::now();;

		lb3rconv0sf.run();auto end313=std::chrono::high_resolution_clock::now();;

		block3r_act0.run();auto end314=std::chrono::high_resolution_clock::now();;

		lb3ract0fs.run();auto end315=std::chrono::high_resolution_clock::now();;

		block3r_con1.run(); auto end316=std::chrono::high_resolution_clock::now();;

		lb3rconv1sf.run();auto end317=std::chrono::high_resolution_clock::now();;

		block3r_act1.run();auto end318=std::chrono::high_resolution_clock::now();;

		lb3ract1fs.run();auto end319=std::chrono::high_resolution_clock::now();;

		block3r_con2.run(); auto end3110=std::chrono::high_resolution_clock::now();;

		 block3l_con0.run(); auto end3111=std::chrono::high_resolution_clock::now();;

		block3_add0.run(); auto end3112=std::chrono::high_resolution_clock::now();;

		lb3add0sf.run();auto end3113=std::chrono::high_resolution_clock::now();;

		block3_act0.run();auto end3114=std::chrono::high_resolution_clock::now();;

		lb3act0fs.run();auto end321=std::chrono::high_resolution_clock::now();;



		block3r_con3.run(); auto end322=std::chrono::high_resolution_clock::now();;

		lb3rconv3sf.run();auto end323=std::chrono::high_resolution_clock::now();;

		block3r_act2.run();auto end324=std::chrono::high_resolution_clock::now();;

		lb3ract2fs.run();auto end325=std::chrono::high_resolution_clock::now();;

		block3r_con4.run(); auto end326=std::chrono::high_resolution_clock::now();;

		lb3rconv4sf.run();auto end327=std::chrono::high_resolution_clock::now();;

		block3r_act3.run();auto end328=std::chrono::high_resolution_clock::now();;

		lb3ract3fs.run();auto end329=std::chrono::high_resolution_clock::now();;

		block3r_con5.run(); auto end3210=std::chrono::high_resolution_clock::now();;

		block3_add1.run(); auto end3211=std::chrono::high_resolution_clock::now();;

		lb3add1sf.run();auto end3212=std::chrono::high_resolution_clock::now();;

		 block3_act1.run();auto end3213=std::chrono::high_resolution_clock::now();;

		lb3act1fs.run();auto end331=std::chrono::high_resolution_clock::now();;



		block3r_con6.run(); auto end332=std::chrono::high_resolution_clock::now();;

		lb3rconv6sf.run();auto end333=std::chrono::high_resolution_clock::now();;

		block3r_act4.run();auto end334=std::chrono::high_resolution_clock::now();;

		lb3ract4fs.run();auto end335=std::chrono::high_resolution_clock::now();;

		block3r_con7.run(); auto end336=std::chrono::high_resolution_clock::now();;

		lb3rconv7sf.run();auto end337=std::chrono::high_resolution_clock::now();;

		block3r_act5.run();auto end338=std::chrono::high_resolution_clock::now();;

		lb3ract5fs.run();auto end339=std::chrono::high_resolution_clock::now();;

		block3r_con8.run(); auto end3310=std::chrono::high_resolution_clock::now();;

		block3_add2.run();auto end3311=std::chrono::high_resolution_clock::now();;

		lb3add2sf.run();auto end3312=std::chrono::high_resolution_clock::now();;

		block3_act2.run();auto end3313=std::chrono::high_resolution_clock::now();;

		lb3act2fs.run();auto end341=std::chrono::high_resolution_clock::now();;



		block3r_con9.run(); auto end342=std::chrono::high_resolution_clock::now();;

		lb3rconv9sf.run();auto end343=std::chrono::high_resolution_clock::now();;

		block3r_act6.run();auto end344=std::chrono::high_resolution_clock::now();;

		lb3ract6fs.run();auto end345=std::chrono::high_resolution_clock::now();;

		block3r_con10.run(); auto end346=std::chrono::high_resolution_clock::now();;

		lb3rconv10sf.run();auto end347=std::chrono::high_resolution_clock::now();;

		block3r_act7.run();auto end348=std::chrono::high_resolution_clock::now();;

		lb3ract7fs.run();auto end349=std::chrono::high_resolution_clock::now();;

		block3r_con11.run(); auto end3410=std::chrono::high_resolution_clock::now();;

		block3_add3.run(); auto end3411=std::chrono::high_resolution_clock::now();;

		lb3add3sf.run();auto end3412=std::chrono::high_resolution_clock::now();;

		block3_act3.run();auto end3413=std::chrono::high_resolution_clock::now();;

		lb3act3fs.run();auto end351=std::chrono::high_resolution_clock::now();;



		block3r_con12.run(); auto end352=std::chrono::high_resolution_clock::now();;

		lb3rconv12sf.run();auto end353=std::chrono::high_resolution_clock::now();;

		block3r_act8.run();auto end354=std::chrono::high_resolution_clock::now();;

		lb3ract8fs.run();auto end355=std::chrono::high_resolution_clock::now();;

		block3r_con13.run(); auto end356=std::chrono::high_resolution_clock::now();;

		lb3rconv13sf.run();auto end357=std::chrono::high_resolution_clock::now();;

		block3r_act9.run();auto end358=std::chrono::high_resolution_clock::now();;

		lb3ract9fs.run();auto end359=std::chrono::high_resolution_clock::now();;

		block3r_con14.run(); auto end3510=std::chrono::high_resolution_clock::now();;

		block3_add4.run();auto end3511=std::chrono::high_resolution_clock::now();;

		lb3add4sf.run();auto end3512=std::chrono::high_resolution_clock::now();;

		block3_act4.run();auto end3513=std::chrono::high_resolution_clock::now();;

		lb3act4fs.run();auto end361=std::chrono::high_resolution_clock::now();;



		block3r_con15.run();auto end362=std::chrono::high_resolution_clock::now();;

		lb3rconv15sf.run();auto end363=std::chrono::high_resolution_clock::now();;

		block3r_act10.run();auto end364=std::chrono::high_resolution_clock::now();;

		lb3ract10fs.run();auto end365=std::chrono::high_resolution_clock::now();;

		block3r_con16.run(); auto end366=std::chrono::high_resolution_clock::now();;

		lb3rconv16sf.run();auto end367=std::chrono::high_resolution_clock::now();;

		block3r_act11.run();auto end368=std::chrono::high_resolution_clock::now();;

		lb3ract11fs.run();auto end369=std::chrono::high_resolution_clock::now();;

		block3r_con17.run(); auto end3610=std::chrono::high_resolution_clock::now();;

		block3_add5.run();auto end3611=std::chrono::high_resolution_clock::now();;

		lb3add5sf.run();auto end3612=std::chrono::high_resolution_clock::now();;

		block3_act5.run();auto end3613=std::chrono::high_resolution_clock::now();;

		lb3act5fs.run();auto end411=std::chrono::high_resolution_clock::now();;



		block4r_con0.run(); auto end412=std::chrono::high_resolution_clock::now();;

		lb4rconv0sf.run();auto end413=std::chrono::high_resolution_clock::now();;

		block4r_act0.run(); auto end414=std::chrono::high_resolution_clock::now();;

		lb4ract0fs.run();auto end415=std::chrono::high_resolution_clock::now();;

		block4r_con1.run();  auto end416=std::chrono::high_resolution_clock::now();;

		lb4rconv1sf.run();auto end417=std::chrono::high_resolution_clock::now();;

		block4r_act1.run(); auto end418=std::chrono::high_resolution_clock::now();;

		lb4ract1fs.run();auto end419=std::chrono::high_resolution_clock::now();;

		block4r_con2.run();  auto end4110=std::chrono::high_resolution_clock::now();;

		block4l_con0.run();  auto end4111=std::chrono::high_resolution_clock::now();;

		block4_add0.run();  auto end4112=std::chrono::high_resolution_clock::now();;

		lb4add0sf.run();auto end4113=std::chrono::high_resolution_clock::now();;

		block4_act0.run(); auto end4114=std::chrono::high_resolution_clock::now();;

		lb4act0fs.run();auto end421=std::chrono::high_resolution_clock::now();;



		block4r_con3.run(); auto end422=std::chrono::high_resolution_clock::now();;

		lb4rconv3sf.run();auto end423=std::chrono::high_resolution_clock::now();;

		block4r_act2.run();auto end424=std::chrono::high_resolution_clock::now();;

		lb4ract2fs.run();auto end425=std::chrono::high_resolution_clock::now();;

		block4r_con4.run(); auto end426=std::chrono::high_resolution_clock::now();;

		lb4rconv4sf.run();auto end427=std::chrono::high_resolution_clock::now();;

		block4r_act3.run();auto end428=std::chrono::high_resolution_clock::now();;

		lb4ract3fs.run();auto end429=std::chrono::high_resolution_clock::now();;

		block4r_con5.run(); auto end4210=std::chrono::high_resolution_clock::now();;

		block4_add1.run(); auto end4211=std::chrono::high_resolution_clock::now();;

		lb4add1sf.run();auto end4212=std::chrono::high_resolution_clock::now();;

		block4_act1.run();auto end4213=std::chrono::high_resolution_clock::now();;

		lb4act1fs.run();auto end431=std::chrono::high_resolution_clock::now();;



		block4r_con6.run(); auto end432=std::chrono::high_resolution_clock::now();;

		lb4rconv6sf.run();auto end433=std::chrono::high_resolution_clock::now();;

		block4r_act4.run();auto end434=std::chrono::high_resolution_clock::now();;

		lb4ract4fs.run();auto end435=std::chrono::high_resolution_clock::now();;

		block4r_con7.run(); auto end436=std::chrono::high_resolution_clock::now();;

		lb4rconv7sf.run();auto end437=std::chrono::high_resolution_clock::now();;

		block4r_act5.run();auto end438=std::chrono::high_resolution_clock::now();;

		lb4ract5fs.run();auto end439=std::chrono::high_resolution_clock::now();;

		block4r_con8.run(); auto end4310=std::chrono::high_resolution_clock::now();;

		block4_add2.run(); auto end4311=std::chrono::high_resolution_clock::now();;

		lb4add2sf.run();auto end4312=std::chrono::high_resolution_clock::now();;

		block4_act2.run();auto end4313=std::chrono::high_resolution_clock::now();;

	



		pool1.run();auto end11=std::chrono::high_resolution_clock::now();;

		lpool1fs.run();auto end12=std::chrono::high_resolution_clock::now();;

		resize.run();end12=std::chrono::high_resolution_clock::now();;

		 con1.run(); auto end13=std::chrono::high_resolution_clock::now();;

		 lconv1sf.run();auto end14=std::chrono::high_resolution_clock::now();;

		flatten.run();auto end15=std::chrono::high_resolution_clock::now();;

		softmax.run();auto end16=std::chrono::high_resolution_clock::now();;

		

		/* calculate layer inference time*/

		if(i>0){

			double one_runtime=0;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end01 - start).count();lend01+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end03 - end02).count();lend02+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end04 - end03).count();lend03+=time; one_runtime+=time;



			time = std::chrono::duration_cast<std::chrono::duration<double>>(end111 - end05).count();lend111+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end113 - end112).count();lend112+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end115 - end114).count();lend113+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end117 - end116).count();lend114+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end119 - end118).count();lend115+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end1110 - end119).count();lend116+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end1111 - end1110).count();lend117+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end1113 - end1112).count();lend118+=time; one_runtime+=time;





			time = std::chrono::duration_cast<std::chrono::duration<double>>(end122 - end121).count();lend121+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end124 - end123).count();lend122+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end126 - end125).count();lend123+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end128 - end127).count();lend124+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end1210 - end129).count();lend125+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end1211 - end1210).count();lend126+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end1213 - end1212).count();lend127+=time; one_runtime+=time;





			time = std::chrono::duration_cast<std::chrono::duration<double>>(end132 - end131).count();lend131+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end134 - end133).count();lend132+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end136 - end135).count();lend133+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end138 - end137).count();lend134+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end1310 - end139).count();lend135+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end1311 - end1310).count();lend136+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end1313 - end1312).count();lend137+=time; one_runtime+=time;



			time = std::chrono::duration_cast<std::chrono::duration<double>>(end212 - end211).count();lend211+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end214 - end213).count();lend212+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end216 - end215).count();lend213+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end218 - end217).count();lend214+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end2110 - end219).count();lend215+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end2111 - end2110).count();lend216+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end2112 - end2111).count();lend217+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end2114 - end2113).count();lend218+=time; one_runtime+=time;





			time = std::chrono::duration_cast<std::chrono::duration<double>>(end222 - end221).count();lend221+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end224 - end223).count();lend222+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end226 - end225).count();lend223+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end228 - end227).count();lend224+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end2210 - end229).count();lend225+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end2211 - end2210).count();lend226+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end2213 - end2212).count();lend227+=time; one_runtime+=time;





			time = std::chrono::duration_cast<std::chrono::duration<double>>(end232 - end231).count();lend231+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end234 - end233).count();lend232+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end236 - end235).count();lend233+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end238 - end237).count();lend234+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end2310 - end239).count();lend235+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end2311 - end2310).count();lend236+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end2313 - end2312).count();lend237+=time; one_runtime+=time;





			time = std::chrono::duration_cast<std::chrono::duration<double>>(end242 - end241).count();lend241+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end244 - end243).count();lend242+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end246 - end245).count();lend243+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end248 - end247).count();lend244+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end2410 - end249).count();lend245+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end2411 - end2410).count();lend246+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end2413 - end2412).count();lend247+=time; one_runtime+=time;





			time = std::chrono::duration_cast<std::chrono::duration<double>>(end312 - end311).count();lend311+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end314 - end313).count();lend312+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end316 - end315).count();lend313+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end318 - end317).count();lend314+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end3110 - end319).count();lend315+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end3111 - end3110).count();lend316+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end3112 - end3111).count();lend317+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end3114 - end3113).count();lend318+=time; one_runtime+=time;





			time = std::chrono::duration_cast<std::chrono::duration<double>>(end322 - end321).count();lend321+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end324 - end323).count();lend322+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end326 - end325).count();lend323+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end328 - end327).count();lend324+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end3210 - end329).count();lend325+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end3211 - end3210).count();lend326+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end3213 - end3212).count();lend327+=time; one_runtime+=time;





			time = std::chrono::duration_cast<std::chrono::duration<double>>(end332 - end331).count();lend331+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end334 - end333).count();lend332+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end336 - end335).count();lend333+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end338 - end337).count();lend334+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end3310 - end339).count();lend335+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end3311 - end3310).count();lend336+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end3313 - end3312).count();lend337+=time; one_runtime+=time;





			time = std::chrono::duration_cast<std::chrono::duration<double>>(end342 - end341).count();lend341+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end344 - end343).count();lend342+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end346 - end345).count();lend343+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end348 - end347).count();lend344+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end3410 - end349).count();lend345+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end3411 - end3410).count();lend346+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end3413 - end3412).count();lend347+=time; one_runtime+=time;





			time = std::chrono::duration_cast<std::chrono::duration<double>>(end352 - end351).count();lend351+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end354 - end353).count();lend352+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end356 - end355).count();lend353+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end358 - end357).count();lend354+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end3510 - end359).count();lend355+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end3511 - end3510).count();lend356+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end3513 - end3512).count();lend357+=time; one_runtime+=time;



			time = std::chrono::duration_cast<std::chrono::duration<double>>(end362 - end361).count();lend361+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end364 - end363).count();lend362+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end366 - end365).count();lend363+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end368 - end367).count();lend364+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end3610 - end369).count();lend365+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end3611 - end3610).count();lend366+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end3613 - end3612).count();lend367+=time; one_runtime+=time;





			time = std::chrono::duration_cast<std::chrono::duration<double>>(end412 - end411).count();lend411+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end414 - end413).count();lend412+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end416 - end415).count();lend413+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end418 - end417).count();lend414+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end4110 - end419).count();lend415+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end4111 - end4110).count();lend416+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end4112 - end4111).count();lend417+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end4114 - end4113).count();lend418+=time; one_runtime+=time;





			time = std::chrono::duration_cast<std::chrono::duration<double>>(end422 - end421).count();lend421+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end424 - end423).count();lend422+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end426 - end425).count();lend423+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end428 - end427).count();lend424+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end4210 - end429).count();lend425+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end4211 - end4210).count();lend426+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end4213 - end4212).count();lend427+=time; one_runtime+=time;





			time = std::chrono::duration_cast<std::chrono::duration<double>>(end432 - end431).count();lend431+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end434 - end433).count();lend432+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end436 - end435).count();lend433+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end438 - end437).count();lend434+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end4310 - end439).count();lend435+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end4311 - end4310).count();lend436+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end4313 - end4312).count();lend437+=time; one_runtime+=time;





			time = std::chrono::duration_cast<std::chrono::duration<double>>(end11 - end4313).count();lend11+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end13 - end12).count();lend12+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end15 - end14).count();lend13+=time; one_runtime+=time;

			time = std::chrono::duration_cast<std::chrono::duration<double>>(end16 - end15).count();lend14+=time; one_runtime+=time;

			if(i>0){

			std::cout<<i<<"---run:"<<std::endl;

			std::cout<<"time="<<one_runtime*1000<<"ms"<<std::endl;

			total_time+=one_runtime;  

			out<<"one run time"<<","<<one_runtime*1000<<std::endl;  

			}

			if(i==0){

				std::cout<<"First run: "<<std::endl;

				std::cout<<"time="<<one_runtime*1000<<"ms"<<std::endl;

			}

			}



		    /* kernel time */

			add_kernel_time(con0.print_kernel_time());



			add_kernel_time(block1r_con0.print_kernel_time());add_kernel_time(block1r_con1.print_kernel_time());add_kernel_time(block1r_con2.print_kernel_time());add_kernel_time(block1l_con0.print_kernel_time());

			add_kernel_time(block1r_con3.print_kernel_time());add_kernel_time(block1r_con4.print_kernel_time());add_kernel_time(block1r_con5.print_kernel_time());

			add_kernel_time(block1r_con6.print_kernel_time());add_kernel_time(block1r_con7.print_kernel_time());add_kernel_time(block1r_con8.print_kernel_time());



			add_kernel_time(block2r_con0.print_kernel_time());add_kernel_time(block2r_con1.print_kernel_time());add_kernel_time(block2r_con2.print_kernel_time());add_kernel_time(block2l_con0.print_kernel_time());

			add_kernel_time(block2r_con3.print_kernel_time());add_kernel_time(block2r_con4.print_kernel_time());add_kernel_time(block2r_con5.print_kernel_time());

			add_kernel_time(block2r_con6.print_kernel_time());add_kernel_time(block2r_con7.print_kernel_time());add_kernel_time(block2r_con8.print_kernel_time());

			add_kernel_time(block2r_con9.print_kernel_time());add_kernel_time(block2r_con10.print_kernel_time());add_kernel_time(block2r_con11.print_kernel_time());



			add_kernel_time(block3r_con0.print_kernel_time());add_kernel_time(block3r_con1.print_kernel_time());add_kernel_time(block3r_con2.print_kernel_time());add_kernel_time(block3l_con0.print_kernel_time());

			add_kernel_time(block3r_con3.print_kernel_time());add_kernel_time(block3r_con4.print_kernel_time());add_kernel_time(block3r_con5.print_kernel_time());

			add_kernel_time(block3r_con6.print_kernel_time());add_kernel_time(block3r_con7.print_kernel_time());add_kernel_time(block3r_con8.print_kernel_time());

			add_kernel_time(block3r_con9.print_kernel_time());add_kernel_time(block3r_con10.print_kernel_time());add_kernel_time(block3r_con11.print_kernel_time());

			add_kernel_time(block3r_con12.print_kernel_time());add_kernel_time(block3r_con13.print_kernel_time());add_kernel_time(block3r_con14.print_kernel_time());

			add_kernel_time(block3r_con15.print_kernel_time());add_kernel_time(block3r_con16.print_kernel_time());add_kernel_time(block3r_con17.print_kernel_time());



			add_kernel_time(block4r_con0.print_kernel_time());add_kernel_time(block4r_con1.print_kernel_time());add_kernel_time(block4r_con2.print_kernel_time());add_kernel_time(block4l_con0.print_kernel_time());

			add_kernel_time(block4r_con3.print_kernel_time());add_kernel_time(block4r_con4.print_kernel_time());add_kernel_time(block4r_con5.print_kernel_time());

			add_kernel_time(block4r_con6.print_kernel_time());add_kernel_time(block4r_con7.print_kernel_time());add_kernel_time(block4r_con8.print_kernel_time());



			add_kernel_time(con1.print_kernel_time());

	}



		std::cout<<"Im2Col                   :"<<im2col_kernel_time*1000/(cycles-1)<<"ms"<<std::endl;

		std::cout<<"Interleave             :"<<interleave_kernel_time*1000/(cycles-1)<<"ms"<<std::endl;

		std::cout<<"Matrix_multiply :"<<matrix_multiply_kernel_time*1000/(cycles-1)<<"ms"<<std::endl;

		std::cout<<"Mmlast                  :"<<mmlast_kernel_time*1000/(cycles-1)<<"ms"<<std::endl;

		std::cout<<"Col2Im                   :"<<col2im_kernel_time*1000/(cycles-1)<<"ms"<<std::endl;



		arm_compute::utils::NPYLoader save;

		save.save_to_npy2(pool1fs,output_filename,false);

		save.save_to_npy2(out_con1,output_filename1,false);

		save.save_to_npy2(out_block4_add2,output_filename2,false);

		save.save_to_npy2(pool1fs,output_filename3,false);

		save_to_npy(conv0sf, "/media/sdcard/ComputeLibrary/data/neon_resnet50/act_output.npy", false);

		save.save_to_npy2(out_con0,"/media/sdcard/ComputeLibrary/data/neon_resnet50/act_output_s8.npy",false);

		

		

		out<<"Resnet50"<<std::endl;

		out << "---conv1       " <<","<< lend01 * 1000/(cycles-1) <<std::endl;

		out << "---relu1       " <<","<< lend02* 1000/(cycles-1) <<std::endl;

		out << "---pooling1    " <<","<< lend03 * 1000/(cycles-1) <<std::endl;



		out<<"---layer1      "<<std::endl;

		out<<"  ---0         "<<std::endl;

		out << "   ---conv1    " << ","<< lend111 * 1000/(cycles-1) <<std::endl;

		out << "   ---relu1    " << ","<< lend112 * 1000/(cycles-1) <<std::endl;

		out << "   ---conv2    " << ","<< lend113 * 1000/(cycles-1) <<std::endl;

		out << "   ---relu2    " << ","<< lend114 * 1000/(cycles-1) <<std::endl;

		out << "   ---conv3    " << ","<< lend115 * 1000/(cycles-1) <<std::endl;

		out << "   ---conv    " << ","<< lend116 * 1000/(cycles-1) <<std::endl;

		out << "   ---add      " << ","<< lend117 * 1000/(cycles-1) <<std::endl;

		out << "   ---relu     " << ","<< lend118 * 1000/(cycles-1) <<std::endl;

		

		out<<"  ---1         "<<std::endl;

		out << "   ---conv1    " << ","<< lend121 * 1000/(cycles-1) <<std::endl;

		out << "   ---relu1    " << ","<< lend122 * 1000/(cycles-1) <<std::endl;

		out << "   ---conv2    " << ","<< lend123 * 1000/(cycles-1) <<std::endl;

		out << "   ---relu2    " << ","<< lend124 * 1000/(cycles-1) <<std::endl;

		out << "   ---conv3    " << ","<< lend125 * 1000/(cycles-1) <<std::endl;

		out << "   ---add      " << ","<< lend126 * 1000/(cycles-1) <<std::endl;

		out << "   ---relu     " << ","<< lend127 * 1000/(cycles-1) <<std::endl;



		out<<"  ---2         "<<std::endl;

		out << "   ---conv1    " << ","<< lend131 * 1000/(cycles-1) <<std::endl;

		out << "   ---relu1    " << ","<< lend132 * 1000/(cycles-1) <<std::endl;

		out << "   ---conv2    " << ","<< lend133 * 1000/(cycles-1) <<std::endl;

		out << "   ---relu2    " << ","<< lend134 * 1000/(cycles-1) <<std::endl;

		out << "   ---conv3    " << ","<< lend135 * 1000/(cycles-1) <<std::endl;

		out << "   ---add      " << ","<< lend136 * 1000/(cycles-1) <<std::endl;

		out << "   ---relu     " << ","<< lend137 * 1000/(cycles-1) <<std::endl;



		out<<"---layer2      "<<std::endl;

		out<<"  ---0         "<<std::endl;

		out << "   ---conv1    " << ","<< lend211 * 1000/(cycles-1) <<std::endl;

		out << "   ---relu1    " << ","<< lend212 * 1000/(cycles-1) <<std::endl;

		out << "   ---conv2    " << ","<< lend213 * 1000/(cycles-1) <<std::endl;

		out << "   ---relu2    " << ","<< lend214 * 1000/(cycles-1) <<std::endl;

		out << "   ---conv3    " << ","<< lend215 * 1000/(cycles-1) <<std::endl;

		out << "   ---conv    " << ","<< lend216 * 1000/(cycles-1) <<std::endl;

		out << "   ---add      " << ","<< lend217 * 1000/(cycles-1) <<std::endl;

		out << "   ---relu     " << ","<< lend218 * 1000/(cycles-1) <<std::endl;

		

		out<<"  ---1         "<<std::endl;

		out << "   ---conv1    " << ","<< lend221 * 1000/(cycles-1) <<std::endl;

		out << "   ---relu1    " << ","<< lend222 * 1000/(cycles-1) <<std::endl;

		out << "   ---conv2    " << ","<< lend223 * 1000/(cycles-1) <<std::endl;

		out << "   ---relu2    " << ","<< lend224 * 1000/(cycles-1) <<std::endl;

		out << "   ---conv3    " << ","<< lend225 * 1000/(cycles-1) <<std::endl;

		out << "   ---add      " << ","<< lend226 * 1000/(cycles-1) <<std::endl;

		out << "   ---relu     " << ","<< lend227 * 1000/(cycles-1) <<std::endl;



		out<<"  ---2         "<<std::endl;

		out << "   ---conv1    " << ","<< lend231* 1000/(cycles-1) <<std::endl;

		out << "   ---relu1    " << ","<< lend232 * 1000/(cycles-1) <<std::endl;

		out << "   ---conv2    " << ","<< lend233 * 1000/(cycles-1) <<std::endl;

		out << "   ---relu2    " << ","<< lend234 * 1000/(cycles-1) <<std::endl;

		out << "   ---conv3    " << ","<< lend235 * 1000/(cycles-1) <<std::endl;

		out << "   ---add      " << ","<< lend236 * 1000/(cycles-1) <<std::endl;

		out << "   ---relu     " << ","<< lend237 * 1000/(cycles-1) <<std::endl;



		out<<"  ---3         "<<std::endl;

		out << "   ---conv1    " << ","<< lend241 * 1000/(cycles-1) <<std::endl;

		out << "   ---relu1    " << ","<< lend242 * 1000/(cycles-1) <<std::endl;

		out << "   ---conv2    " << ","<< lend243 * 1000/(cycles-1) <<std::endl;

		out << "   ---relu2    " << ","<< lend244 * 1000/(cycles-1) <<std::endl;

		out << "   ---conv3    " << ","<< lend245 * 1000/(cycles-1) <<std::endl;

		out << "   ---add      " << ","<< lend246 * 1000/(cycles-1) <<std::endl;

		out << "   ---relu     " << ","<< lend247 * 1000/(cycles-1) <<std::endl;



		out<<"---layer3      "<<std::endl;

		out<<"  ---0         "<<std::endl;

		out << "   ---conv1    " << ","<< lend311 * 1000/(cycles-1) <<std::endl;

		out << "   ---relu1    " << ","<< lend312 * 1000/(cycles-1) <<std::endl;

		out << "   ---conv2    " << ","<< lend313 * 1000/(cycles-1) <<std::endl;

		out << "   ---relu2    " << ","<< lend314 * 1000/(cycles-1) <<std::endl;

		out << "   ---conv3    " << ","<< lend315 * 1000/(cycles-1) <<std::endl;

		out << "   ---conv    " << ","<< lend316 * 1000/(cycles-1) <<std::endl;

		out << "   ---add      " << ","<< lend317 * 1000/(cycles-1) <<std::endl;

		out << "   ---relu     " << ","<< lend318 * 1000/(cycles-1) <<std::endl;

		

		out<<"  ---1         "<<std::endl;

		out << "   ---conv1    " << ","<< lend321 * 1000/(cycles-1) <<std::endl;

		out << "   ---relu1    " << ","<< lend322 * 1000/(cycles-1) <<std::endl;

		out << "   ---conv2    " << ","<< lend323 * 1000/(cycles-1) <<std::endl;

		out << "   ---relu2    " << ","<< lend324 * 1000/(cycles-1) <<std::endl;

		out << "   ---conv3    " << ","<< lend325 * 1000/(cycles-1) <<std::endl;

		out << "   ---add      " << ","<< lend326 * 1000/(cycles-1) <<std::endl;

		out << "   ---relu     " << ","<< lend327 * 1000/(cycles-1) <<std::endl;



		out<<"  ---2         "<<std::endl;

		out << "   ---conv1    " << ","<< lend331 * 1000/(cycles-1) <<std::endl;

		out << "   ---relu1    " << ","<< lend332 * 1000/(cycles-1) <<std::endl;

		out << "   ---conv2    " << ","<< lend333 * 1000/(cycles-1) <<std::endl;

		out << "   ---relu2    " << ","<< lend334 * 1000/(cycles-1) <<std::endl;

		out << "   ---conv3    " << ","<< lend335 * 1000/(cycles-1) <<std::endl;

		out << "   ---add      " << ","<< lend336 * 1000/(cycles-1) <<std::endl;

		out << "   ---relu     " << ","<< lend337 * 1000/(cycles-1) <<std::endl;



		out<<"  ---3         "<<std::endl;

		out << "   ---conv1    " << ","<< lend341 * 1000/(cycles-1) <<std::endl;

		out << "   ---relu1    " << ","<< lend342 * 1000/(cycles-1) <<std::endl;

		out << "   ---conv2    " << ","<< lend343 * 1000/(cycles-1) <<std::endl;

		out << "   ---relu2    " << ","<< lend344 * 1000/(cycles-1) <<std::endl;

		out << "   ---conv3    " << ","<< lend345 * 1000/(cycles-1) <<std::endl;

		out << "   ---add      " << ","<< lend346 * 1000/(cycles-1) <<std::endl;

		out << "   ---relu     " << ","<< lend347 * 1000/(cycles-1) <<std::endl;



		out<<"  ---4         "<<std::endl;

		out << "   ---conv1    " << ","<< lend351 * 1000/(cycles-1) <<std::endl;

		out << "   ---relu1    " << ","<< lend352 * 1000/(cycles-1) <<std::endl;

		out << "   ---conv2    " << ","<< lend353 * 1000/(cycles-1) <<std::endl;

		out << "   ---relu2    " << ","<< lend354 * 1000/(cycles-1) <<std::endl;

		out << "   ---conv3    " << ","<< lend355 * 1000/(cycles-1) <<std::endl;

		out << "   ---add      " << ","<< lend356 * 1000/(cycles-1) <<std::endl;

		out << "   ---relu     " << ","<< lend357 * 1000/(cycles-1) <<std::endl;



		out<<"  ---5         "<<std::endl;

		out << "   ---conv1    " << ","<< lend361 * 1000/(cycles-1) <<std::endl;

		out << "   ---relu1    " << ","<< lend362 * 1000/(cycles-1) <<std::endl;

		out << "   ---conv2    " << ","<< lend363 * 1000/(cycles-1) <<std::endl;

		out << "   ---relu2    " << ","<< lend364 * 1000/(cycles-1) <<std::endl;

		out << "   ---conv3    " << ","<< lend365 * 1000/(cycles-1) <<std::endl;

		out << "   ---add      " << ","<< lend366 * 1000/(cycles-1) <<std::endl;

		out << "   ---relu     " << ","<< lend367 * 1000/(cycles-1) <<std::endl;



		out<<"---layer4      "<<std::endl;

		out<<"  ---0         "<<std::endl;

		out << "   ---conv1    " << ","<< lend411 * 1000/(cycles-1) <<std::endl;

		out << "   ---relu1    " << ","<< lend412 * 1000/(cycles-1) <<std::endl;

		out << "   ---conv2    " << ","<< lend413 * 1000/(cycles-1) <<std::endl;

		out << "   ---relu2    " << ","<< lend414 * 1000/(cycles-1) <<std::endl;

		out << "   ---conv3    " << ","<< lend415 * 1000/(cycles-1) <<std::endl;

		out << "   ---conv    " << ","<< lend416 * 1000/(cycles-1) <<std::endl;

		out << "   ---add      " << ","<< lend417 * 1000/(cycles-1) <<std::endl;

		out << "   ---relu     " << ","<< lend418 * 1000/(cycles-1) <<std::endl;

		

		out<<"  ---1         "<<std::endl;

		out << "   ---conv1    " << ","<< lend421 * 1000/(cycles-1) <<std::endl;

		out << "   ---relu1    " << ","<< lend422 * 1000/(cycles-1) <<std::endl;

		out << "   ---conv2    " << ","<< lend423 * 1000/(cycles-1) <<std::endl;

		out << "   ---relu2    " << ","<< lend424 * 1000/(cycles-1) <<std::endl;

		out << "   ---conv3    " << ","<< lend425 * 1000/(cycles-1) <<std::endl;

		out << "   ---add      " << ","<< lend426 * 1000/(cycles-1) <<std::endl;

		out << "   ---relu     " << ","<< lend427 * 1000/(cycles-1) <<std::endl;



		out<<"  ---2         "<<std::endl;

		out << "   ---conv1    " << ","<< lend431 * 1000/(cycles-1) <<std::endl;

		out << "   ---relu1    " << ","<< lend432 * 1000/(cycles-1) <<std::endl;

		out << "   ---conv2    " << ","<< lend433 * 1000/(cycles-1) <<std::endl;

		out << "   ---relu2    " << ","<< lend434 * 1000/(cycles-1) <<std::endl;

		out << "   ---conv3    " << ","<< lend435 * 1000/(cycles-1) <<std::endl;

		out << "   ---add      " << ","<< lend436 * 1000/(cycles-1) <<std::endl;

		out << "   ---relu     " << ","<< lend437 * 1000/(cycles-1) <<std::endl;	



		out << "---pooling     " << ","<< lend11 * 1000/(cycles-1) <<std::endl;

		out << "---conv1       " << ","<<lend12 * 1000/(cycles-1) <<std::endl;

		out << "---flatten     "<<","<< lend13 * 1000/(cycles-1) <<std::endl;

		out << "---softmax   " <<","<< lend14 * 1000/(cycles-1) <<std::endl;



			if(cycles>1)

			{

				out<<"avg time="<<","<<total_time*1000/(cycles-1)<<std::endl;

			}





			std::cout<<"Resnet50"<<std::endl;

			std::cout << "---conv1       " << "		"<< lend01 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "---relu1       " <<"		"<< lend02* 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "---pooling1    " <<"		"<< lend03 * 1000/(cycles-1) << "ms" << std::endl;



			std::cout<<"---layer1      "<<std::endl;

			std::cout<<"  ---0         "<<std::endl;

			std::cout << "   ---conv1    " << "		"<< lend111 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---relu1    " << "		"<< lend112 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---conv2    " << "		"<< lend113 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---relu2    " << "		"<< lend114 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---conv3    " << "		"<< lend115 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---conv    " << "		"<< lend116 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---add      " << "		"<< lend117 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---relu     " << "		"<< lend118 * 1000/(cycles-1) << "ms" << std::endl;

			

			std::cout<<"  ---1         "<<std::endl;

			std::cout << "   ---conv1    " << "		"<< lend121 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---relu1    " << "		"<< lend122 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---conv2    " << "		"<< lend123 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---relu2    " << "		"<< lend124 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---conv3    " << "		"<< lend125 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---add      " << "		"<< lend126 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---relu     " << "		"<< lend127 * 1000/(cycles-1) << "ms" << std::endl;



			std::cout<<"  ---2         "<<std::endl;

			std::cout << "   ---conv1    " << "		"<< lend131 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---relu1    " << "		"<< lend132 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---conv2    " << "		"<< lend133 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---relu2    " << "		"<< lend134 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---conv3    " << "		"<< lend135 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---add      " << "		"<< lend136 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---relu     " << "		"<< lend137 * 1000/(cycles-1) << "ms" << std::endl;



			std::cout<<"---layer2      "<<std::endl;

			std::cout<<"  ---0         "<<std::endl;

			std::cout << "   ---conv1    " << "		"<< lend211 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---relu1    " << "		"<< lend212 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---conv2    " << "		"<< lend213 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---relu2    " << "		"<< lend214 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---conv3    " << "		"<< lend215 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---conv    " << "		"<< lend216 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---add      " << "		"<< lend217 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---relu     " << "		"<< lend218 * 1000/(cycles-1) << "ms" << std::endl;

			

			std::cout<<"  ---1         "<<std::endl;

			std::cout << "   ---conv1    " << "		"<< lend221 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---relu1    " << "		"<< lend222 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---conv2    " << "		"<< lend223 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---relu2    " << "		"<< lend224 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---conv3    " << "		"<< lend225 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---add      " << "		"<< lend226 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---relu     " << "		"<< lend227 * 1000/(cycles-1) << "ms" << std::endl;



			std::cout<<"  ---2         "<<std::endl;

			std::cout << "   ---conv1    " << "		"<< lend231* 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---relu1    " << "		"<< lend232 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---conv2    " << "		"<< lend233 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---relu2    " << "		"<< lend234 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---conv3    " << "		"<< lend235 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---add      " << "		"<< lend236 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---relu     " << "		"<< lend237 * 1000/(cycles-1) << "ms" << std::endl;



			std::cout<<"  ---3         "<<std::endl;

			std::cout << "   ---conv1    " << "		"<< lend241 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---relu1    " << "		"<< lend242 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---conv2    " << "		"<< lend243 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---relu2    " << "		"<< lend244 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---conv3    " << "		"<< lend245 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---add      " << "		"<< lend246 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---relu     " << "		"<< lend247 * 1000/(cycles-1) << "ms" << std::endl;



			std::cout<<"---layer3      "<<std::endl;

			std::cout<<"  ---0         "<<std::endl;

			std::cout << "   ---conv1    " << "		"<< lend311 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---relu1    " << "		"<< lend312 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---conv2    " << "		"<< lend313 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---relu2    " << "		"<< lend314 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---conv3    " << "		"<< lend315 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---conv    " << "		"<< lend316 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---add      " << "		"<< lend317 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---relu     " << "		"<< lend318 * 1000/(cycles-1) << "ms" << std::endl;

			

			std::cout<<"  ---1         "<<std::endl;

			std::cout << "   ---conv1    " << "		"<< lend321 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---relu1    " << "		"<< lend322 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---conv2    " << "		"<< lend323 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---relu2    " << "		"<< lend324 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---conv3    " << "		"<< lend325 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---add      " << "		"<< lend326 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---relu     " << "		"<< lend327 * 1000/(cycles-1) << "ms" << std::endl;



			std::cout<<"  ---2         "<<std::endl;

			std::cout << "   ---conv1    " << "		"<< lend331 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---relu1    " << "		"<< lend332 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---conv2    " << "		"<< lend333 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---relu2    " << "		"<< lend334 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---conv3    " << "		"<< lend335 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---add      " << "		"<< lend336 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---relu     " << "		"<< lend337 * 1000/(cycles-1) << "ms" << std::endl;



			std::cout<<"  ---3         "<<std::endl;

			std::cout << "   ---conv1    " << "		"<< lend341 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---relu1    " << "		"<< lend342 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---conv2    " << "		"<< lend343 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---relu2    " << "		"<< lend344 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---conv3    " << "		"<< lend345 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---add      " << "		"<< lend346 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---relu     " << "		"<< lend347 * 1000/(cycles-1) << "ms" << std::endl;



			std::cout<<"  ---4         "<<std::endl;

			std::cout << "   ---conv1    " << "		"<< lend351 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---relu1    " << "		"<< lend352 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---conv2    " << "		"<< lend353 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---relu2    " << "		"<< lend354 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---conv3    " << "		"<< lend355 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---add      " << "		"<< lend356 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---relu     " << "		"<< lend357 * 1000/(cycles-1) << "ms" << std::endl;



			std::cout<<"  ---5         "<<std::endl;

			std::cout << "   ---conv1    " << "		"<< lend361 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---relu1    " << "		"<< lend362 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---conv2    " << "		"<< lend363 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---relu2    " << "		"<< lend364 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---conv3    " << "		"<< lend365 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---add      " << "		"<< lend366 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---relu     " << "		"<< lend367 * 1000/(cycles-1) << "ms" << std::endl;



			std::cout<<"---layer4      "<<std::endl;

			std::cout<<"  ---0         "<<std::endl;

			std::cout << "   ---conv1    " << "		"<< lend411 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---relu1    " << "		"<< lend412 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---conv2    " << "		"<< lend413 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---relu2    " << "		"<< lend414 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---conv3    " << "		"<< lend415 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---conv    " << "		"<< lend416 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---add      " << "		"<< lend417 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---relu     " << "		"<< lend418 * 1000/(cycles-1) << "ms" << std::endl;

			

			std::cout<<"  ---1         "<<std::endl;

			std::cout << "   ---conv1    " << "		"<< lend421 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---relu1    " << "		"<< lend422 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---conv2    " << "		"<< lend423 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---relu2    " << "		"<< lend424 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---conv3    " << "		"<< lend425 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---add      " << "		"<< lend426 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---relu     " << "		"<< lend427 * 1000/(cycles-1) << "ms" << std::endl;



			std::cout<<"  ---2         "<<std::endl;

			std::cout << "   ---conv1    " << "		"<< lend431 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---relu1    " << "		"<< lend432 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---conv2    " << "		"<< lend433 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---relu2    " << "		"<< lend434 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---conv3    " << "		"<< lend435 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---add      " << "		"<< lend436 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "   ---relu     " << "		"<< lend437 * 1000/(cycles-1) << "ms" << std::endl;	



			std::cout << "---pooling     " << "		"<< lend11 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "---conv1       " << "		"<<lend12 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "---flatten     "<<"		"<< lend13 * 1000/(cycles-1) << "ms" << std::endl;

			std::cout << "---softmax   " <<"		"<< lend14 * 1000/(cycles-1) << "ms" << std::endl;



			if(cycles>1)

			{

				std::cout<<"avg time="<<total_time*1000/(cycles-1)<<"ms"<<std::endl;

			}



}

private:

	/*precision table*/

	unsigned int  precision[54][4] ={

                        {8, 7, 4, 6                      

                         },/*layer-1*/

                        {7, 7, 6,  5

                         },/*layer-2*/

                        {7, 7, 5, 6

                         },/*layer-3*/

                        {5, 6, 6, 7

                         },/*layer-4*/

                        {7, 7, 6, 5

                        },/*layer-5*/

                        {8, 6, 6, 6

                         },/*layer-6*/

                        {7, 7, 6, 6

                         },/*layer-7*/

                        {7, 6, 6, 6

                        },/*layer-8*/

                        {6, 8, 6, 7

                         },/*layer-9*/

                        {7, 8, 7, 4

                         },/*layer-10*/

                        {4, 7, 4, 4

                       },/*layer-11*/

                        {7, 7, 7, 7

                         },/*layer-12*/

                        {9, 6, 7, 7

                         },/*layer-13*/

                        {7, 6, 7, 6

                         },/*layer-14*/

                        {8, 6, 7, 6

                        },/*layer-15*/

                        {6, 8, 4, 4

                         },/*layer-16*/

                        {7, 9, 4, 5

                         },/*layer-17*/

                        {5, 9, 5, 5

                        },/*layer-18*/

                        {7, 7, 7, 7

                         },/*layer-19*/

                        {6, 8, 7, 4

                         },/*layer-20*/

                        {5, 8, 4, 8

                        },/*layer-21*/

                        {8, 7, 6, 3

                         },/*layer-22*/

                        {6, 8, 3, 6

                         },/*layer-23*/

                        {7, 6, 6, 6

                        },/*layer-24*/

                        {8, 7, 7, 5

                         },/*layer-25*/

                        {10, 6, 5, 5

                         },/*layer-26*/

                        {6, 7, 5, 7

                         },/*layer-27*/

                        {8, 7, 7, 7

                        },/*layer-28*/

                        {9, 8, 5, 4

                         },/*layer-29*/

                        {8, 8, 4, 4

                         },/*layer-30*/

                        {8, 7, 4, 8

                        },/*layer-31*/

                        {8, 9, 6, 6

                         },/*layer-32*/

                        {8, 7, 6, 7

                         },/*layer-33*/

                        {7, 8, 7, 7

                        },/*layer-34*/

                        {7, 9, 5, 5

                         },/*layer-35*/

                        {8, 9, 5, 6

                         },/*layer-36*/

                        {7, 8, 6, 6

                        },/*layer-37*/

                        {7, 9, 5, 5

                         },/*layer-38*/

                        {7, 8, 5, 5

                         },/*layer-39*/

                        {6, 6, 5, 7

                        },/*layer-40*/

                        {9, 9, 6, 6

                         },/*layer-41*/

                        {9, 7, 6, 7

                         },/*layer-42*/

                        {6, 6, 7, 7

                        },/*layer-43*/

                        {8, 9, 6, 6

                         },/*layer-44*/

                        {9, 7, 6, 6

                         },/*layer-45*/

                        {7, 9, 6, 6

                         },/*layer-46*/

                        {8, 8, 6, 6

                        },/*layer-47*/

                        {7, 7, 7, 5

                         },/*layer-48*/

                        {9, 7, 5, 7

                         },/*layer-49*/

                        {8, 6, 7, 5

                     	  },/*layer-50*/

                        {7, 7, 5, 5

                         },/*layer-51*/

                        {9, 8, 5, 7

                         },/*layer-52*/

                        {3, 7, 7, 0

                        },/*layer-53*/

                        {8, 9, 2, 1

                         }};/*layer-54*/





	int fp[16][3]={

		{7,5,6},

		{6,6,6},

		{4,6,7},

		{6,6,4},

		{5,4,7},

		{8,7,6},

		{6,6,7},

		{7,7,5},

		{8,5,6},

		{7,6,5},

		{6,5,5},

		{7,5,6},

		{7,6,6},

		{6,6,7},

		{5,7,5},

		{0,5,2}

	};

	string Q_table_datapath="/media/sdcard/ComputeLibrary/data/neon_resnet50/Q_table/";

	string WT_buffer_datapath="/media/sdcard/ComputeLibrary/data/neon_resnet50/WT_buffer/";

	string bias_datapath="/media/sdcard/ComputeLibrary/data/neon_resnet50/bias/";

	string Q_table_name[54]={

		"1.conv1.weight.Q_table.npy",

		"1.layer1.0.conv1.weight.Q_table.npy",

		"1.layer1.0.conv2.weight.Q_table.npy",

		"1.layer1.0.conv3.weight.Q_table.npy",

		"1.layer1.0.downsample.0.weight.Q_table.npy",

		"1.layer1.1.conv1.weight.Q_table.npy",

		"1.layer1.1.conv2.weight.Q_table.npy",

		"1.layer1.1.conv3.weight.Q_table.npy",

		"1.layer1.2.conv1.weight.Q_table.npy",

		"1.layer1.2.conv2.weight.Q_table.npy",

		"1.layer1.2.conv3.weight.Q_table.npy",

		"1.layer2.0.conv1.weight.Q_table.npy",

		"1.layer2.0.conv2.weight.Q_table.npy",

		"1.layer2.0.conv3.weight.Q_table.npy",

		"1.layer2.0.downsample.0.weight.Q_table.npy",

		"1.layer2.1.conv1.weight.Q_table.npy",

		"1.layer2.1.conv2.weight.Q_table.npy",

		"1.layer2.1.conv3.weight.Q_table.npy",

		"1.layer2.2.conv1.weight.Q_table.npy",

		"1.layer2.2.conv2.weight.Q_table.npy",

		"1.layer2.2.conv3.weight.Q_table.npy",

		"1.layer2.3.conv1.weight.Q_table.npy",

		"1.layer2.3.conv2.weight.Q_table.npy",

		"1.layer2.3.conv3.weight.Q_table.npy",

		"1.layer3.0.conv1.weight.Q_table.npy",

		"1.layer3.0.conv2.weight.Q_table.npy",

		"1.layer3.0.conv3.weight.Q_table.npy",

		"1.layer3.0.downsample.0.weight.Q_table.npy",

		"1.layer3.1.conv1.weight.Q_table.npy",

		"1.layer3.1.conv2.weight.Q_table.npy",

		"1.layer3.1.conv3.weight.Q_table.npy",

		"1.layer3.2.conv1.weight.Q_table.npy",

		"1.layer3.2.conv2.weight.Q_table.npy",

		"1.layer3.2.conv3.weight.Q_table.npy",

		"1.layer3.3.conv1.weight.Q_table.npy",

		"1.layer3.3.conv2.weight.Q_table.npy",

		"1.layer3.3.conv3.weight.Q_table.npy",

		"1.layer3.4.conv1.weight.Q_table.npy",

		"1.layer3.4.conv2.weight.Q_table.npy",

		"1.layer3.4.conv3.weight.Q_table.npy",

		"1.layer3.5.conv1.weight.Q_table.npy",

		"1.layer3.5.conv2.weight.Q_table.npy",

		"1.layer3.5.conv3.weight.Q_table.npy",

		"1.layer4.0.conv1.weight.Q_table.npy",

		"1.layer4.0.conv2.weight.Q_table.npy",

		"1.layer4.0.conv3.weight.Q_table.npy",

		"1.layer4.0.downsample.0.weight.Q_table.npy",

		"1.layer4.1.conv1.weight.Q_table.npy",

		"1.layer4.1.conv2.weight.Q_table.npy",

		"1.layer4.1.conv3.weight.Q_table.npy",

		"1.layer4.2.conv1.weight.Q_table.npy",

		"1.layer4.2.conv2.weight.Q_table.npy",

		"1.layer4.2.conv3.weight.Q_table.npy",

		"1.fc.weight.Q_table.npy"

	};

	string WT_buffer_name[54]={

		"1.conv1.weight.WT_buffer.npy",

		"1.layer1.0.conv1.weight.WT_buffer.npy",

		"1.layer1.0.conv2.weight.WT_buffer.npy",

		"1.layer1.0.conv3.weight.WT_buffer.npy",

		"1.layer1.0.downsample.0.weight.WT_buffer.npy",

		"1.layer1.1.conv1.weight.WT_buffer.npy",

		"1.layer1.1.conv2.weight.WT_buffer.npy",

		"1.layer1.1.conv3.weight.WT_buffer.npy",

		"1.layer1.2.conv1.weight.WT_buffer.npy",

		"1.layer1.2.conv2.weight.WT_buffer.npy",

		"1.layer1.2.conv3.weight.WT_buffer.npy",

		"1.layer2.0.conv1.weight.WT_buffer.npy",

		"1.layer2.0.conv2.weight.WT_buffer.npy",

		"1.layer2.0.conv3.weight.WT_buffer.npy",

		"1.layer2.0.downsample.0.weight.WT_buffer.npy",

		"1.layer2.1.conv1.weight.WT_buffer.npy",

		"1.layer2.1.conv2.weight.WT_buffer.npy",

		"1.layer2.1.conv3.weight.WT_buffer.npy",

		"1.layer2.2.conv1.weight.WT_buffer.npy",

		"1.layer2.2.conv2.weight.WT_buffer.npy",

		"1.layer2.2.conv3.weight.WT_buffer.npy",

		"1.layer2.3.conv1.weight.WT_buffer.npy",

		"1.layer2.3.conv2.weight.WT_buffer.npy",

		"1.layer2.3.conv3.weight.WT_buffer.npy",

		"1.layer3.0.conv1.weight.WT_buffer.npy",

		"1.layer3.0.conv2.weight.WT_buffer.npy",

		"1.layer3.0.conv3.weight.WT_buffer.npy",

		"1.layer3.0.downsample.0.weight.WT_buffer.npy",

		"1.layer3.1.conv1.weight.WT_buffer.npy",

		"1.layer3.1.conv2.weight.WT_buffer.npy",

		"1.layer3.1.conv3.weight.WT_buffer.npy",

		"1.layer3.2.conv1.weight.WT_buffer.npy",

		"1.layer3.2.conv2.weight.WT_buffer.npy",

		"1.layer3.2.conv3.weight.WT_buffer.npy",

		"1.layer3.3.conv1.weight.WT_buffer.npy",

		"1.layer3.3.conv2.weight.WT_buffer.npy",

		"1.layer3.3.conv3.weight.WT_buffer.npy",

		"1.layer3.4.conv1.weight.WT_buffer.npy",

		"1.layer3.4.conv2.weight.WT_buffer.npy",

		"1.layer3.4.conv3.weight.WT_buffer.npy",

		"1.layer3.5.conv1.weight.WT_buffer.npy",

		"1.layer3.5.conv2.weight.WT_buffer.npy",

		"1.layer3.5.conv3.weight.WT_buffer.npy",

		"1.layer4.0.conv1.weight.WT_buffer.npy",

		"1.layer4.0.conv2.weight.WT_buffer.npy",

		"1.layer4.0.conv3.weight.WT_buffer.npy",

		"1.layer4.0.downsample.0.weight.WT_buffer.npy",

		"1.layer4.1.conv1.weight.WT_buffer.npy",

		"1.layer4.1.conv2.weight.WT_buffer.npy",

		"1.layer4.1.conv3.weight.WT_buffer.npy",

		"1.layer4.2.conv1.weight.WT_buffer.npy",

		"1.layer4.2.conv2.weight.WT_buffer.npy",

		"1.layer4.2.conv3.weight.WT_buffer.npy",

		"1.fc.weight.WT_buffer.npy"

	};

	string bias_name[54]={

		"1.conv1.bias.npy",

		"1.layer1.0.conv1.bias.npy",

		"1.layer1.0.conv2.bias.npy",

		"1.layer1.0.conv3.bias.npy",

		"1.layer1.0.downsample.0.bias.npy",

		"1.layer1.1.conv1.bias.npy",

		"1.layer1.1.conv2.bias.npy",

		"1.layer1.1.conv3.bias.npy",

		"1.layer1.2.conv1.bias.npy",

		"1.layer1.2.conv2.bias.npy",

		"1.layer1.2.conv3.bias.npy",

		"1.layer2.0.conv1.bias.npy",

		"1.layer2.0.conv2.bias.npy",

		"1.layer2.0.conv3.bias.npy",

		"1.layer2.0.downsample.0.bias.npy",

		"1.layer2.1.conv1.bias.npy",

		"1.layer2.1.conv2.bias.npy",

		"1.layer2.1.conv3.bias.npy",

		"1.layer2.2.conv1.bias.npy",

		"1.layer2.2.conv2.bias.npy",

		"1.layer2.2.conv3.bias.npy",

		"1.layer2.3.conv1.bias.npy",

		"1.layer2.3.conv2.bias.npy",

		"1.layer2.3.conv3.bias.npy",

		"1.layer3.0.conv1.bias.npy",

		"1.layer3.0.conv2.bias.npy",

		"1.layer3.0.conv3.bias.npy",

		"1.layer3.0.downsample.0.bias.npy",

		"1.layer3.1.conv1.bias.npy",

		"1.layer3.1.conv2.bias.npy",

		"1.layer3.1.conv3.bias.npy",

		"1.layer3.2.conv1.bias.npy",

		"1.layer3.2.conv2.bias.npy",

		"1.layer3.2.conv3.bias.npy",

		"1.layer3.3.conv1.bias.npy",

		"1.layer3.3.conv2.bias.npy",

		"1.layer3.3.conv3.bias.npy",

		"1.layer3.4.conv1.bias.npy",

		"1.layer3.4.conv2.bias.npy",

		"1.layer3.4.conv3.bias.npy",

		"1.layer3.5.conv1.bias.npy",

		"1.layer3.5.conv2.bias.npy",

		"1.layer3.5.conv3.bias.npy",

		"1.layer4.0.conv1.bias.npy",

		"1.layer4.0.conv2.bias.npy",

		"1.layer4.0.conv3.bias.npy",

		"1.layer4.0.downsample.0.bias.npy",

		"1.layer4.1.conv1.bias.npy",

		"1.layer4.1.conv2.bias.npy",

		"1.layer4.1.conv3.bias.npy",

		"1.layer4.2.conv1.bias.npy",

		"1.layer4.2.conv2.bias.npy",

		"1.layer4.2.conv3.bias.npy",

		"1.fc.bias.npy"

	};

	/*Tensor*/

		bool is_fortran{};

		string output_filename="/media/sdcard/ComputeLibrary/data/resnet50_resnet50_float_output/output.npy";

		string output_filename1="/media/sdcard/ComputeLibrary/data/resnet50_resnet50_float_output/output1.npy";

		string output_filename2="/media/sdcard/ComputeLibrary/data/resnet50_resnet50_float_output/output2.npy";

		string output_filename3="/media/sdcard/ComputeLibrary/data/resnet50_resnet50_float_output/output3.npy";

		Tensor src{}; Tensor Q_table_con0{};Tensor WT_buffer_con0{};Tensor bias_con0{};



		Tensor Q_table_block1r_con0{}; Tensor Q_table_block1r_con1{}; Tensor Q_table_block1r_con2{};

		Tensor Q_table_block1r_con3{}; Tensor Q_table_block1r_con4{}; Tensor Q_table_block1r_con5{};

		Tensor Q_table_block1r_con6{}; Tensor Q_table_block1r_con7{}; Tensor Q_table_block1r_con8{};

		Tensor Q_table_block1l_con0{}; 

		Tensor WT_buffer_block1r_con0{}; Tensor WT_buffer_block1r_con1{}; Tensor WT_buffer_block1r_con2{};

		Tensor WT_buffer_block1r_con3{}; Tensor WT_buffer_block1r_con4{}; Tensor WT_buffer_block1r_con5{};

		Tensor WT_buffer_block1r_con6{}; Tensor WT_buffer_block1r_con7{}; Tensor WT_buffer_block1r_con8{};

		Tensor WT_buffer_block1l_con0{}; 



		Tensor Q_table_block2r_con0{}; Tensor Q_table_block2r_con1{}; Tensor Q_table_block2r_con2{};

		Tensor Q_table_block2r_con3{}; Tensor Q_table_block2r_con4{}; Tensor Q_table_block2r_con5{};

		Tensor Q_table_block2r_con6{}; Tensor Q_table_block2r_con7{}; Tensor Q_table_block2r_con8{};

		Tensor Q_table_block2r_con9{}; Tensor Q_table_block2r_con10{}; Tensor Q_table_block2r_con11{};

		Tensor Q_table_block2l_con0{}; 

		Tensor WT_buffer_block2r_con0{}; Tensor WT_buffer_block2r_con1{}; Tensor WT_buffer_block2r_con2{};

		Tensor WT_buffer_block2r_con3{}; Tensor WT_buffer_block2r_con4{}; Tensor WT_buffer_block2r_con5{};

		Tensor WT_buffer_block2r_con6{}; Tensor WT_buffer_block2r_con7{}; Tensor WT_buffer_block2r_con8{};

		Tensor WT_buffer_block2r_con9{}; Tensor WT_buffer_block2r_con10{}; Tensor WT_buffer_block2r_con11{};

		Tensor WT_buffer_block2l_con0{}; 



		Tensor Q_table_block3r_con0{}; Tensor Q_table_block3r_con1{}; Tensor Q_table_block3r_con2{};

		Tensor Q_table_block3r_con3{}; Tensor Q_table_block3r_con4{}; Tensor Q_table_block3r_con5{};

		Tensor Q_table_block3r_con6{}; Tensor Q_table_block3r_con7{}; Tensor Q_table_block3r_con8{};

		Tensor Q_table_block3r_con9{}; Tensor Q_table_block3r_con10{}; Tensor Q_table_block3r_con11{};

		Tensor Q_table_block3r_con12{}; Tensor Q_table_block3r_con13{}; Tensor Q_table_block3r_con14{};

		Tensor Q_table_block3r_con15{}; Tensor Q_table_block3r_con16{}; Tensor Q_table_block3r_con17{};

		Tensor Q_table_block3l_con0{}; 

		Tensor WT_buffer_block3r_con0{}; Tensor WT_buffer_block3r_con1{}; Tensor WT_buffer_block3r_con2{};

		Tensor WT_buffer_block3r_con3{}; Tensor WT_buffer_block3r_con4{}; Tensor WT_buffer_block3r_con5{};

		Tensor WT_buffer_block3r_con6{}; Tensor WT_buffer_block3r_con7{}; Tensor WT_buffer_block3r_con8{};

		Tensor WT_buffer_block3r_con9{}; Tensor WT_buffer_block3r_con10{}; Tensor WT_buffer_block3r_con11{};

		Tensor WT_buffer_block3r_con12{}; Tensor WT_buffer_block3r_con13{}; Tensor WT_buffer_block3r_con14{};

		Tensor WT_buffer_block3r_con15{}; Tensor WT_buffer_block3r_con16{}; Tensor WT_buffer_block3r_con17{};

		Tensor WT_buffer_block3l_con0{}; 



		Tensor Q_table_block4r_con0{}; Tensor Q_table_block4r_con1{}; Tensor Q_table_block4r_con2{};

		Tensor Q_table_block4r_con3{}; Tensor Q_table_block4r_con4{}; Tensor Q_table_block4r_con5{};

		Tensor Q_table_block4r_con6{}; Tensor Q_table_block4r_con7{}; Tensor Q_table_block4r_con8{};

		Tensor Q_table_block4l_con0{}; 

		Tensor WT_buffer_block4r_con0{}; Tensor WT_buffer_block4r_con1{}; Tensor WT_buffer_block4r_con2{};

		Tensor WT_buffer_block4r_con3{}; Tensor WT_buffer_block4r_con4{}; Tensor WT_buffer_block4r_con5{};

		Tensor WT_buffer_block4r_con6{}; Tensor WT_buffer_block4r_con7{}; Tensor WT_buffer_block4r_con8{};

		Tensor WT_buffer_block4l_con0{}; 



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

		

		/*Tensor input_fc{};*/

		Tensor Q_table_con1{};Tensor WT_buffer_con1{};Tensor  bias_con1{};









		Tensor out_con0{}; 

		Tensor out_act0{}; Tensor out_pool0{};

		/*block1*/

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

		/*block2*/

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

		/*block3*/

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



		/*block4*/

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





		/*type change tensor*/

		Tensor conv0sf{}; Tensor pool0fs{};



		Tensor b1rconv0sf{}; Tensor b1ract0fs{}; Tensor b1rconv1sf{}; Tensor b1ract1fs{}; Tensor b1add0sf{};

		Tensor b1act0fs{}; Tensor b1rconv3sf{}; Tensor b1ract2fs{}; Tensor b1rconv4sf{}; Tensor b1ract3fs{}; Tensor b1add1sf{};

		Tensor b1act1fs{}; Tensor b1rconv6sf{}; Tensor b1ract4fs{}; Tensor b1rconv7sf{}; Tensor b1ract5fs{}; Tensor b1add2sf{};



		Tensor b1act2fs{};Tensor b2rconv0sf{}; Tensor b2ract0fs{}; Tensor b2rconv1sf{}; Tensor b2ract1fs{}; Tensor b2add0sf{};

		Tensor b2act0fs{}; Tensor b2rconv3sf{}; Tensor b2ract2fs{}; Tensor b2rconv4sf{}; Tensor b2ract3fs{}; Tensor b2add1sf{};

		Tensor b2act1fs{}; Tensor b2rconv6sf{}; Tensor b2ract4fs{}; Tensor b2rconv7sf{}; Tensor b2ract5fs{}; Tensor b2add2sf{};

		Tensor b2act2fs{}; Tensor b2rconv9sf{}; Tensor b2ract6fs{}; Tensor b2rconv10sf{}; Tensor b2ract7fs{}; Tensor b2add3sf{};



		Tensor b2act3fs{};Tensor b3rconv0sf{}; Tensor b3ract0fs{}; Tensor b3rconv1sf{}; Tensor b3ract1fs{}; Tensor b3add0sf{};

		Tensor b3act0fs{}; Tensor b3rconv3sf{}; Tensor b3ract2fs{}; Tensor b3rconv4sf{}; Tensor b3ract3fs{}; Tensor b3add1sf{};

		Tensor b3act1fs{}; Tensor b3rconv6sf{}; Tensor b3ract4fs{}; Tensor b3rconv7sf{}; Tensor b3ract5fs{}; Tensor b3add2sf{};

		Tensor b3act2fs{}; Tensor b3rconv9sf{}; Tensor b3ract6fs{}; Tensor b3rconv10sf{}; Tensor b3ract7fs{}; Tensor b3add3sf{};

		Tensor b3act3fs{}; Tensor b3rconv12sf{}; Tensor b3ract8fs{}; Tensor b3rconv13sf{}; Tensor b3ract9fs{}; Tensor b3add4sf{};

		Tensor b3act4fs{}; Tensor b3rconv15sf{}; Tensor b3ract10fs{}; Tensor b3rconv16sf{}; Tensor b3ract11fs{}; Tensor b3add5sf{};



		Tensor b3act5fs{};Tensor b4rconv0sf{}; Tensor b4ract0fs{}; Tensor b4rconv1sf{}; Tensor b4ract1fs{}; Tensor b4add0sf{};

		Tensor b4act0fs{}; Tensor b4rconv3sf{}; Tensor b4ract2fs{}; Tensor b4rconv4sf{}; Tensor b4ract3fs{};Tensor b4add1sf{};

		Tensor b4act1fs{}; Tensor b4rconv6sf{}; Tensor b4ract4fs{}; Tensor b4rconv7sf{}; Tensor b4ract5fs{}; Tensor b4add2sf{};



		Tensor pool1fs{}; Tensor conv1sf{};



		Tensor out_resize{};







		/*Layer*/

		NEABMConvolutionLayer con0{}; NEActivationLayer act0{}; NEPoolingLayer pool0{};

		/*block1*/

		NEABMConvolutionLayer  block1r_con0{};   NEActivationLayer  block1r_act0{};

		NEABMConvolutionLayer  block1r_con1{};  NEActivationLayer  block1r_act1{};

		NEABMConvolutionLayer  block1r_con2{};   NEABMConvolutionLayer block1l_con0{}; 

		NEFPAdditionLayer  block1_add0{}; NEActivationLayer  block1_act0{};





		NEABMConvolutionLayer  block1r_con3{}; NEActivationLayer  block1r_act2{};

		NEABMConvolutionLayer  block1r_con4{};   NEActivationLayer  block1r_act3{};

		NEABMConvolutionLayer  block1r_con5{}; 

		NEFPAdditionLayer  block1_add1{}; NEActivationLayer  block1_act1{};



		NEABMConvolutionLayer  block1r_con6{}; NEActivationLayer  block1r_act4{};

		NEABMConvolutionLayer  block1r_con7{};   NEActivationLayer  block1r_act5{};

		NEABMConvolutionLayer  block1r_con8{};  

		NEFPAdditionLayer  block1_add2{}; NEActivationLayer  block1_act2{};

		/*block2*/

		NEABMConvolutionLayer  block2r_con0{};  NEActivationLayer  block2r_act0{};

		NEABMConvolutionLayer  block2r_con1{};  NEActivationLayer  block2r_act1{};

		NEABMConvolutionLayer  block2r_con2{};  NEABMConvolutionLayer block2l_con0{}; 

		NEFPAdditionLayer  block2_add0{}; NEActivationLayer  block2_act0{};



		NEABMConvolutionLayer  block2r_con3{};   NEActivationLayer  block2r_act2{};

		NEABMConvolutionLayer  block2r_con4{};  NEActivationLayer  block2r_act3{};

		NEABMConvolutionLayer  block2r_con5{}; 

		NEFPAdditionLayer  block2_add1{}; NEActivationLayer  block2_act1{};



		NEABMConvolutionLayer  block2r_con6{}; NEActivationLayer  block2r_act4{};

		NEABMConvolutionLayer  block2r_con7{};  NEActivationLayer  block2r_act5{};

		NEABMConvolutionLayer  block2r_con8{}; 

		NEFPAdditionLayer  block2_add2{}; NEActivationLayer  block2_act2{};



		NEABMConvolutionLayer  block2r_con9{};   NEActivationLayer  block2r_act6{};

		NEABMConvolutionLayer  block2r_con10{}; NEActivationLayer  block2r_act7{};

		NEABMConvolutionLayer  block2r_con11{};  

		NEFPAdditionLayer  block2_add3{}; NEActivationLayer  block2_act3{};

		/*block3*/

		NEABMConvolutionLayer  block3r_con0{};  NEActivationLayer  block3r_act0{};

		NEABMConvolutionLayer  block3r_con1{};  NEActivationLayer  block3r_act1{};

		NEABMConvolutionLayer  block3r_con2{};  NEABMConvolutionLayer block3l_con0{}; 

		NEFPAdditionLayer  block3_add0{}; NEActivationLayer  block3_act0{};



		NEABMConvolutionLayer  block3r_con3{}; NEActivationLayer  block3r_act2{};

		NEABMConvolutionLayer  block3r_con4{};   NEActivationLayer  block3r_act3{};

		NEABMConvolutionLayer  block3r_con5{}; 

		NEFPAdditionLayer  block3_add1{}; NEActivationLayer  block3_act1{};



		NEABMConvolutionLayer  block3r_con6{};NEActivationLayer  block3r_act4{};

		NEABMConvolutionLayer  block3r_con7{};  NEActivationLayer  block3r_act5{};

		NEABMConvolutionLayer  block3r_con8{};  

		NEFPAdditionLayer  block3_add2{}; NEActivationLayer  block3_act2{};



		NEABMConvolutionLayer  block3r_con9{};  NEActivationLayer  block3r_act6{};

		NEABMConvolutionLayer  block3r_con10{}; NEActivationLayer  block3r_act7{};

		NEABMConvolutionLayer  block3r_con11{}; 

		NEFPAdditionLayer  block3_add3{}; NEActivationLayer  block3_act3{};



		NEABMConvolutionLayer  block3r_con12{};  NEActivationLayer  block3r_act8{};

		NEABMConvolutionLayer  block3r_con13{}; NEActivationLayer  block3r_act9{};

		NEABMConvolutionLayer  block3r_con14{}; 

		NEFPAdditionLayer  block3_add4{}; NEActivationLayer  block3_act4{};



		NEABMConvolutionLayer  block3r_con15{};  NEActivationLayer  block3r_act10{};

		NEABMConvolutionLayer  block3r_con16{};   NEActivationLayer  block3r_act11{};

		NEABMConvolutionLayer  block3r_con17{}; 

		NEFPAdditionLayer  block3_add5{}; NEActivationLayer  block3_act5{};

		/*block4*/

		NEABMConvolutionLayer  block4r_con0{};  NEActivationLayer  block4r_act0{};

		NEABMConvolutionLayer  block4r_con1{}; NEActivationLayer  block4r_act1{};

		NEABMConvolutionLayer  block4r_con2{}; NEABMConvolutionLayer block4l_con0{}; 

		NEFPAdditionLayer  block4_add0{}; NEActivationLayer  block4_act0{};



		NEABMConvolutionLayer  block4r_con3{}; NEActivationLayer  block4r_act2{};

		NEABMConvolutionLayer  block4r_con4{}; NEActivationLayer  block4r_act3{};

		NEABMConvolutionLayer  block4r_con5{};

		NEFPAdditionLayer  block4_add1{}; NEActivationLayer  block4_act1{};





		NEABMConvolutionLayer  block4r_con6{};  NEActivationLayer  block4r_act4{};

		NEABMConvolutionLayer  block4r_con7{};NEActivationLayer  block4r_act5{};

		NEABMConvolutionLayer  block4r_con8{}; 

		NEFPAdditionLayer  block4_add2{}; NEActivationLayer  block4_act2{};





		NEPoolingLayer pool1{}; NEABMConvolutionLayer con1{}; NEFlattenLayer flatten{}; NESoftmaxLayer softmax{};



		NES8toF32Layer lconv0sf{}; NEF32toS8Layer lpool0fs{};



		NES8toF32Layer lb1rconv0sf{}; NEF32toS8Layer lb1ract0fs{}; NES8toF32Layer lb1rconv1sf{}; NEF32toS8Layer lb1ract1fs{};NES8toF32Layer lb1add0sf{};

		NEF32toS8Layer lb1act0fs{}; NES8toF32Layer lb1rconv3sf{}; NEF32toS8Layer lb1ract2fs{}; NES8toF32Layer lb1rconv4sf{}; NEF32toS8Layer lb1ract3fs{}; NES8toF32Layer lb1add1sf{};

		NEF32toS8Layer lb1act1fs{}; NES8toF32Layer lb1rconv6sf{}; NEF32toS8Layer lb1ract4fs{}; NES8toF32Layer lb1rconv7sf{}; NEF32toS8Layer lb1ract5fs{}; NES8toF32Layer lb1add2sf{};



		NEF32toS8Layer lb1act2fs{};NES8toF32Layer lb2rconv0sf{}; NEF32toS8Layer lb2ract0fs{}; NES8toF32Layer lb2rconv1sf{}; NEF32toS8Layer lb2ract1fs{}; NES8toF32Layer lb2add0sf{};

		NEF32toS8Layer lb2act0fs{}; NES8toF32Layer lb2rconv3sf{}; NEF32toS8Layer lb2ract2fs{}; NES8toF32Layer lb2rconv4sf{}; NEF32toS8Layer lb2ract3fs{}; NES8toF32Layer lb2add1sf{};

		NEF32toS8Layer lb2act1fs{}; NES8toF32Layer lb2rconv6sf{}; NEF32toS8Layer lb2ract4fs{}; NES8toF32Layer lb2rconv7sf{}; NEF32toS8Layer lb2ract5fs{}; NES8toF32Layer lb2add2sf{};

		NEF32toS8Layer lb2act2fs{}; NES8toF32Layer lb2rconv9sf{}; NEF32toS8Layer lb2ract6fs{}; NES8toF32Layer lb2rconv10sf{}; NEF32toS8Layer lb2ract7fs{};NES8toF32Layer lb2add3sf{};



		NEF32toS8Layer lb2act3fs{};NES8toF32Layer lb3rconv0sf{}; NEF32toS8Layer lb3ract0fs{}; NES8toF32Layer lb3rconv1sf{}; NEF32toS8Layer lb3ract1fs{}; NES8toF32Layer lb3add0sf{};

		NEF32toS8Layer lb3act0fs{}; NES8toF32Layer lb3rconv3sf{}; NEF32toS8Layer lb3ract2fs{}; NES8toF32Layer lb3rconv4sf{}; NEF32toS8Layer lb3ract3fs{}; NES8toF32Layer lb3add1sf{};

		NEF32toS8Layer lb3act1fs{}; NES8toF32Layer lb3rconv6sf{}; NEF32toS8Layer lb3ract4fs{}; NES8toF32Layer lb3rconv7sf{}; NEF32toS8Layer lb3ract5fs{};NES8toF32Layer lb3add2sf{};

		NEF32toS8Layer lb3act2fs{}; NES8toF32Layer lb3rconv9sf{}; NEF32toS8Layer lb3ract6fs{}; NES8toF32Layer lb3rconv10sf{}; NEF32toS8Layer lb3ract7fs{};NES8toF32Layer lb3add3sf{};

		NEF32toS8Layer lb3act3fs{}; NES8toF32Layer lb3rconv12sf{}; NEF32toS8Layer lb3ract8fs{}; NES8toF32Layer lb3rconv13sf{}; NEF32toS8Layer lb3ract9fs{}; NES8toF32Layer lb3add4sf{};

		NEF32toS8Layer lb3act4fs{}; NES8toF32Layer lb3rconv15sf{}; NEF32toS8Layer lb3ract10fs{}; NES8toF32Layer lb3rconv16sf{}; NEF32toS8Layer lb3ract11fs{};NES8toF32Layer lb3add5sf{};



		NEF32toS8Layer lb3act5fs{};NES8toF32Layer lb4rconv0sf{}; NEF32toS8Layer lb4ract0fs{}; NES8toF32Layer lb4rconv1sf{}; NEF32toS8Layer lb4ract1fs{};NES8toF32Layer lb4add0sf{};

		NEF32toS8Layer lb4act0fs{}; NES8toF32Layer lb4rconv3sf{}; NEF32toS8Layer lb4ract2fs{}; NES8toF32Layer lb4rconv4sf{}; NEF32toS8Layer lb4ract3fs{}; NES8toF32Layer lb4add1sf{};

		NEF32toS8Layer lb4act1fs{}; NES8toF32Layer lb4rconv6sf{}; NEF32toS8Layer lb4ract4fs{}; NES8toF32Layer lb4rconv7sf{}; NEF32toS8Layer lb4ract5fs{}; NES8toF32Layer lb4add2sf{};



		NEF32toS8Layer lpool1fs{}; NES8toF32Layer lconv1sf{};



		NEResizeLayer resize{};



		ConvertPolicy A{};



		double im2col_kernel_time=0;

		double interleave_kernel_time=0;

		double matrix_multiply_kernel_time=0;

		double mmlast_kernel_time=0;

		double col2im_kernel_time=0;



};/*end of class*/

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


