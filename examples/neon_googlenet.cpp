/*GoogleNet Accuracy May Be Not Correct!
    1  Precision Array
    2  Before pooling layers, we directly apply data-type change layers.
*/
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

class NEONGOOGLENETExample : public Example
{
public:
    bool do_setup(int argc, char **argv) override
    {
        const int  size2=112, size3=56, size4=28, size5=14, size6=7, size7=1;
        string data_path="/media/sdcard/ComputeLibrary/data/neon_googlenet/";
        NPYLoader npy_input; npy_input.open(data_path+"input.npy");npy_input.init_tensor2(src, DataType::S8);
/*************Pre-layers*************************/
        NPYLoader npy_1_conv0_q, npy_1_conv0_wt, npy_1_conv0_b;
        npy_1_conv0_q.open(Q_table_datapath+Q_table_name[0]);npy_1_conv0_q.init_tensor2(Q_table_1_conv0, DataType::S16);
        npy_1_conv0_wt.open(WT_buffer_datapath+WT_buffer_name[0]);npy_1_conv0_wt.init_tensor2(WT_buffer_1_conv0, DataType::U16);
        npy_1_conv0_b.open(bias_datapath+bias_name[0]);npy_1_conv0_b.init_tensor2(bias_1_conv0, DataType::S8);
        WeightsInfo weights_1_conv0(false, 7,7,64, false);
        const TensorShape out_1_conv0_shape(size2, size2, weights_1_conv0.num_kernels());
        out_1_conv0.allocator()->init(TensorInfo(out_1_conv0_shape, 1, DataType::S8));
        TensorShape out_1_pool0_shape=out_1_conv0_shape;
        out_1_pool0_shape.set(0, out_1_pool0_shape.x()/2);
        out_1_pool0_shape.set(1, out_1_pool0_shape.y()/2);
        out_1_pool0.allocator()->init(TensorInfo(out_1_pool0_shape, 1, DataType::F32));
        NPYLoader npy_1_conv1_q, npy_1_conv1_wt, npy_1_conv1_b;
        npy_1_conv1_q.open(Q_table_datapath+Q_table_name[1]);npy_1_conv1_q.init_tensor2(Q_table_1_conv1, DataType::S16);
        npy_1_conv1_wt.open(WT_buffer_datapath+WT_buffer_name[1]);npy_1_conv1_wt.init_tensor2(WT_buffer_1_conv1, DataType::U16);
        npy_1_conv1_b.open(bias_datapath+bias_name[1]);npy_1_conv1_b.init_tensor2(bias_1_conv1, DataType::S8);
        WeightsInfo weights_1_conv1(false, 1,1,64, false);
        const TensorShape out_1_conv1_shape(size3, size3, weights_1_conv1.num_kernels());
        out_1_conv1.allocator()->init(TensorInfo(out_1_conv1_shape, 1, DataType::S8));
        NPYLoader npy_1_conv2_q, npy_1_conv2_wt, npy_1_conv2_b;
        npy_1_conv2_q.open(Q_table_datapath+Q_table_name[2]);npy_1_conv2_q.init_tensor2(Q_table_1_conv2, DataType::S16);
        npy_1_conv2_wt.open(WT_buffer_datapath+WT_buffer_name[2]);npy_1_conv2_wt.init_tensor2(WT_buffer_1_conv2, DataType::U16);
        npy_1_conv2_b.open(bias_datapath+bias_name[2]);npy_1_conv2_b.init_tensor2(bias_1_conv2, DataType::S8);
        WeightsInfo weights_1_conv2(false, 3,3,192, false);
        const TensorShape out_1_conv2_shape(size3, size3, weights_1_conv2.num_kernels());
        out_1_conv2.allocator()->init(TensorInfo(out_1_conv2_shape, 1, DataType::S8));
        out_1_act0.allocator()->init(TensorInfo(out_1_conv2_shape, 1, DataType::S8));
        TensorShape out_1_pool1_shape=out_1_conv2_shape;
        out_1_pool1_shape.set(0, out_1_pool1_shape.x()/2);
        out_1_pool1_shape.set(1, out_1_pool1_shape.y()/2);
        out_1_pool1.allocator()->init(TensorInfo(out_1_pool1_shape, 1, DataType::F32));
        
/**********************************************a3*******************/
        NPYLoader npy_2_conv0_q, npy_2_conv0_wt, npy_2_conv0_b;
        npy_2_conv0_q.open(Q_table_datapath+Q_table_name[3]); npy_2_conv0_q.init_tensor2(Q_table_2_conv0, DataType::S16);
        npy_2_conv0_wt.open(Q_table_datapath+Q_table_name[3]); npy_2_conv0_wt.init_tensor2(WT_buffer_2_conv0, DataType::U16);
        npy_2_conv0_b.open(bias_datapath+bias_name[3]); npy_2_conv0_b.init_tensor2(bias_2_conv0, DataType::S8);
        WeightsInfo weights_2_conv0(false, 1, 1, 64, false);
        const TensorShape out_2_conv0_shape(size4, size4, weights_2_conv0.num_kernels());
        out_2_conv0.allocator()->init(TensorInfo(out_2_conv0_shape, 1, DataType::S8));
        out_2_act0.allocator()->init(TensorInfo(out_2_conv0_shape, 1, DataType::S8));
                NPYLoader npy_2_conv1_q, npy_2_conv1_wt, npy_2_conv1_b;
        npy_2_conv1_q.open(Q_table_datapath+Q_table_name[4]); npy_2_conv1_q.init_tensor2(Q_table_2_conv1, DataType::S16);
        npy_2_conv1_wt.open(Q_table_datapath+Q_table_name[4]); npy_2_conv1_wt.init_tensor2(WT_buffer_2_conv1, DataType::U16);
        npy_2_conv1_b.open(bias_datapath+bias_name[4]); npy_2_conv1_b.init_tensor2(bias_2_conv1, DataType::S8);
        WeightsInfo weights_2_conv1(false, 1, 1, 64, false);
        const TensorShape out_2_conv1_shape(size4, size4, weights_2_conv1.num_kernels());
        out_2_conv1.allocator()->init(TensorInfo(out_2_conv1_shape, 1, DataType::S8));
        out_2_act1.allocator()->init(TensorInfo(out_2_conv1_shape, 1, DataType::S8));
                NPYLoader npy_2_conv2_q, npy_2_conv2_wt, npy_2_conv2_b;
        npy_2_conv2_q.open(Q_table_datapath+Q_table_name[5]); npy_2_conv2_q.init_tensor2(Q_table_2_conv2, DataType::S16);
        npy_2_conv2_wt.open(Q_table_datapath+Q_table_name[5]); npy_2_conv2_wt.init_tensor2(WT_buffer_2_conv2, DataType::U16);
        npy_2_conv2_b.open(bias_datapath+bias_name[5]); npy_2_conv2_b.init_tensor2(bias_2_conv2, DataType::S8);
        WeightsInfo weights_2_conv2(false, 3, 3, 64, false);
        const TensorShape out_2_conv2_shape(size4, size4, weights_2_conv2.num_kernels());
        out_2_conv2.allocator()->init(TensorInfo(out_2_conv2_shape, 1, DataType::S8));
        out_2_act2.allocator()->init(TensorInfo(out_2_conv2_shape, 1, DataType::S8));
                NPYLoader npy_2_conv3_q, npy_2_conv3_wt, npy_2_conv3_b;
        npy_2_conv3_q.open(Q_table_datapath+Q_table_name[6]); npy_2_conv3_q.init_tensor2(Q_table_2_conv3, DataType::S16);
        npy_2_conv3_wt.open(Q_table_datapath+Q_table_name[6]); npy_2_conv3_wt.init_tensor2(WT_buffer_2_conv3, DataType::U16);
        npy_2_conv3_b.open(bias_datapath+bias_name[6]); npy_2_conv3_b.init_tensor2(bias_2_conv3, DataType::S8);
        WeightsInfo weights_2_conv3(false, 1, 1, 64, false);
        const TensorShape out_2_conv3_shape(size4, size4, weights_2_conv3.num_kernels());
        out_2_conv3.allocator()->init(TensorInfo(out_2_conv3_shape, 1, DataType::S8));
        out_2_act3.allocator()->init(TensorInfo(out_2_conv3_shape, 1, DataType::S8));
                NPYLoader npy_2_conv4_q, npy_2_conv4_wt, npy_2_conv4_b;
        npy_2_conv4_q.open(Q_table_datapath+Q_table_name[7]); npy_2_conv4_q.init_tensor2(Q_table_2_conv4, DataType::S16);
        npy_2_conv4_wt.open(Q_table_datapath+Q_table_name[7]); npy_2_conv4_wt.init_tensor2(WT_buffer_2_conv4, DataType::U16);
        npy_2_conv4_b.open(bias_datapath+bias_name[7]); npy_2_conv4_b.init_tensor2(bias_2_conv4, DataType::S8);
        WeightsInfo weights_2_conv4(false, 3, 3, 96, false);
        const TensorShape out_2_conv4_shape(size4, size4, weights_2_conv4.num_kernels());
        out_2_conv4.allocator()->init(TensorInfo(out_2_conv4_shape, 1, DataType::S8));
        out_2_act4.allocator()->init(TensorInfo(out_2_conv4_shape, 1, DataType::S8));
                NPYLoader npy_2_conv5_q, npy_2_conv5_wt, npy_2_conv5_b;
        npy_2_conv5_q.open(Q_table_datapath+Q_table_name[8]); npy_2_conv5_q.init_tensor2(Q_table_2_conv5, DataType::S16);
        npy_2_conv5_wt.open(Q_table_datapath+Q_table_name[8]); npy_2_conv5_wt.init_tensor2(WT_buffer_2_conv5, DataType::U16);
        npy_2_conv5_b.open(bias_datapath+bias_name[8]); npy_2_conv5_b.init_tensor2(bias_2_conv5, DataType::S8);
        WeightsInfo weights_2_conv5(false, 3, 3, 96, false);
        const TensorShape out_2_conv5_shape(size4, size4, weights_2_conv5.num_kernels());
        out_2_conv5.allocator()->init(TensorInfo(out_2_conv5_shape, 1, DataType::S8));
        out_2_act5.allocator()->init(TensorInfo(out_2_conv5_shape, 1, DataType::S8));
                NPYLoader npy_2_conv6_q, npy_2_conv6_wt, npy_2_conv6_b;
        npy_2_conv6_q.open(Q_table_datapath+Q_table_name[9]); npy_2_conv6_q.init_tensor2(Q_table_2_conv6, DataType::S16);
        npy_2_conv6_wt.open(Q_table_datapath+Q_table_name[9]); npy_2_conv6_wt.init_tensor2(WT_buffer_2_conv6, DataType::U16);
        npy_2_conv6_b.open(bias_datapath+bias_name[9]); npy_2_conv6_b.init_tensor2(bias_2_conv6, DataType::S8);
        WeightsInfo weights_2_conv6(false, 1, 1, 32, false);
        const TensorShape out_2_conv6_shape(size4, size4, weights_2_conv6.num_kernels());
        out_2_conv6.allocator()->init(TensorInfo(out_2_conv6_shape, 1, DataType::S8));
        out_2_act6.allocator()->init(TensorInfo(out_2_conv6_shape, 1, DataType::S8));
        TensorShape out_2_cat_shape(size4, size4, weights_2_conv0.num_kernels()+weights_2_conv2.num_kernels()+weights_2_conv5.num_kernels()+weights_2_conv6.num_kernels());
        out_2_cat.allocator()->init(TensorInfo(out_2_cat_shape, 1, DataType::F32));
/**********************************************b3*******************/
        NPYLoader npy_3_conv0_q, npy_3_conv0_wt, npy_3_conv0_b;
        npy_3_conv0_q.open(Q_table_datapath+Q_table_name[10]); npy_3_conv0_q.init_tensor2(Q_table_3_conv0, DataType::S16);
        npy_3_conv0_wt.open(Q_table_datapath+Q_table_name[10]); npy_3_conv0_wt.init_tensor2(WT_buffer_3_conv0, DataType::U16);
        npy_3_conv0_b.open(bias_datapath+bias_name[10]); npy_3_conv0_b.init_tensor2(bias_3_conv0,  DataType::S8);
        WeightsInfo weights_3_conv0(false, 1, 1, 64, false);
        const TensorShape out_3_conv0_shape(size4, size4, weights_3_conv0.num_kernels());
        out_3_conv0.allocator()->init(TensorInfo(out_3_conv0_shape, 1, DataType::S8));
        out_3_act0.allocator()->init(TensorInfo(out_3_conv0_shape, 1, DataType::S8));
                NPYLoader npy_3_conv1_q, npy_3_conv1_wt, npy_3_conv1_b;
        npy_3_conv1_q.open(Q_table_datapath+Q_table_name[11]); npy_3_conv1_q.init_tensor2(Q_table_3_conv1, DataType::S16);
        npy_3_conv1_wt.open(Q_table_datapath+Q_table_name[11]); npy_3_conv1_wt.init_tensor2(WT_buffer_3_conv1, DataType::U16);
        npy_3_conv1_b.open(bias_datapath+bias_name[11]); npy_3_conv1_b.init_tensor2(bias_3_conv1, DataType::S8);
        WeightsInfo weights_3_conv1(false, 1, 1, 64, false);
        const TensorShape out_3_conv1_shape(size4, size4, weights_3_conv1.num_kernels());
        out_3_conv1.allocator()->init(TensorInfo(out_3_conv1_shape, 1, DataType::S8));
        out_3_act1.allocator()->init(TensorInfo(out_3_conv1_shape, 1, DataType::S8));
                NPYLoader npy_3_conv2_q, npy_3_conv2_wt, npy_3_conv2_b;
        npy_3_conv2_q.open(Q_table_datapath+Q_table_name[12]); npy_3_conv2_q.init_tensor2(Q_table_3_conv2, DataType::S16);
        npy_3_conv2_wt.open(Q_table_datapath+Q_table_name[12]); npy_3_conv2_wt.init_tensor2(WT_buffer_3_conv2, DataType::U16);
        npy_3_conv2_b.open(bias_datapath+bias_name[12]); npy_3_conv2_b.init_tensor2(bias_3_conv2, DataType::S8);
        WeightsInfo weights_3_conv2(false, 3, 3, 96, false);
        const TensorShape out_3_conv2_shape(size4, size4, weights_3_conv2.num_kernels());
        out_3_conv2.allocator()->init(TensorInfo(out_3_conv2_shape, 1, DataType::S8));
        out_3_act2.allocator()->init(TensorInfo(out_3_conv2_shape, 1, DataType::S8));
                NPYLoader npy_3_conv3_q, npy_3_conv3_wt, npy_3_conv3_b;
        npy_3_conv3_q.open(Q_table_datapath+Q_table_name[13]); npy_3_conv3_q.init_tensor2(Q_table_3_conv3, DataType::S16);
        npy_3_conv3_wt.open(Q_table_datapath+Q_table_name[13]); npy_3_conv3_wt.init_tensor2(WT_buffer_3_conv3, DataType::U16);
        npy_3_conv3_b.open(bias_datapath+bias_name[13]); npy_3_conv3_b.init_tensor2(bias_3_conv3, DataType::S8);
        WeightsInfo weights_3_conv3(false, 1, 1, 64, false);
        const TensorShape out_3_conv3_shape(size4, size4, weights_3_conv3.num_kernels());
        out_3_conv3.allocator()->init(TensorInfo(out_3_conv3_shape, 1, DataType::S8));
        out_3_act3.allocator()->init(TensorInfo(out_3_conv3_shape, 1, DataType::S8));
                NPYLoader npy_3_conv4_q, npy_3_conv4_wt, npy_3_conv4_b;
        npy_3_conv4_q.open(Q_table_datapath+Q_table_name[14]); npy_3_conv4_q.init_tensor2(Q_table_3_conv4, DataType::S16);
        npy_3_conv4_wt.open(Q_table_datapath+Q_table_name[14]); npy_3_conv4_wt.init_tensor2(WT_buffer_3_conv4, DataType::U16);
        npy_3_conv4_b.open(bias_datapath+bias_name[14]); npy_3_conv4_b.init_tensor2(bias_3_conv4, DataType::S8);
        WeightsInfo weights_3_conv4(false, 3, 3, 96, false);
        const TensorShape out_3_conv4_shape(size4, size4, weights_3_conv4.num_kernels());
        out_3_conv4.allocator()->init(TensorInfo(out_3_conv4_shape, 1, DataType::S8));
        out_3_act4.allocator()->init(TensorInfo(out_3_conv4_shape, 1, DataType::S8));
                NPYLoader npy_3_conv5_q, npy_3_conv5_wt, npy_3_conv5_b;
        npy_3_conv5_q.open(Q_table_datapath+Q_table_name[15]); npy_3_conv5_q.init_tensor2(Q_table_3_conv5, DataType::S16);
        npy_3_conv5_wt.open(Q_table_datapath+Q_table_name[15]); npy_3_conv5_wt.init_tensor2(WT_buffer_3_conv5, DataType::U16);
        npy_3_conv5_b.open(bias_datapath+bias_name[15]); npy_3_conv5_b.init_tensor2(bias_3_conv5, DataType::S8);
        WeightsInfo weights_3_conv5(false, 3, 3, 96, false);
        const TensorShape out_3_conv5_shape(size4, size4, weights_3_conv5.num_kernels());
        out_3_conv5.allocator()->init(TensorInfo(out_3_conv5_shape, 1, DataType::S8));
        out_3_act5.allocator()->init(TensorInfo(out_3_conv5_shape, 1, DataType::S8));
                NPYLoader npy_3_conv6_q, npy_3_conv6_wt, npy_3_conv6_b;
        npy_3_conv6_q.open(Q_table_datapath+Q_table_name[16]); npy_3_conv6_q.init_tensor2(Q_table_3_conv6, DataType::S16);
        npy_3_conv6_wt.open(Q_table_datapath+Q_table_name[16]); npy_3_conv6_wt.init_tensor2(WT_buffer_3_conv6, DataType::U16);
        npy_3_conv6_b.open(bias_datapath+bias_name[16]); npy_3_conv6_b.init_tensor2(bias_3_conv6, DataType::S8);
        WeightsInfo weights_3_conv6(false, 1, 1, 64, false);
        const TensorShape out_3_conv6_shape(size4, size4, weights_3_conv6.num_kernels());
        out_3_conv6.allocator()->init(TensorInfo(out_3_conv6_shape, 1, DataType::S8));
        out_3_act6.allocator()->init(TensorInfo(out_3_conv6_shape, 1, DataType::S8));
        TensorShape out_3_cat_shape(size4, size4, weights_3_conv0.num_kernels()+weights_3_conv2.num_kernels()+weights_3_conv5.num_kernels()+weights_3_conv6.num_kernels());
        out_3_cat.allocator()->init(TensorInfo(out_3_cat_shape, 1, DataType::F32));
/**********************************************c3*******************/
        NPYLoader npy_4_conv0_q, npy_4_conv0_wt, npy_4_conv0_b;
        npy_4_conv0_q.open(Q_table_datapath+Q_table_name[17]); npy_4_conv0_q.init_tensor2(Q_table_4_conv0, DataType::S16);
        npy_4_conv0_wt.open(Q_table_datapath+Q_table_name[17]); npy_4_conv0_wt.init_tensor2(WT_buffer_4_conv0, DataType::U16);
        npy_4_conv0_b.open(bias_datapath+bias_name[17]); npy_4_conv0_b.init_tensor2(bias_4_conv0, DataType::S8);
        WeightsInfo weights_4_conv0(false, 1, 1, 160, false);
        const TensorShape out_4_conv0_shape(size4, size4, weights_4_conv0.num_kernels());
        out_4_conv0.allocator()->init(TensorInfo(out_4_conv0_shape, 1, DataType::S8));
        out_4_act0.allocator()->init(TensorInfo(out_4_conv0_shape, 1, DataType::S8));
                NPYLoader npy_4_conv1_q, npy_4_conv1_wt, npy_4_conv1_b;
        npy_4_conv1_q.open(Q_table_datapath+Q_table_name[18]); npy_4_conv1_q.init_tensor2(Q_table_4_conv1, DataType::S16);
        npy_4_conv1_wt.open(Q_table_datapath+Q_table_name[18]); npy_4_conv1_wt.init_tensor2(WT_buffer_4_conv1, DataType::U16);
        npy_4_conv1_b.open(bias_datapath+bias_name[18]); npy_4_conv1_b.init_tensor2(bias_4_conv1, DataType::S8);
        WeightsInfo weights_4_conv1(false, 1, 1, 128, false);
        const TensorShape out_4_conv1_shape(size4, size4, weights_4_conv1.num_kernels());
        out_4_conv1.allocator()->init(TensorInfo(out_4_conv1_shape, 1, DataType::S8));
        out_4_act1.allocator()->init(TensorInfo(out_4_conv1_shape, 1, DataType::S8));
                NPYLoader npy_4_conv2_q, npy_4_conv2_wt, npy_4_conv2_b;
        npy_4_conv2_q.open(Q_table_datapath+Q_table_name[19]); npy_4_conv2_q.init_tensor2(Q_table_4_conv2, DataType::S16);
        npy_4_conv2_wt.open(Q_table_datapath+Q_table_name[19]); npy_4_conv2_wt.init_tensor2(WT_buffer_4_conv2, DataType::U16);
        npy_4_conv2_b.open(bias_datapath+bias_name[19]); npy_4_conv2_b.init_tensor2(bias_4_conv2, DataType::S8);
        WeightsInfo weights_4_conv2(false, 3, 3, 160, false);
        const TensorShape out_4_conv2_shape(size4, size4, weights_4_conv2.num_kernels());
        out_4_conv2.allocator()->init(TensorInfo(out_4_conv2_shape, 1, DataType::S8));
        out_4_act2.allocator()->init(TensorInfo(out_4_conv2_shape, 1, DataType::S8));
                NPYLoader npy_4_conv3_q, npy_4_conv3_wt, npy_4_conv3_b;
        npy_4_conv3_q.open(Q_table_datapath+Q_table_name[20]); npy_4_conv3_q.init_tensor2(Q_table_4_conv3, DataType::S16);
        npy_4_conv3_wt.open(Q_table_datapath+Q_table_name[20]); npy_4_conv3_wt.init_tensor2(WT_buffer_4_conv3, DataType::U16);
        npy_4_conv3_b.open(bias_datapath+bias_name[20]); npy_4_conv3_b.init_tensor2(bias_4_conv3, DataType::S8);
        WeightsInfo weights_4_conv3(false, 1, 1, 64, false);
        const TensorShape out_4_conv3_shape(size4, size4, weights_4_conv3.num_kernels());
        out_4_conv3.allocator()->init(TensorInfo(out_4_conv3_shape, 1, DataType::S8));
        out_4_act3.allocator()->init(TensorInfo(out_4_conv3_shape, 1, DataType::S8));
                NPYLoader npy_4_conv4_q, npy_4_conv4_wt, npy_4_conv4_b;
        npy_4_conv4_q.open(Q_table_datapath+Q_table_name[21]); npy_4_conv4_q.init_tensor2(Q_table_4_conv4, DataType::S16);
        npy_4_conv4_wt.open(Q_table_datapath+Q_table_name[21]); npy_4_conv4_wt.init_tensor2(WT_buffer_4_conv4, DataType::U16);
        npy_4_conv4_b.open(bias_datapath+bias_name[21]); npy_4_conv4_b.init_tensor2(bias_4_conv4, DataType::S8);
        WeightsInfo weights_4_conv4(false, 3, 3, 96, false);
        const TensorShape out_4_conv4_shape(size4, size4, weights_4_conv4.num_kernels());
        out_4_conv4.allocator()->init(TensorInfo(out_4_conv4_shape, 1, DataType::S8));
        out_4_act4.allocator()->init(TensorInfo(out_4_conv4_shape, 1, DataType::S8));
                NPYLoader npy_4_conv5_q, npy_4_conv5_wt, npy_4_conv5_b;
        npy_4_conv5_q.open(Q_table_datapath+Q_table_name[22]); npy_4_conv5_q.init_tensor2(Q_table_4_conv5, DataType::S16);
        npy_4_conv5_wt.open(Q_table_datapath+Q_table_name[22]); npy_4_conv5_wt.init_tensor2(WT_buffer_4_conv5, DataType::U16);
        npy_4_conv5_b.open(bias_datapath+bias_name[22]); npy_4_conv5_b.init_tensor2(bias_4_conv5, DataType::S8);
        WeightsInfo weights_4_conv5(false, 3, 3, 96, false);
        const TensorShape out_4_conv5_shape(size4, size4, weights_4_conv5.num_kernels());
        out_4_conv5.allocator()->init(TensorInfo(out_4_conv5_shape, 1, DataType::S8));
        out_4_act5.allocator()->init(TensorInfo(out_4_conv5_shape, 1, DataType::S8));
                NPYLoader npy_4_conv6_q, npy_4_conv6_wt, npy_4_conv6_b;
        npy_4_conv6_q.open(Q_table_datapath+Q_table_name[23]); npy_4_conv6_q.init_tensor2(Q_table_4_conv6, DataType::S16);
        npy_4_conv6_wt.open(Q_table_datapath+Q_table_name[23]); npy_4_conv6_wt.init_tensor2(WT_buffer_4_conv6, DataType::U16);
        npy_4_conv6_b.open(bias_datapath+bias_name[23]); npy_4_conv6_b.init_tensor2(bias_4_conv6, DataType::S8);
        WeightsInfo weights_4_conv6(false, 1, 1, 160, false);
        const TensorShape out_4_conv6_shape(size4, size4, weights_4_conv6.num_kernels());
        out_4_conv6.allocator()->init(TensorInfo(out_4_conv6_shape, 1, DataType::S8));
        out_4_act6.allocator()->init(TensorInfo(out_4_conv6_shape, 1, DataType::S8));
        TensorShape out_4_cat_shape(size4, size4, weights_4_conv0.num_kernels()+weights_4_conv2.num_kernels()+weights_4_conv5.num_kernels()+weights_4_conv6.num_kernels());
        out_4_cat.allocator()->init(TensorInfo(out_4_cat_shape, 1, DataType::F32));
/**********************************************maxpool***********************/
        TensorShape out_5_pool0_shape=out_4_cat_shape;
        out_5_pool0_shape.set(0, out_5_pool0_shape.x()/2);
        out_5_pool0_shape.set(1, out_5_pool0_shape.y()/2);
        out_5_pool0.allocator()->init(TensorInfo(out_5_pool0_shape, 1, DataType::F32));
/**************************************a4***************************************/
        NPYLoader npy_6_conv0_q, npy_6_conv0_wt, npy_6_conv0_b;
        npy_6_conv0_q.open(Q_table_datapath+Q_table_name[24]); npy_6_conv0_q.init_tensor2(Q_table_6_conv0, DataType::S16);
        npy_6_conv0_wt.open(Q_table_datapath+Q_table_name[24]); npy_6_conv0_wt.init_tensor2(WT_buffer_6_conv0, DataType::U16);
        npy_6_conv0_b.open(bias_datapath+bias_name[24]); npy_6_conv0_b.init_tensor2(bias_6_conv0, DataType::S8);
        WeightsInfo weights_6_conv0(false, 1, 1, 224, false);
        const TensorShape out_6_conv0_shape(size5, size5, weights_6_conv0.num_kernels());
        out_6_conv0.allocator()->init(TensorInfo(out_6_conv0_shape, 1, DataType::S8));
        out_6_act0.allocator()->init(TensorInfo(out_6_conv0_shape, 1, DataType::S8));
                NPYLoader npy_6_conv1_q, npy_6_conv1_wt, npy_6_conv1_b;
        npy_6_conv1_q.open(Q_table_datapath+Q_table_name[25]); npy_6_conv1_q.init_tensor2(Q_table_6_conv1, DataType::S16);
        npy_6_conv1_wt.open(Q_table_datapath+Q_table_name[25]); npy_6_conv1_wt.init_tensor2(WT_buffer_6_conv1, DataType::U16);
        npy_6_conv1_b.open(bias_datapath+bias_name[25]); npy_6_conv1_b.init_tensor2(bias_6_conv1, DataType::S8);
        WeightsInfo weights_6_conv1(false, 1, 1, 64, false);
        const TensorShape out_6_conv1_shape(size5, size5, weights_6_conv1.num_kernels());
        out_6_conv1.allocator()->init(TensorInfo(out_6_conv1_shape, 1, DataType::S8));
        out_6_act1.allocator()->init(TensorInfo(out_6_conv1_shape, 1, DataType::S8));
                NPYLoader npy_6_conv2_q, npy_6_conv2_wt, npy_6_conv2_b;
        npy_6_conv2_q.open(Q_table_datapath+Q_table_name[26]); npy_6_conv2_q.init_tensor2(Q_table_6_conv2, DataType::S16);
        npy_6_conv2_wt.open(Q_table_datapath+Q_table_name[26]); npy_6_conv2_wt.init_tensor2(WT_buffer_6_conv2, DataType::U16);
        npy_6_conv2_b.open(bias_datapath+bias_name[26]); npy_6_conv2_b.init_tensor2(bias_6_conv2, DataType::S8);
        WeightsInfo weights_6_conv2(false, 3, 3, 96, false);
        const TensorShape out_6_conv2_shape(size5, size5, weights_6_conv2.num_kernels());
        out_6_conv2.allocator()->init(TensorInfo(out_6_conv2_shape, 1, DataType::S8));
        out_6_act2.allocator()->init(TensorInfo(out_6_conv2_shape, 1, DataType::S8));
                NPYLoader npy_6_conv3_q, npy_6_conv3_wt, npy_6_conv3_b;
        npy_6_conv3_q.open(Q_table_datapath+Q_table_name[27]); npy_6_conv3_q.init_tensor2(Q_table_6_conv3, DataType::S16);
        npy_6_conv3_wt.open(Q_table_datapath+Q_table_name[27]); npy_6_conv3_wt.init_tensor2(WT_buffer_6_conv3, DataType::U16);
        npy_6_conv3_b.open(bias_datapath+bias_name[27]); npy_6_conv3_b.init_tensor2(bias_6_conv3, DataType::S8);
        WeightsInfo weights_6_conv3(false, 1, 1, 96, false);
        const TensorShape out_6_conv3_shape(size5, size5, weights_6_conv3.num_kernels());
        out_6_conv3.allocator()->init(TensorInfo(out_6_conv3_shape, 1, DataType::S8));
        out_6_act3.allocator()->init(TensorInfo(out_6_conv3_shape, 1, DataType::S8));
                NPYLoader npy_6_conv4_q, npy_6_conv4_wt, npy_6_conv4_b;
        npy_6_conv4_q.open(Q_table_datapath+Q_table_name[28]); npy_6_conv4_q.init_tensor2(Q_table_6_conv4, DataType::S16);
        npy_6_conv4_wt.open(Q_table_datapath+Q_table_name[28]); npy_6_conv4_wt.init_tensor2(WT_buffer_6_conv4, DataType::U16);
        npy_6_conv4_b.open(bias_datapath+bias_name[28]); npy_6_conv4_b.init_tensor2(bias_6_conv4, DataType::S8);
        WeightsInfo weights_6_conv4(false, 3, 3, 128, false);
        const TensorShape out_6_conv4_shape(size5, size5, weights_6_conv4.num_kernels());
        out_6_conv4.allocator()->init(TensorInfo(out_6_conv4_shape, 1, DataType::S8));
        out_6_act4.allocator()->init(TensorInfo(out_6_conv4_shape, 1, DataType::S8));
                NPYLoader npy_6_conv5_q, npy_6_conv5_wt, npy_6_conv5_b;
        npy_6_conv5_q.open(Q_table_datapath+Q_table_name[29]); npy_6_conv5_q.init_tensor2(Q_table_6_conv5, DataType::S16);
        npy_6_conv5_wt.open(Q_table_datapath+Q_table_name[29]); npy_6_conv5_wt.init_tensor2(WT_buffer_6_conv5, DataType::U16);
        npy_6_conv5_b.open(bias_datapath+bias_name[29]); npy_6_conv5_b.init_tensor2(bias_6_conv5, DataType::S8);
        WeightsInfo weights_6_conv5(false, 3, 3, 128, false);
        const TensorShape out_6_conv5_shape(size5, size5, weights_6_conv5.num_kernels());
        out_6_conv5.allocator()->init(TensorInfo(out_6_conv5_shape, 1, DataType::S8));
        out_6_act5.allocator()->init(TensorInfo(out_6_conv5_shape, 1, DataType::S8));
                NPYLoader npy_6_conv6_q, npy_6_conv6_wt, npy_6_conv6_b;
        npy_6_conv6_q.open(Q_table_datapath+Q_table_name[30]); npy_6_conv6_q.init_tensor2(Q_table_6_conv6, DataType::S16);
        npy_6_conv6_wt.open(Q_table_datapath+Q_table_name[30]); npy_6_conv6_wt.init_tensor2(WT_buffer_6_conv6, DataType::U16);
        npy_6_conv6_b.open(bias_datapath+bias_name[30]); npy_6_conv6_b.init_tensor2(bias_6_conv6, DataType::S8);
        WeightsInfo weights_6_conv6(false, 1, 1, 128, false);
        const TensorShape out_6_conv6_shape(size5, size5, weights_6_conv6.num_kernels());
        out_6_conv6.allocator()->init(TensorInfo(out_6_conv6_shape, 1, DataType::S8));
        out_6_act6.allocator()->init(TensorInfo(out_6_conv6_shape, 1, DataType::S8));
        TensorShape out_6_cat_shape(size5, size5, weights_6_conv0.num_kernels()+weights_6_conv2.num_kernels()+weights_6_conv5.num_kernels()+weights_6_conv6.num_kernels());
        out_6_cat.allocator()->init(TensorInfo(out_6_cat_shape, 1, DataType::F32));
/**************************************b4***************************************/
        NPYLoader npy_7_conv0_q, npy_7_conv0_wt, npy_7_conv0_b;
        npy_7_conv0_q.open(Q_table_datapath+Q_table_name[31]); npy_7_conv0_q.init_tensor2(Q_table_7_conv0, DataType::S16);
        npy_7_conv0_wt.open(Q_table_datapath+Q_table_name[31]); npy_7_conv0_wt.init_tensor2(WT_buffer_7_conv0, DataType::U16);
        npy_7_conv0_b.open(bias_datapath+bias_name[31]); npy_7_conv0_b.init_tensor2(bias_7_conv0, DataType::S8);
        WeightsInfo weights_7_conv0(false, 1, 1, 192, false);
        const TensorShape out_7_conv0_shape(size5, size5, weights_7_conv0.num_kernels());
        out_7_conv0.allocator()->init(TensorInfo(out_7_conv0_shape, 1, DataType::S8));
        out_7_act0.allocator()->init(TensorInfo(out_7_conv0_shape, 1, DataType::S8));
                NPYLoader npy_7_conv1_q, npy_7_conv1_wt, npy_7_conv1_b;
        npy_7_conv1_q.open(Q_table_datapath+Q_table_name[32]); npy_7_conv1_q.init_tensor2(Q_table_7_conv1, DataType::S16);
        npy_7_conv1_wt.open(Q_table_datapath+Q_table_name[32]); npy_7_conv1_wt.init_tensor2(WT_buffer_7_conv1, DataType::U16);
        npy_7_conv1_b.open(bias_datapath+bias_name[32]); npy_7_conv1_b.init_tensor2(bias_7_conv1, DataType::S8);
        WeightsInfo weights_7_conv1(false, 1, 1, 96, false);
        const TensorShape out_7_conv1_shape(size5, size5, weights_7_conv1.num_kernels());
        out_7_conv1.allocator()->init(TensorInfo(out_7_conv1_shape, 1, DataType::S8));
        out_7_act1.allocator()->init(TensorInfo(out_7_conv1_shape, 1, DataType::S8));
                NPYLoader npy_7_conv2_q, npy_7_conv2_wt, npy_7_conv2_b;
        npy_7_conv2_q.open(Q_table_datapath+Q_table_name[33]); npy_7_conv2_q.init_tensor2(Q_table_7_conv2, DataType::S16);
        npy_7_conv2_wt.open(Q_table_datapath+Q_table_name[33]); npy_7_conv2_wt.init_tensor2(WT_buffer_7_conv2, DataType::U16);
        npy_7_conv2_b.open(bias_datapath+bias_name[33]); npy_7_conv2_b.init_tensor2(bias_7_conv2, DataType::S8);
        WeightsInfo weights_7_conv2(false, 3, 3, 128, false);
        const TensorShape out_7_conv2_shape(size5, size5, weights_7_conv2.num_kernels());
        out_7_conv2.allocator()->init(TensorInfo(out_7_conv2_shape, 1, DataType::S8));
        out_7_act2.allocator()->init(TensorInfo(out_7_conv2_shape, 1, DataType::S8));
                NPYLoader npy_7_conv3_q, npy_7_conv3_wt, npy_7_conv3_b;
        npy_7_conv3_q.open(Q_table_datapath+Q_table_name[34]); npy_7_conv3_q.init_tensor2(Q_table_7_conv3, DataType::S16);
        npy_7_conv3_wt.open(Q_table_datapath+Q_table_name[34]); npy_7_conv3_wt.init_tensor2(WT_buffer_7_conv3, DataType::U16);
        npy_7_conv3_b.open(bias_datapath+bias_name[34]); npy_7_conv3_b.init_tensor2(bias_7_conv3, DataType::S8);
        WeightsInfo weights_7_conv3(false, 1, 1, 96, false);
        const TensorShape out_7_conv3_shape(size5, size5, weights_7_conv3.num_kernels());
        out_7_conv3.allocator()->init(TensorInfo(out_7_conv3_shape, 1, DataType::S8));
        out_7_act3.allocator()->init(TensorInfo(out_7_conv3_shape, 1, DataType::S8));
                NPYLoader npy_7_conv4_q, npy_7_conv4_wt, npy_7_conv4_b;
        npy_7_conv4_q.open(Q_table_datapath+Q_table_name[35]); npy_7_conv4_q.init_tensor2(Q_table_7_conv4, DataType::S16);
        npy_7_conv4_wt.open(Q_table_datapath+Q_table_name[35]); npy_7_conv4_wt.init_tensor2(WT_buffer_7_conv4, DataType::U16);
        npy_7_conv4_b.open(bias_datapath+bias_name[35]); npy_7_conv4_b.init_tensor2(bias_7_conv4, DataType::S8);
        WeightsInfo weights_7_conv4(false, 3, 3, 128, false);
        const TensorShape out_7_conv4_shape(size5, size5, weights_7_conv4.num_kernels());
        out_7_conv4.allocator()->init(TensorInfo(out_7_conv4_shape, 1, DataType::S8));
        out_7_act4.allocator()->init(TensorInfo(out_7_conv4_shape, 1, DataType::S8));
                NPYLoader npy_7_conv5_q, npy_7_conv5_wt, npy_7_conv5_b;
        npy_7_conv5_q.open(Q_table_datapath+Q_table_name[36]); npy_7_conv5_q.init_tensor2(Q_table_7_conv5, DataType::S16);
        npy_7_conv5_wt.open(Q_table_datapath+Q_table_name[36]); npy_7_conv5_wt.init_tensor2(WT_buffer_7_conv5, DataType::U16);
        npy_7_conv5_b.open(bias_datapath+bias_name[36]); npy_7_conv5_b.init_tensor2(bias_7_conv5, DataType::S8);
        WeightsInfo weights_7_conv5(false, 3, 3, 128, false);
        const TensorShape out_7_conv5_shape(size5, size5, weights_7_conv5.num_kernels());
        out_7_conv5.allocator()->init(TensorInfo(out_7_conv5_shape, 1, DataType::S8));
        out_7_act5.allocator()->init(TensorInfo(out_7_conv5_shape, 1, DataType::S8));
                NPYLoader npy_7_conv6_q, npy_7_conv6_wt, npy_7_conv6_b;
        npy_7_conv6_q.open(Q_table_datapath+Q_table_name[37]); npy_7_conv6_q.init_tensor2(Q_table_7_conv6, DataType::S16);
        npy_7_conv6_wt.open(Q_table_datapath+Q_table_name[37]); npy_7_conv6_wt.init_tensor2(WT_buffer_7_conv6, DataType::U16);
        npy_7_conv6_b.open(bias_datapath+bias_name[37]); npy_7_conv6_b.init_tensor2(bias_7_conv6, DataType::S8);
        WeightsInfo weights_7_conv6(false, 1, 1, 128, false);
        const TensorShape out_7_conv6_shape(size5, size5, weights_7_conv6.num_kernels());
        out_7_conv6.allocator()->init(TensorInfo(out_7_conv6_shape, 1, DataType::S8));
        out_7_act6.allocator()->init(TensorInfo(out_7_conv6_shape, 1, DataType::S8));
        TensorShape out_7_cat_shape(size5, size5, weights_7_conv0.num_kernels()+weights_7_conv2.num_kernels()+weights_7_conv5.num_kernels()+weights_7_conv6.num_kernels());
        out_7_cat.allocator()->init(TensorInfo(out_7_cat_shape, 1, DataType::F32));
/**************************************c4***************************************/
        NPYLoader npy_8_conv0_q, npy_8_conv0_wt, npy_8_conv0_b;
        npy_8_conv0_q.open(Q_table_datapath+Q_table_name[38]); npy_8_conv0_q.init_tensor2(Q_table_8_conv0, DataType::S16);
        npy_8_conv0_wt.open(Q_table_datapath+Q_table_name[38]); npy_8_conv0_wt.init_tensor2(WT_buffer_8_conv0, DataType::U16);
        npy_8_conv0_b.open(bias_datapath+bias_name[38]); npy_8_conv0_b.init_tensor2(bias_8_conv0, DataType::S8);
        WeightsInfo weights_8_conv0(false, 1, 1, 160, false);
        const TensorShape out_8_conv0_shape(size5, size5, weights_8_conv0.num_kernels());
        out_8_conv0.allocator()->init(TensorInfo(out_8_conv0_shape, 1, DataType::S8));
        out_8_act0.allocator()->init(TensorInfo(out_8_conv0_shape, 1, DataType::S8));
                NPYLoader npy_8_conv1_q, npy_8_conv1_wt, npy_8_conv1_b;
        npy_8_conv1_q.open(Q_table_datapath+Q_table_name[39]); npy_8_conv1_q.init_tensor2(Q_table_8_conv1, DataType::S16);
        npy_8_conv1_wt.open(Q_table_datapath+Q_table_name[39]); npy_8_conv1_wt.init_tensor2(WT_buffer_8_conv1, DataType::U16);
        npy_8_conv1_b.open(bias_datapath+bias_name[39]); npy_8_conv1_b.init_tensor2(bias_8_conv1, DataType::S8);
        WeightsInfo weights_8_conv1(false, 1, 1, 128, false);
        const TensorShape out_8_conv1_shape(size5, size5, weights_8_conv1.num_kernels());
        out_8_conv1.allocator()->init(TensorInfo(out_8_conv1_shape, 1, DataType::S8));
        out_8_act1.allocator()->init(TensorInfo(out_8_conv1_shape, 1, DataType::S8));
                NPYLoader npy_8_conv2_q, npy_8_conv2_wt, npy_8_conv2_b;
        npy_8_conv2_q.open(Q_table_datapath+Q_table_name[40]); npy_8_conv2_q.init_tensor2(Q_table_8_conv2, DataType::S16);
        npy_8_conv2_wt.open(Q_table_datapath+Q_table_name[40]); npy_8_conv2_wt.init_tensor2(WT_buffer_8_conv2, DataType::U16);
        npy_8_conv2_b.open(bias_datapath+bias_name[40]); npy_8_conv2_b.init_tensor2(bias_8_conv2, DataType::S8);
        WeightsInfo weights_8_conv2(false, 3, 3, 160, false);
        const TensorShape out_8_conv2_shape(size5, size5, weights_8_conv2.num_kernels());
        out_8_conv2.allocator()->init(TensorInfo(out_8_conv2_shape, 1, DataType::S8));
        out_8_act2.allocator()->init(TensorInfo(out_8_conv2_shape, 1, DataType::S8));
                NPYLoader npy_8_conv3_q, npy_8_conv3_wt, npy_8_conv3_b;
        npy_8_conv3_q.open(Q_table_datapath+Q_table_name[41]); npy_8_conv3_q.init_tensor2(Q_table_8_conv3, DataType::S16);
        npy_8_conv3_wt.open(Q_table_datapath+Q_table_name[41]); npy_8_conv3_wt.init_tensor2(WT_buffer_8_conv3, DataType::U16);
        npy_8_conv3_b.open(bias_datapath+bias_name[41]); npy_8_conv3_b.init_tensor2(bias_8_conv3, DataType::S8);
        WeightsInfo weights_8_conv3(false, 1, 1, 128, false);
        const TensorShape out_8_conv3_shape(size5, size5, weights_8_conv3.num_kernels());
        out_8_conv3.allocator()->init(TensorInfo(out_8_conv3_shape, 1, DataType::S8));
        out_8_act3.allocator()->init(TensorInfo(out_8_conv3_shape, 1, DataType::S8));
                NPYLoader npy_8_conv4_q, npy_8_conv4_wt, npy_8_conv4_b;
        npy_8_conv4_q.open(Q_table_datapath+Q_table_name[42]); npy_8_conv4_q.init_tensor2(Q_table_8_conv4, DataType::S16);
        npy_8_conv4_wt.open(Q_table_datapath+Q_table_name[42]); npy_8_conv4_wt.init_tensor2(WT_buffer_8_conv4, DataType::U16);
        npy_8_conv4_b.open(bias_datapath+bias_name[42]); npy_8_conv4_b.init_tensor2(bias_8_conv4, DataType::S8);
        WeightsInfo weights_8_conv4(false, 3, 3, 160, false);
        const TensorShape out_8_conv4_shape(size5, size5, weights_8_conv4.num_kernels());
        out_8_conv4.allocator()->init(TensorInfo(out_8_conv4_shape, 1, DataType::S8));
        out_8_act4.allocator()->init(TensorInfo(out_8_conv4_shape, 1, DataType::S8));
                NPYLoader npy_8_conv5_q, npy_8_conv5_wt, npy_8_conv5_b;
        npy_8_conv5_q.open(Q_table_datapath+Q_table_name[43]); npy_8_conv5_q.init_tensor2(Q_table_8_conv5, DataType::S16);
        npy_8_conv5_wt.open(Q_table_datapath+Q_table_name[43]); npy_8_conv5_wt.init_tensor2(WT_buffer_8_conv5, DataType::U16);
        npy_8_conv5_b.open(bias_datapath+bias_name[43]); npy_8_conv5_b.init_tensor2(bias_8_conv5, DataType::S8);
        WeightsInfo weights_8_conv5(false, 3, 3, 160, false);
        const TensorShape out_8_conv5_shape(size5, size5, weights_8_conv5.num_kernels());
        out_8_conv5.allocator()->init(TensorInfo(out_8_conv5_shape, 1, DataType::S8));
        out_8_act5.allocator()->init(TensorInfo(out_8_conv5_shape, 1, DataType::S8));
                NPYLoader npy_8_conv6_q, npy_8_conv6_wt, npy_8_conv6_b;
        npy_8_conv6_q.open(Q_table_datapath+Q_table_name[44]); npy_8_conv6_q.init_tensor2(Q_table_8_conv6, DataType::S16);
        npy_8_conv6_wt.open(Q_table_datapath+Q_table_name[44]); npy_8_conv6_wt.init_tensor2(WT_buffer_8_conv6, DataType::U16);
        npy_8_conv6_b.open(bias_datapath+bias_name[44]); npy_8_conv6_b.init_tensor2(bias_8_conv6, DataType::S8);
        WeightsInfo weights_8_conv6(false, 1, 1, 128, false);
        const TensorShape out_8_conv6_shape(size5, size5, weights_8_conv6.num_kernels());
        out_8_conv6.allocator()->init(TensorInfo(out_8_conv6_shape, 1, DataType::S8));
        out_8_act6.allocator()->init(TensorInfo(out_8_conv6_shape, 1, DataType::S8));
        TensorShape out_8_cat_shape(size5, size5, weights_8_conv0.num_kernels()+weights_8_conv2.num_kernels()+weights_8_conv5.num_kernels()+weights_8_conv6.num_kernels());
        out_8_cat.allocator()->init(TensorInfo(out_8_cat_shape, 1, DataType::F32));
/**************************************d4***************************************/
        NPYLoader npy_9_conv0_q, npy_9_conv0_wt, npy_9_conv0_b;
        npy_9_conv0_q.open(Q_table_datapath+Q_table_name[45]); npy_9_conv0_q.init_tensor2(Q_table_9_conv0, DataType::S16);
        npy_9_conv0_wt.open(Q_table_datapath+Q_table_name[45]); npy_9_conv0_wt.init_tensor2(WT_buffer_9_conv0, DataType::U16);
        npy_9_conv0_b.open(bias_datapath+bias_name[45]); npy_9_conv0_b.init_tensor2(bias_9_conv0, DataType::S8);
        WeightsInfo weights_9_conv0(false, 1, 1, 96, false);
        const TensorShape out_9_conv0_shape(size5, size5, weights_9_conv0.num_kernels());
        out_9_conv0.allocator()->init(TensorInfo(out_9_conv0_shape, 1, DataType::S8));
        out_9_act0.allocator()->init(TensorInfo(out_9_conv0_shape, 1, DataType::S8));
                NPYLoader npy_9_conv1_q, npy_9_conv1_wt, npy_9_conv1_b;
        npy_9_conv1_q.open(Q_table_datapath+Q_table_name[46]); npy_9_conv1_q.init_tensor2(Q_table_9_conv1, DataType::S16);
        npy_9_conv1_wt.open(Q_table_datapath+Q_table_name[46]); npy_9_conv1_wt.init_tensor2(WT_buffer_9_conv1, DataType::U16);
        npy_9_conv1_b.open(bias_datapath+bias_name[46]); npy_9_conv1_b.init_tensor2(bias_9_conv1, DataType::S8);
        WeightsInfo weights_9_conv1(false, 1, 1, 128, false);
        const TensorShape out_9_conv1_shape(size5, size5, weights_9_conv1.num_kernels());
        out_9_conv1.allocator()->init(TensorInfo(out_9_conv1_shape, 1, DataType::S8));
        out_9_act1.allocator()->init(TensorInfo(out_9_conv1_shape, 1, DataType::S8));
                NPYLoader npy_9_conv2_q, npy_9_conv2_wt, npy_9_conv2_b;
        npy_9_conv2_q.open(Q_table_datapath+Q_table_name[47]); npy_9_conv2_q.init_tensor2(Q_table_9_conv2, DataType::S16);
        npy_9_conv2_wt.open(Q_table_datapath+Q_table_name[47]); npy_9_conv2_wt.init_tensor2(WT_buffer_9_conv2, DataType::U16);
        npy_9_conv2_b.open(bias_datapath+bias_name[47]); npy_9_conv2_b.init_tensor2(bias_9_conv2, DataType::S8);
        WeightsInfo weights_9_conv2(false, 3, 3, 192, false);
        const TensorShape out_9_conv2_shape(size5, size5, weights_9_conv2.num_kernels());
        out_9_conv2.allocator()->init(TensorInfo(out_9_conv2_shape, 1, DataType::S8));
        out_9_act2.allocator()->init(TensorInfo(out_9_conv2_shape, 1, DataType::S8));
                NPYLoader npy_9_conv3_q, npy_9_conv3_wt, npy_9_conv3_b;
        npy_9_conv3_q.open(Q_table_datapath+Q_table_name[48]); npy_9_conv3_q.init_tensor2(Q_table_9_conv3, DataType::S16);
        npy_9_conv3_wt.open(Q_table_datapath+Q_table_name[48]); npy_9_conv3_wt.init_tensor2(WT_buffer_9_conv3, DataType::U16);
        npy_9_conv3_b.open(bias_datapath+bias_name[48]); npy_9_conv3_b.init_tensor2(bias_9_conv3, DataType::S8);
        WeightsInfo weights_9_conv3(false, 1, 1, 160, false);
        const TensorShape out_9_conv3_shape(size5, size5, weights_9_conv3.num_kernels());
        out_9_conv3.allocator()->init(TensorInfo(out_9_conv3_shape, 1, DataType::S8));
        out_9_act3.allocator()->init(TensorInfo(out_9_conv3_shape, 1, DataType::S8));
                NPYLoader npy_9_conv4_q, npy_9_conv4_wt, npy_9_conv4_b;
        npy_9_conv4_q.open(Q_table_datapath+Q_table_name[49]); npy_9_conv4_q.init_tensor2(Q_table_9_conv4, DataType::S16);
        npy_9_conv4_wt.open(Q_table_datapath+Q_table_name[49]); npy_9_conv4_wt.init_tensor2(WT_buffer_9_conv4, DataType::U16);
        npy_9_conv4_b.open(bias_datapath+bias_name[49]); npy_9_conv4_b.init_tensor2(bias_9_conv4, DataType::S8);
        WeightsInfo weights_9_conv4(false, 3, 3, 192, false);
        const TensorShape out_9_conv4_shape(size5, size5, weights_9_conv4.num_kernels());
        out_9_conv4.allocator()->init(TensorInfo(out_9_conv4_shape, 1, DataType::S8));
        out_9_act4.allocator()->init(TensorInfo(out_9_conv4_shape, 1, DataType::S8));
                NPYLoader npy_9_conv5_q, npy_9_conv5_wt, npy_9_conv5_b;
        npy_9_conv5_q.open(Q_table_datapath+Q_table_name[50]); npy_9_conv5_q.init_tensor2(Q_table_9_conv5, DataType::S16);
        npy_9_conv5_wt.open(Q_table_datapath+Q_table_name[50]); npy_9_conv5_wt.init_tensor2(WT_buffer_9_conv5, DataType::U16);
        npy_9_conv5_b.open(bias_datapath+bias_name[50]); npy_9_conv5_b.init_tensor2(bias_9_conv5, DataType::S8);
        WeightsInfo weights_9_conv5(false, 3, 3, 192, false);
        const TensorShape out_9_conv5_shape(size5, size5, weights_9_conv5.num_kernels());
        out_9_conv5.allocator()->init(TensorInfo(out_9_conv5_shape, 1, DataType::S8));
        out_9_act5.allocator()->init(TensorInfo(out_9_conv5_shape, 1, DataType::S8));
                NPYLoader npy_9_conv6_q, npy_9_conv6_wt, npy_9_conv6_b;
        npy_9_conv6_q.open(Q_table_datapath+Q_table_name[51]); npy_9_conv6_q.init_tensor2(Q_table_9_conv6, DataType::S16);
        npy_9_conv6_wt.open(Q_table_datapath+Q_table_name[51]); npy_9_conv6_wt.init_tensor2(WT_buffer_9_conv6, DataType::U16);
        npy_9_conv6_b.open(bias_datapath+bias_name[51]); npy_9_conv6_b.init_tensor2(bias_9_conv6, DataType::S8);
        WeightsInfo weights_9_conv6(false, 1, 1, 128, false);
        const TensorShape out_9_conv6_shape(size5, size5, weights_9_conv6.num_kernels());
        out_9_conv6.allocator()->init(TensorInfo(out_9_conv6_shape, 1, DataType::S8));
        out_9_act6.allocator()->init(TensorInfo(out_9_conv6_shape, 1, DataType::S8));
        TensorShape out_9_cat_shape(size5, size5, weights_9_conv0.num_kernels()+weights_9_conv2.num_kernels()+weights_9_conv5.num_kernels()+weights_9_conv6.num_kernels());
        out_9_cat.allocator()->init(TensorInfo(out_9_cat_shape, 1, DataType::F32));
/**************************************e4***************************************/
        NPYLoader npy_A_conv0_q, npy_A_conv0_wt, npy_A_conv0_b;
        npy_A_conv0_q.open(Q_table_datapath+Q_table_name[52]); npy_A_conv0_q.init_tensor2(Q_table_A_conv0, DataType::S16);
        npy_A_conv0_wt.open(Q_table_datapath+Q_table_name[52]); npy_A_conv0_wt.init_tensor2(WT_buffer_A_conv0, DataType::U16);
        npy_A_conv0_b.open(bias_datapath+bias_name[52]); npy_A_conv0_b.init_tensor2(bias_A_conv0, DataType::S8);
        WeightsInfo weights_A_conv0(false, 1, 1, 480, false);
        const TensorShape out_A_conv0_shape(size5, size5, weights_A_conv0.num_kernels());
        out_A_conv0.allocator()->init(TensorInfo(out_A_conv0_shape, 1, DataType::S8));
        out_A_act0.allocator()->init(TensorInfo(out_A_conv0_shape, 1, DataType::S8));
                NPYLoader npy_A_conv1_q, npy_A_conv1_wt, npy_A_conv1_b;
        npy_A_conv1_q.open(Q_table_datapath+Q_table_name[53]); npy_A_conv1_q.init_tensor2(Q_table_A_conv1, DataType::S16);
        npy_A_conv1_wt.open(Q_table_datapath+Q_table_name[53]); npy_A_conv1_wt.init_tensor2(WT_buffer_A_conv1, DataType::U16);
        npy_A_conv1_b.open(bias_datapath+bias_name[53]); npy_A_conv1_b.init_tensor2(bias_A_conv1, DataType::S8);
        WeightsInfo weights_A_conv1(false, 1, 1, 128, false);
        const TensorShape out_A_conv1_shape(size5, size5, weights_A_conv1.num_kernels());
        out_A_conv1.allocator()->init(TensorInfo(out_A_conv1_shape, 1, DataType::S8));
        out_A_act1.allocator()->init(TensorInfo(out_A_conv1_shape, 1, DataType::S8));
                NPYLoader npy_A_conv2_q, npy_A_conv2_wt, npy_A_conv2_b;
        npy_A_conv2_q.open(Q_table_datapath+Q_table_name[54]); npy_A_conv2_q.init_tensor2(Q_table_A_conv2, DataType::S16);
        npy_A_conv2_wt.open(Q_table_datapath+Q_table_name[54]); npy_A_conv2_wt.init_tensor2(WT_buffer_A_conv2, DataType::U16);
        npy_A_conv2_b.open(bias_datapath+bias_name[54]); npy_A_conv2_b.init_tensor2(bias_A_conv2, DataType::S8);
        WeightsInfo weights_A_conv2(false, 3, 3, 192, false);
        const TensorShape out_A_conv2_shape(size5, size5, weights_A_conv2.num_kernels());
        out_A_conv2.allocator()->init(TensorInfo(out_A_conv2_shape, 1, DataType::S8));
        out_A_act2.allocator()->init(TensorInfo(out_A_conv2_shape, 1, DataType::S8));
                NPYLoader npy_A_conv3_q, npy_A_conv3_wt, npy_A_conv3_b;
        npy_A_conv3_q.open(Q_table_datapath+Q_table_name[55]); npy_A_conv3_q.init_tensor2(Q_table_A_conv3, DataType::S16);
        npy_A_conv3_wt.open(Q_table_datapath+Q_table_name[55]); npy_A_conv3_wt.init_tensor2(WT_buffer_A_conv3, DataType::U16);
        npy_A_conv3_b.open(bias_datapath+bias_name[55]); npy_A_conv3_b.init_tensor2(bias_A_conv3, DataType::S8);
        WeightsInfo weights_A_conv3(false, 1, 1, 192, false);
        const TensorShape out_A_conv3_shape(size5, size5, weights_A_conv3.num_kernels());
        out_A_conv3.allocator()->init(TensorInfo(out_A_conv3_shape, 1, DataType::S8));
        out_A_act3.allocator()->init(TensorInfo(out_A_conv3_shape, 1, DataType::S8));
                NPYLoader npy_A_conv4_q, npy_A_conv4_wt, npy_A_conv4_b;
        npy_A_conv4_q.open(Q_table_datapath+Q_table_name[56]); npy_A_conv4_q.init_tensor2(Q_table_A_conv4, DataType::S16);
        npy_A_conv4_wt.open(Q_table_datapath+Q_table_name[56]); npy_A_conv4_wt.init_tensor2(WT_buffer_A_conv4, DataType::U16);
        npy_A_conv4_b.open(bias_datapath+bias_name[56]); npy_A_conv4_b.init_tensor2(bias_A_conv4, DataType::S8);
        WeightsInfo weights_A_conv4(false, 3, 3, 256, false);
        const TensorShape out_A_conv4_shape(size5, size5, weights_A_conv4.num_kernels());
        out_A_conv4.allocator()->init(TensorInfo(out_A_conv4_shape, 1, DataType::S8));
        out_A_act4.allocator()->init(TensorInfo(out_A_conv4_shape, 1, DataType::S8));
                NPYLoader npy_A_conv5_q, npy_A_conv5_wt, npy_A_conv5_b;
        npy_A_conv5_q.open(Q_table_datapath+Q_table_name[57]); npy_A_conv5_q.init_tensor2(Q_table_A_conv5, DataType::S16);
        npy_A_conv5_wt.open(Q_table_datapath+Q_table_name[57]); npy_A_conv5_wt.init_tensor2(WT_buffer_A_conv5, DataType::U16);
        npy_A_conv5_b.open(bias_datapath+bias_name[57]); npy_A_conv5_b.init_tensor2(bias_A_conv5, DataType::S8);
        WeightsInfo weights_A_conv5(false, 3, 3, 256, false);
        const TensorShape out_A_conv5_shape(size5, size5, weights_A_conv5.num_kernels());
        out_A_conv5.allocator()->init(TensorInfo(out_A_conv5_shape, 1, DataType::S8));
        out_A_act5.allocator()->init(TensorInfo(out_A_conv5_shape, 1, DataType::S8));
                NPYLoader npy_A_conv6_q, npy_A_conv6_wt, npy_A_conv6_b;
        npy_A_conv6_q.open(Q_table_datapath+Q_table_name[58]); npy_A_conv6_q.init_tensor2(Q_table_A_conv6, DataType::S16);
        npy_A_conv6_wt.open(Q_table_datapath+Q_table_name[58]); npy_A_conv6_wt.init_tensor2(WT_buffer_A_conv6, DataType::U16);
        npy_A_conv6_b.open(bias_datapath+bias_name[58]); npy_A_conv6_b.init_tensor2(bias_A_conv6, DataType::S8);
        WeightsInfo weights_A_conv6(false, 1, 1, 128, false);
        const TensorShape out_A_conv6_shape(size5, size5, weights_A_conv6.num_kernels());
        out_A_conv6.allocator()->init(TensorInfo(out_A_conv6_shape, 1, DataType::S8));
        out_A_act6.allocator()->init(TensorInfo(out_A_conv6_shape, 1, DataType::S8));
        TensorShape out_A_cat_shape(size5, size5, weights_A_conv0.num_kernels()+weights_A_conv2.num_kernels()+weights_A_conv5.num_kernels()+weights_A_conv6.num_kernels());
        out_A_cat.allocator()->init(TensorInfo(out_A_cat_shape, 1, DataType::F32));
/*******************************maxpool*************************/
        TensorShape out_B_pool0_shape=out_A_cat_shape;
        out_B_pool0_shape.set(0, out_B_pool0_shape.x()/2);
        out_B_pool0_shape.set(1, out_B_pool0_shape.y()/2);
        out_B_pool0.allocator()->init(TensorInfo(out_B_pool0_shape, 1, DataType::F32));
/******************************a5**************************************************/
        NPYLoader npy_C_conv0_q, npy_C_conv0_wt, npy_C_conv0_b;
        npy_C_conv0_q.open(Q_table_datapath+Q_table_name[59]); npy_C_conv0_q.init_tensor2(Q_table_C_conv0, DataType::S16);
        npy_C_conv0_wt.open(Q_table_datapath+Q_table_name[59]); npy_C_conv0_wt.init_tensor2(WT_buffer_C_conv0, DataType::U16);
        npy_C_conv0_b.open(bias_datapath+bias_name[59]); npy_C_conv0_b.init_tensor2(bias_C_conv0, DataType::S8);
        WeightsInfo weights_C_conv0(false, 1, 1, 352, false);
        const TensorShape out_C_conv0_shape(size6, size6, weights_C_conv0.num_kernels());
        out_C_conv0.allocator()->init(TensorInfo(out_C_conv0_shape, 1, DataType::S8));
        out_C_act0.allocator()->init(TensorInfo(out_C_conv0_shape, 1, DataType::S8));
                NPYLoader npy_C_conv1_q, npy_C_conv1_wt, npy_C_conv1_b;
        npy_C_conv1_q.open(Q_table_datapath+Q_table_name[60]); npy_C_conv1_q.init_tensor2(Q_table_C_conv1, DataType::S16);
        npy_C_conv1_wt.open(Q_table_datapath+Q_table_name[60]); npy_C_conv1_wt.init_tensor2(WT_buffer_C_conv1, DataType::U16);
        npy_C_conv1_b.open(bias_datapath+bias_name[60]); npy_C_conv1_b.init_tensor2(bias_C_conv1, DataType::S8);
        WeightsInfo weights_C_conv1(false, 1, 1, 192, false);
        const TensorShape out_C_conv1_shape(size6, size6, weights_C_conv1.num_kernels());
        out_C_conv1.allocator()->init(TensorInfo(out_C_conv1_shape, 1, DataType::S8));
        out_C_act1.allocator()->init(TensorInfo(out_C_conv1_shape, 1, DataType::S8));
                NPYLoader npy_C_conv2_q, npy_C_conv2_wt, npy_C_conv2_b;
        npy_C_conv2_q.open(Q_table_datapath+Q_table_name[61]); npy_C_conv2_q.init_tensor2(Q_table_C_conv2, DataType::S16);
        npy_C_conv2_wt.open(Q_table_datapath+Q_table_name[61]); npy_C_conv2_wt.init_tensor2(WT_buffer_C_conv2, DataType::U16);
        npy_C_conv2_b.open(bias_datapath+bias_name[61]); npy_C_conv2_b.init_tensor2(bias_C_conv2, DataType::S8);
        WeightsInfo weights_C_conv2(false, 3, 3, 320, false);
        const TensorShape out_C_conv2_shape(size6, size6, weights_C_conv2.num_kernels());
        out_C_conv2.allocator()->init(TensorInfo(out_C_conv2_shape, 1, DataType::S8));
        out_C_act2.allocator()->init(TensorInfo(out_C_conv2_shape, 1, DataType::S8));
                NPYLoader npy_C_conv3_q, npy_C_conv3_wt, npy_C_conv3_b;
        npy_C_conv3_q.open(Q_table_datapath+Q_table_name[62]); npy_C_conv3_q.init_tensor2(Q_table_C_conv3, DataType::S16);
        npy_C_conv3_wt.open(Q_table_datapath+Q_table_name[62]); npy_C_conv3_wt.init_tensor2(WT_buffer_C_conv3, DataType::U16);
        npy_C_conv3_b.open(bias_datapath+bias_name[62]); npy_C_conv3_b.init_tensor2(bias_C_conv3, DataType::S8);
        WeightsInfo weights_C_conv3(false, 1, 1, 160, false);
        const TensorShape out_C_conv3_shape(size6, size6, weights_C_conv3.num_kernels());
        out_C_conv3.allocator()->init(TensorInfo(out_C_conv3_shape, 1, DataType::S8));
        out_C_act3.allocator()->init(TensorInfo(out_C_conv3_shape, 1, DataType::S8));
                NPYLoader npy_C_conv4_q, npy_C_conv4_wt, npy_C_conv4_b;
        npy_C_conv4_q.open(Q_table_datapath+Q_table_name[63]); npy_C_conv4_q.init_tensor2(Q_table_C_conv4, DataType::S16);
        npy_C_conv4_wt.open(Q_table_datapath+Q_table_name[63]); npy_C_conv4_wt.init_tensor2(WT_buffer_C_conv4, DataType::U16);
        npy_C_conv4_b.open(bias_datapath+bias_name[63]); npy_C_conv4_b.init_tensor2(bias_C_conv4, DataType::S8);
        WeightsInfo weights_C_conv4(false, 3, 3, 224, false);
        const TensorShape out_C_conv4_shape(size6, size6, weights_C_conv4.num_kernels());
        out_C_conv4.allocator()->init(TensorInfo(out_C_conv4_shape, 1, DataType::S8));
        out_C_act4.allocator()->init(TensorInfo(out_C_conv4_shape, 1, DataType::S8));
                NPYLoader npy_C_conv5_q, npy_C_conv5_wt, npy_C_conv5_b;
        npy_C_conv5_q.open(Q_table_datapath+Q_table_name[64]); npy_C_conv5_q.init_tensor2(Q_table_C_conv5, DataType::S16);
        npy_C_conv5_wt.open(Q_table_datapath+Q_table_name[64]); npy_C_conv5_wt.init_tensor2(WT_buffer_C_conv5, DataType::U16);
        npy_C_conv5_b.open(bias_datapath+bias_name[64]); npy_C_conv5_b.init_tensor2(bias_C_conv5, DataType::S8);
        WeightsInfo weights_C_conv5(false, 3, 3, 224, false);
        const TensorShape out_C_conv5_shape(size6, size6, weights_C_conv5.num_kernels());
        out_C_conv5.allocator()->init(TensorInfo(out_C_conv5_shape, 1, DataType::S8));
        out_C_act5.allocator()->init(TensorInfo(out_C_conv5_shape, 1, DataType::S8));
                NPYLoader npy_C_conv6_q, npy_C_conv6_wt, npy_C_conv6_b;
        npy_C_conv6_q.open(Q_table_datapath+Q_table_name[65]); npy_C_conv6_q.init_tensor2(Q_table_C_conv6, DataType::S16);
        npy_C_conv6_wt.open(Q_table_datapath+Q_table_name[65]); npy_C_conv6_wt.init_tensor2(WT_buffer_C_conv6, DataType::U16);
        npy_C_conv6_b.open(bias_datapath+bias_name[65]); npy_C_conv6_b.init_tensor2(bias_C_conv6, DataType::S8);
        WeightsInfo weights_C_conv6(false, 1, 1, 128, false);
        const TensorShape out_C_conv6_shape(size6, size6, weights_C_conv6.num_kernels());
        out_C_conv6.allocator()->init(TensorInfo(out_C_conv6_shape, 1, DataType::S8));
        out_C_act6.allocator()->init(TensorInfo(out_C_conv6_shape, 1, DataType::S8));
        TensorShape out_C_cat_shape(size6, size6, weights_C_conv0.num_kernels()+weights_C_conv2.num_kernels()+weights_C_conv5.num_kernels()+weights_C_conv6.num_kernels());
        out_C_cat.allocator()->init(TensorInfo(out_C_cat_shape, 1, DataType::F32));
/******************************b5**************************************************/
        NPYLoader npy_D_conv0_q, npy_D_conv0_wt, npy_D_conv0_b;
        npy_D_conv0_q.open(Q_table_datapath+Q_table_name[66]); npy_D_conv0_q.init_tensor2(Q_table_D_conv0, DataType::S16);
        npy_D_conv0_wt.open(Q_table_datapath+Q_table_name[66]); npy_D_conv0_wt.init_tensor2(WT_buffer_D_conv0, DataType::U16);
        npy_D_conv0_b.open(bias_datapath+bias_name[66]); npy_D_conv0_b.init_tensor2(bias_D_conv0, DataType::S8);
        WeightsInfo weights_D_conv0(false, 1, 1, 352, false);
        const TensorShape out_D_conv0_shape(size6, size6, weights_D_conv0.num_kernels());
        out_D_conv0.allocator()->init(TensorInfo(out_D_conv0_shape, 1, DataType::S8));
        out_D_act0.allocator()->init(TensorInfo(out_D_conv0_shape, 1, DataType::S8));
                NPYLoader npy_D_conv1_q, npy_D_conv1_wt, npy_D_conv1_b;
        npy_D_conv1_q.open(Q_table_datapath+Q_table_name[67]); npy_D_conv1_q.init_tensor2(Q_table_D_conv1, DataType::S16);
        npy_D_conv1_wt.open(Q_table_datapath+Q_table_name[67]); npy_D_conv1_wt.init_tensor2(WT_buffer_D_conv1, DataType::U16);
        npy_D_conv1_b.open(bias_datapath+bias_name[67]); npy_D_conv1_b.init_tensor2(bias_D_conv1, DataType::S8);
        WeightsInfo weights_D_conv1(false, 1, 1, 192, false);
        const TensorShape out_D_conv1_shape(size6, size6, weights_D_conv1.num_kernels());
        out_D_conv1.allocator()->init(TensorInfo(out_D_conv1_shape, 1, DataType::S8));
        out_D_act1.allocator()->init(TensorInfo(out_D_conv1_shape, 1, DataType::S8));
                NPYLoader npy_D_conv2_q, npy_D_conv2_wt, npy_D_conv2_b;
        npy_D_conv2_q.open(Q_table_datapath+Q_table_name[68]); npy_D_conv2_q.init_tensor2(Q_table_D_conv2, DataType::S16);
        npy_D_conv2_wt.open(Q_table_datapath+Q_table_name[68]); npy_D_conv2_wt.init_tensor2(WT_buffer_D_conv2, DataType::U16);
        npy_D_conv2_b.open(bias_datapath+bias_name[68]); npy_D_conv2_b.init_tensor2(bias_D_conv2, DataType::S8);
        WeightsInfo weights_D_conv2(false, 3, 3, 320, false);
        const TensorShape out_D_conv2_shape(size6, size6, weights_D_conv2.num_kernels());
        out_D_conv2.allocator()->init(TensorInfo(out_D_conv2_shape, 1, DataType::S8));
        out_D_act2.allocator()->init(TensorInfo(out_D_conv2_shape, 1, DataType::S8));
                NPYLoader npy_D_conv3_q, npy_D_conv3_wt, npy_D_conv3_b;
        npy_D_conv3_q.open(Q_table_datapath+Q_table_name[69]); npy_D_conv3_q.init_tensor2(Q_table_D_conv3, DataType::S16);
        npy_D_conv3_wt.open(Q_table_datapath+Q_table_name[69]); npy_D_conv3_wt.init_tensor2(WT_buffer_D_conv3, DataType::U16);
        npy_D_conv3_b.open(bias_datapath+bias_name[69]); npy_D_conv3_b.init_tensor2(bias_D_conv3, DataType::S8);
        WeightsInfo weights_D_conv3(false, 1, 1, 160, false);
        const TensorShape out_D_conv3_shape(size6, size6, weights_D_conv3.num_kernels());
        out_D_conv3.allocator()->init(TensorInfo(out_D_conv3_shape, 1, DataType::S8));
        out_D_act3.allocator()->init(TensorInfo(out_D_conv3_shape, 1, DataType::S8));
                NPYLoader npy_D_conv4_q, npy_D_conv4_wt, npy_D_conv4_b;
        npy_D_conv4_q.open(Q_table_datapath+Q_table_name[70]); npy_D_conv4_q.init_tensor2(Q_table_D_conv4, DataType::S16);
        npy_D_conv4_wt.open(Q_table_datapath+Q_table_name[70]); npy_D_conv4_wt.init_tensor2(WT_buffer_D_conv4, DataType::U16);
        npy_D_conv4_b.open(bias_datapath+bias_name[70]); npy_D_conv4_b.init_tensor2(bias_D_conv4, DataType::S8);
        WeightsInfo weights_D_conv4(false, 3, 3, 224, false);
        const TensorShape out_D_conv4_shape(size6, size6, weights_D_conv4.num_kernels());
        out_D_conv4.allocator()->init(TensorInfo(out_D_conv4_shape, 1, DataType::S8));
        out_D_act4.allocator()->init(TensorInfo(out_D_conv4_shape, 1, DataType::S8));
                NPYLoader npy_D_conv5_q, npy_D_conv5_wt, npy_D_conv5_b;
        npy_D_conv5_q.open(Q_table_datapath+Q_table_name[71]); npy_D_conv5_q.init_tensor2(Q_table_D_conv5, DataType::S16);
        npy_D_conv5_wt.open(Q_table_datapath+Q_table_name[71]); npy_D_conv5_wt.init_tensor2(WT_buffer_D_conv5, DataType::U16);
        npy_D_conv5_b.open(bias_datapath+bias_name[71]); npy_D_conv5_b.init_tensor2(bias_D_conv5, DataType::S8);
        WeightsInfo weights_D_conv5(false, 3, 3, 224, false);
        const TensorShape out_D_conv5_shape(size6, size6, weights_D_conv5.num_kernels());
        out_D_conv5.allocator()->init(TensorInfo(out_D_conv5_shape, 1, DataType::S8));
        out_D_act5.allocator()->init(TensorInfo(out_D_conv5_shape, 1, DataType::S8));
                NPYLoader npy_D_conv6_q, npy_D_conv6_wt, npy_D_conv6_b;
        npy_D_conv6_q.open(Q_table_datapath+Q_table_name[72]); npy_D_conv6_q.init_tensor2(Q_table_D_conv6, DataType::S16);
        npy_D_conv6_wt.open(Q_table_datapath+Q_table_name[72]); npy_D_conv6_wt.init_tensor2(WT_buffer_D_conv6, DataType::U16);
        npy_D_conv6_b.open(bias_datapath+bias_name[72]); npy_D_conv6_b.init_tensor2(bias_D_conv6, DataType::S8);
        WeightsInfo weights_D_conv6(false, 1, 1, 128, false);
        const TensorShape out_D_conv6_shape(size6, size6, weights_D_conv6.num_kernels());
        out_D_conv6.allocator()->init(TensorInfo(out_D_conv6_shape, 1, DataType::S8));
        out_D_act6.allocator()->init(TensorInfo(out_D_conv6_shape, 1, DataType::S8));
        TensorShape out_D_cat_shape(size6, size6, weights_D_conv0.num_kernels()+weights_D_conv2.num_kernels()+weights_D_conv5.num_kernels()+weights_D_conv6.num_kernels());
        out_D_cat.allocator()->init(TensorInfo(out_D_cat_shape, 1, DataType::F32));
/************************************avgpool***********************/
        TensorShape out_E_pool0_shape=out_D_cat_shape;
        out_E_pool0_shape.set(0, out_E_pool0_shape.x()/7);
        out_E_pool0_shape.set(1, out_E_pool0_shape.y()/7);
        out_E_pool0.allocator()->init(TensorInfo(out_E_pool0_shape, 1, DataType::S8));
/**************************************fc****************************/
        NPYLoader npy_F_conv0_q, npy_F_conv0_wt, npy_F_conv0_b;
        npy_F_conv0_q.open(Q_table_datapath+Q_table_name[73]); npy_F_conv0_q.init_tensor2(Q_table_F_conv0, DataType::S16);
        npy_F_conv0_wt.open(Q_table_datapath+Q_table_name[73]); npy_F_conv0_wt.init_tensor2(WT_buffer_F_conv0, DataType::U16);
        npy_F_conv0_b.open(bias_datapath+bias_name[73]); npy_F_conv0_b.init_tensor2(bias_F_conv0, DataType::S8);
        WeightsInfo weights_F_conv0(false, 1, 1, 1000, false);
        const TensorShape out_F_conv0_shape(size7, size7, weights_F_conv0.num_kernels());

/*configure*/
/*pre_layers*/
        _1_conv0.configure(&src, &Q_table_1_conv0, &WT_buffer_1_conv0, &bias_1_conv0, &out_1_conv0, PadStrideInfo(2,2,3,3), weights_1_conv0, precision[0], 1);
        _1_sf0.configure(&out_1_conv0, &out_1_sf0);
        _1_pool0.configure(&out_1_sf0, &out_1_pool0, PoolingLayerInfo(PoolingType::MAX, 2, PadStrideInfo(2, 2, 0, 0)));
        _1_fs0.configure(&out_1_pool0, &out_1_fs0);
        _1_conv1.configure(&out_1_fs0, &Q_table_1_conv1, &WT_buffer_1_conv1, &bias_1_conv1, &out_1_conv1, PadStrideInfo(1,1,0,0), weights_1_conv1, precision[1], 1);
        _1_conv2.configure(&out_1_conv1, &Q_table_1_conv2, &WT_buffer_1_conv2, &bias_1_conv2, &out_1_conv2, PadStrideInfo(1,1,1,1), weights_1_conv2, precision[2], 1);
        _1_act0.configure(&out_1_conv2, &out_1_act0,ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        _1_sf1.configure(&out_1_act0, &out_1_sf1);
        _1_pool1.configure(&out_1_sf1, &out_1_pool1, PoolingLayerInfo(PoolingType::MAX, 2, PadStrideInfo(2, 2, 0, 0)));
        _1_fs1.configure(&out_1_pool1, &out_1_fs1);

/*a3*/
        _2_conv0.configure(&out_1_fs1, &Q_table_2_conv0, &WT_buffer_2_conv0, &bias_2_conv0, &out_2_conv0, PadStrideInfo(1,1,0,0), weights_2_conv0, precision[3], 1);
        _2_act0.configure(&out_2_conv0, &out_2_act0, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        _2_conv1.configure(&out_1_fs1, &Q_table_2_conv1, &WT_buffer_2_conv1, &bias_2_conv1, &out_2_conv1, PadStrideInfo(1,1,0,0), weights_2_conv1, precision[4], 1);
        _2_act1.configure(&out_2_conv1, &out_2_act1, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        _2_conv2.configure(&out_2_act1, &Q_table_2_conv2, &WT_buffer_2_conv2, &bias_2_conv2, &out_2_conv2, PadStrideInfo(1,1,1,1), weights_2_conv2, precision[5], 1);
        _2_act2.configure(&out_2_conv2, &out_2_act2, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        _2_conv3.configure(&out_1_fs1, &Q_table_2_conv3, &WT_buffer_2_conv3, &bias_2_conv3, &out_2_conv3, PadStrideInfo(1,1,0,0), weights_2_conv3, precision[6], 1);
        _2_act3.configure(&out_2_conv3, &out_2_act3, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        _2_conv4.configure(&out_2_act3, &Q_table_2_conv4, &WT_buffer_2_conv4, &bias_2_conv4, &out_2_conv4, PadStrideInfo(1,1,1,1), weights_2_conv4, precision[7], 1);
        _2_act4.configure(&out_2_conv4, &out_2_act4, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        _2_conv5.configure(&out_2_act4, &Q_table_2_conv5, &WT_buffer_2_conv5, &bias_2_conv5, &out_2_conv5, PadStrideInfo(1,1,1,1), weights_2_conv5, precision[8], 1);
        _2_act5.configure(&out_2_conv5, &out_2_act5, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        _2_conv6.configure(&out_1_fs1, &Q_table_2_conv6, &WT_buffer_2_conv6, &bias_2_conv6, &out_2_conv6, PadStrideInfo(1,1,0,0), weights_2_conv6, precision[9], 1);
        _2_act6.configure(&out_2_conv6, &out_2_act6, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        _2_sf0.configure(&out_2_act0, &out_2_sf0); std::vector<ITensor*> v2; v2.push_back(&out_2_sf0);
        _2_sf1.configure(&out_2_act2, &out_2_sf1);v2.push_back(&out_2_sf1);
        _2_sf2.configure(&out_2_act5, &out_2_sf2);v2.push_back(&out_2_sf2);
        _2_sf3.configure(&out_2_act6, &out_2_sf3);v2.push_back(&out_2_sf3);
        _2_cat.configure(v2, &out_2_cat, 2);
        _2_fs0.configure(&out_2_cat, &out_2_fs0);
/*b3*/
        _3_conv0.configure(&out_2_fs0, &Q_table_3_conv0, &WT_buffer_3_conv0, &bias_3_conv0, &out_3_conv0, PadStrideInfo(1,1,0,0), weights_3_conv0, precision[10], 1);
        _3_act0.configure(&out_3_conv0, &out_3_act0, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        _3_conv1.configure(&out_2_fs0, &Q_table_3_conv1, &WT_buffer_3_conv1, &bias_3_conv1, &out_3_conv1, PadStrideInfo(1,1,0,0), weights_3_conv1, precision[11], 1);
        _3_act1.configure(&out_3_conv1, &out_3_act1, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        _3_conv2.configure(&out_3_act1, &Q_table_3_conv2, &WT_buffer_3_conv2, &bias_3_conv2, &out_3_conv2, PadStrideInfo(1,1,1,1), weights_3_conv2, precision[12], 1);
        _3_act2.configure(&out_3_conv2, &out_3_act2, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        _3_conv3.configure(&out_2_fs0, &Q_table_3_conv3, &WT_buffer_3_conv3, &bias_3_conv3, &out_3_conv3, PadStrideInfo(1,1,0,0), weights_3_conv3, precision[13], 1);
        _3_act3.configure(&out_3_conv3, &out_3_act3, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        _3_conv4.configure(&out_3_act3, &Q_table_3_conv4, &WT_buffer_3_conv4, &bias_3_conv4, &out_3_conv4, PadStrideInfo(1,1,1,1), weights_3_conv4, precision[14], 1);
        _3_act4.configure(&out_3_conv4, &out_3_act4, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        _3_conv5.configure(&out_3_act4, &Q_table_3_conv5, &WT_buffer_3_conv5, &bias_3_conv5, &out_3_conv5, PadStrideInfo(1,1,1,1), weights_3_conv5, precision[15], 1);
        _3_act5.configure(&out_3_conv5, &out_3_act5, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        _3_conv6.configure(&out_2_fs0, &Q_table_3_conv6, &WT_buffer_3_conv6, &bias_3_conv6, &out_3_conv6, PadStrideInfo(1,1,0,0), weights_3_conv6, precision[16], 1);
        _3_act6.configure(&out_3_conv6, &out_3_act6, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        _3_sf0.configure(&out_3_act0, &out_3_sf0); std::vector<ITensor*> v3; v3.push_back(&out_3_sf0);
        _3_sf1.configure(&out_3_act2, &out_3_sf1);v3.push_back(&out_3_sf1);
        _3_sf2.configure(&out_3_act5, &out_3_sf2);v3.push_back(&out_3_sf2);
        _3_sf3.configure(&out_3_act6, &out_3_sf3);v3.push_back(&out_3_sf3);
        _3_cat.configure(v3, &out_3_cat, 2);
        _3_fs0.configure(&out_3_cat, &out_3_fs0);
/*c3*/
        _4_conv0.configure(&out_3_fs0, &Q_table_4_conv0, &WT_buffer_4_conv0, &bias_4_conv0, &out_4_conv0, PadStrideInfo(1,1,0,0), weights_4_conv0, precision[17], 1);
        _4_act0.configure(&out_4_conv0, &out_4_act0, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        _4_conv1.configure(&out_3_fs0, &Q_table_4_conv1, &WT_buffer_4_conv1, &bias_4_conv1, &out_4_conv1, PadStrideInfo(1,1,0,0), weights_4_conv1, precision[18], 1);
        _4_act1.configure(&out_4_conv1, &out_4_act1, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        _4_conv2.configure(&out_4_act1, &Q_table_4_conv2, &WT_buffer_4_conv2, &bias_4_conv2, &out_4_conv2, PadStrideInfo(1,1,1,1), weights_4_conv2, precision[19], 1);
        _4_act2.configure(&out_4_conv2, &out_4_act2, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        _4_conv3.configure(&out_3_fs0, &Q_table_4_conv3, &WT_buffer_4_conv3, &bias_4_conv3, &out_4_conv3, PadStrideInfo(1,1,0,0), weights_4_conv3, precision[20], 1);
        _4_act3.configure(&out_4_conv3, &out_4_act3, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        _4_conv4.configure(&out_4_act3, &Q_table_4_conv4, &WT_buffer_4_conv4, &bias_4_conv4, &out_4_conv4, PadStrideInfo(1,1,1,1), weights_4_conv4, precision[21], 1);
        _4_act4.configure(&out_4_conv4, &out_4_act4, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        _4_conv5.configure(&out_4_act4, &Q_table_4_conv5, &WT_buffer_4_conv5, &bias_4_conv5, &out_4_conv5, PadStrideInfo(1,1,1,1), weights_4_conv5, precision[22], 1);
        _4_act5.configure(&out_4_conv5, &out_4_act5, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        _4_conv6.configure(&out_3_fs0, &Q_table_4_conv6, &WT_buffer_4_conv6, &bias_4_conv6, &out_4_conv6, PadStrideInfo(1,1,0,0), weights_4_conv6, precision[23], 1);
        _4_act6.configure(&out_4_conv6, &out_4_act6, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        _4_sf0.configure(&out_4_act0, &out_4_sf0); std::vector<ITensor*> v4; v4.push_back(&out_4_sf0);
        _4_sf1.configure(&out_4_act2, &out_4_sf1);v4.push_back(&out_4_sf1);
        _4_sf2.configure(&out_4_act5, &out_4_sf2);v4.push_back(&out_4_sf2);
        _4_sf3.configure(&out_4_act6, &out_4_sf3);v4.push_back(&out_4_sf3);
        _4_cat.configure(v4, &out_4_cat, 2);
        _4_fs0.configure(&out_4_cat, &out_4_fs0);
/*maxpool*/
        _5_sf0.configure(&out_4_fs0, &out_5_sf0);
        _5_pool0.configure(&out_5_sf0, &out_5_pool0, PoolingLayerInfo(PoolingType::MAX, 2, PadStrideInfo(2, 2, 0, 0)));
        _5_fs0.configure(&out_5_pool0, &out_5_fs0);
/*a4*/
        _6_conv0.configure(&out_5_fs0, &Q_table_6_conv0, &WT_buffer_6_conv0, &bias_6_conv0, &out_6_conv0, PadStrideInfo(1,1,0,0), weights_6_conv0, precision[24], 1);
        _6_act0.configure(&out_6_conv0, &out_6_act0, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        _6_conv1.configure(&out_5_fs0, &Q_table_6_conv1, &WT_buffer_6_conv1, &bias_6_conv1, &out_6_conv1, PadStrideInfo(1,1,0,0), weights_6_conv1, precision[25], 1);
        _6_act1.configure(&out_6_conv1, &out_6_act1, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        _6_conv2.configure(&out_6_act1, &Q_table_6_conv2, &WT_buffer_6_conv2, &bias_6_conv2, &out_6_conv2, PadStrideInfo(1,1,1,1), weights_6_conv2, precision[26], 1);
        _6_act2.configure(&out_6_conv2, &out_6_act2, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        _6_conv3.configure(&out_5_fs0, &Q_table_6_conv3, &WT_buffer_6_conv3, &bias_6_conv3, &out_6_conv3, PadStrideInfo(1,1,0,0), weights_6_conv3, precision[27], 1);
        _6_act3.configure(&out_6_conv3, &out_6_act3, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        _6_conv4.configure(&out_6_act3, &Q_table_6_conv4, &WT_buffer_6_conv4, &bias_6_conv4, &out_6_conv4, PadStrideInfo(1,1,1,1), weights_6_conv4, precision[28], 1);
        _6_act4.configure(&out_6_conv4, &out_6_act4, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        _6_conv5.configure(&out_6_act4, &Q_table_6_conv5, &WT_buffer_6_conv5, &bias_6_conv5, &out_6_conv5, PadStrideInfo(1,1,1,1), weights_6_conv5, precision[29], 1);
        _6_act5.configure(&out_6_conv5, &out_6_act5, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        _6_conv6.configure(&out_5_fs0, &Q_table_6_conv6, &WT_buffer_6_conv6, &bias_6_conv6, &out_6_conv6, PadStrideInfo(1,1,0,0), weights_6_conv6, precision[30], 1);
        _6_act6.configure(&out_6_conv6, &out_6_act6, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        _6_sf0.configure(&out_6_act0, &out_6_sf0); std::vector<ITensor*> v6; v6.push_back(&out_6_sf0);
        _6_sf1.configure(&out_6_act2, &out_6_sf1);v6.push_back(&out_6_sf1);
        _6_sf2.configure(&out_6_act5, &out_6_sf2);v6.push_back(&out_6_sf2);
        _6_sf3.configure(&out_6_act6, &out_6_sf3);v6.push_back(&out_6_sf3);
        _6_cat.configure(v6, &out_6_cat, 2);
        _6_fs0.configure(&out_6_cat, &out_6_fs0);       
/*b4*/
        _7_conv0.configure(&out_6_fs0, &Q_table_7_conv0, &WT_buffer_7_conv0, &bias_7_conv0, &out_7_conv0, PadStrideInfo(1,1,0,0), weights_7_conv0, precision[31], 1);
        _7_act0.configure(&out_7_conv0, &out_7_act0, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        _7_conv1.configure(&out_6_fs0, &Q_table_7_conv1, &WT_buffer_7_conv1, &bias_7_conv1, &out_7_conv1, PadStrideInfo(1,1,0,0), weights_7_conv1, precision[32], 1);
        _7_act1.configure(&out_7_conv1, &out_7_act1, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        _7_conv2.configure(&out_7_act1, &Q_table_7_conv2, &WT_buffer_7_conv2, &bias_7_conv2, &out_7_conv2, PadStrideInfo(1,1,1,1), weights_7_conv2, precision[33], 1);
        _7_act2.configure(&out_7_conv2, &out_7_act2, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        _7_conv3.configure(&out_6_fs0, &Q_table_7_conv3, &WT_buffer_7_conv3, &bias_7_conv3, &out_7_conv3, PadStrideInfo(1,1,0,0), weights_7_conv3, precision[34], 1);
        _7_act3.configure(&out_7_conv3, &out_7_act3, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        _7_conv4.configure(&out_7_act3, &Q_table_7_conv4, &WT_buffer_7_conv4, &bias_7_conv4, &out_7_conv4, PadStrideInfo(1,1,1,1), weights_7_conv4, precision[35], 1);
        _7_act4.configure(&out_7_conv4, &out_7_act4, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        _7_conv5.configure(&out_7_act4, &Q_table_7_conv5, &WT_buffer_7_conv5, &bias_7_conv5, &out_7_conv5, PadStrideInfo(1,1,1,1), weights_7_conv5, precision[36], 1);
        _7_act5.configure(&out_7_conv5, &out_7_act5, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        _7_conv6.configure(&out_6_fs0, &Q_table_7_conv6, &WT_buffer_7_conv6, &bias_7_conv6, &out_7_conv6, PadStrideInfo(1,1,0,0), weights_7_conv6, precision[37], 1);
        _7_act6.configure(&out_7_conv6, &out_7_act6, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        _7_sf0.configure(&out_7_act0, &out_7_sf0); std::vector<ITensor*> v7; v7.push_back(&out_7_sf0);
        _7_sf1.configure(&out_7_act2, &out_7_sf1);v7.push_back(&out_7_sf1);
        _7_sf2.configure(&out_7_act5, &out_7_sf2);v7.push_back(&out_7_sf2);
        _7_sf3.configure(&out_7_act6, &out_7_sf3);v7.push_back(&out_7_sf3);
        _7_cat.configure(v7, &out_7_cat, 2);
        _7_fs0.configure(&out_7_cat, &out_7_fs0);              
/*c4*/
        _8_conv0.configure(&out_7_fs0, &Q_table_8_conv0, &WT_buffer_8_conv0, &bias_8_conv0, &out_8_conv0, PadStrideInfo(1,1,0,0), weights_8_conv0, precision[38], 1);
        _8_act0.configure(&out_8_conv0, &out_8_act0, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        _8_conv1.configure(&out_7_fs0, &Q_table_8_conv1, &WT_buffer_8_conv1, &bias_8_conv1, &out_8_conv1, PadStrideInfo(1,1,0,0), weights_8_conv1, precision[39], 1);
        _8_act1.configure(&out_8_conv1, &out_8_act1, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        _8_conv2.configure(&out_8_act1, &Q_table_8_conv2, &WT_buffer_8_conv2, &bias_8_conv2, &out_8_conv2, PadStrideInfo(1,1,1,1), weights_8_conv2, precision[40], 1);
        _8_act2.configure(&out_8_conv2, &out_8_act2, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        _8_conv3.configure(&out_7_fs0, &Q_table_8_conv3, &WT_buffer_8_conv3, &bias_8_conv3, &out_8_conv3, PadStrideInfo(1,1,0,0), weights_8_conv3, precision[41], 1);
        _8_act3.configure(&out_8_conv3, &out_8_act3, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        _8_conv4.configure(&out_8_act3, &Q_table_8_conv4, &WT_buffer_8_conv4, &bias_8_conv4, &out_8_conv4, PadStrideInfo(1,1,1,1), weights_8_conv4, precision[42], 1);
        _8_act4.configure(&out_8_conv4, &out_8_act4, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        _8_conv5.configure(&out_8_act4, &Q_table_8_conv5, &WT_buffer_8_conv5, &bias_8_conv5, &out_8_conv5, PadStrideInfo(1,1,1,1), weights_8_conv5, precision[43], 1);
        _8_act5.configure(&out_8_conv5, &out_8_act5, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        _8_conv6.configure(&out_7_fs0, &Q_table_8_conv6, &WT_buffer_8_conv6, &bias_8_conv6, &out_8_conv6, PadStrideInfo(1,1,0,0), weights_8_conv6, precision[44], 1);
        _8_act6.configure(&out_8_conv6, &out_8_act6, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        _8_sf0.configure(&out_8_act0, &out_8_sf0); std::vector<ITensor*> v8; v8.push_back(&out_8_sf0);
        _8_sf1.configure(&out_8_act2, &out_8_sf1);v8.push_back(&out_8_sf1);
        _8_sf2.configure(&out_8_act5, &out_8_sf2);v8.push_back(&out_8_sf2);
        _8_sf3.configure(&out_8_act6, &out_8_sf3);v8.push_back(&out_8_sf3);
        _8_cat.configure(v8, &out_8_cat, 2);
        _8_fs0.configure(&out_8_cat, &out_8_fs0);           
/*d4*/
        _9_conv0.configure(&out_8_fs0, &Q_table_9_conv0, &WT_buffer_9_conv0, &bias_9_conv0, &out_9_conv0, PadStrideInfo(1,1,0,0), weights_9_conv0, precision[45], 1);
        _9_act0.configure(&out_9_conv0, &out_9_act0, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        _9_conv1.configure(&out_8_fs0, &Q_table_9_conv1, &WT_buffer_9_conv1, &bias_9_conv1, &out_9_conv1, PadStrideInfo(1,1,0,0), weights_9_conv1, precision[46], 1);
        _9_act1.configure(&out_9_conv1, &out_9_act1, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        _9_conv2.configure(&out_9_act1, &Q_table_9_conv2, &WT_buffer_9_conv2, &bias_9_conv2, &out_9_conv2, PadStrideInfo(1,1,1,1), weights_9_conv2, precision[47], 1);
        _9_act2.configure(&out_9_conv2, &out_9_act2, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        _9_conv3.configure(&out_8_fs0, &Q_table_9_conv3, &WT_buffer_9_conv3, &bias_9_conv3, &out_9_conv3, PadStrideInfo(1,1,0,0), weights_9_conv3, precision[48], 1);
        _9_act3.configure(&out_9_conv3, &out_9_act3, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        _9_conv4.configure(&out_9_act3, &Q_table_9_conv4, &WT_buffer_9_conv4, &bias_9_conv4, &out_9_conv4, PadStrideInfo(1,1,1,1), weights_9_conv4, precision[49], 1);
        _9_act4.configure(&out_9_conv4, &out_9_act4, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        _9_conv5.configure(&out_9_act4, &Q_table_9_conv5, &WT_buffer_9_conv5, &bias_9_conv5, &out_9_conv5, PadStrideInfo(1,1,1,1), weights_9_conv5, precision[50], 1);
        _9_act5.configure(&out_9_conv5, &out_9_act5, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        _9_conv6.configure(&out_8_fs0, &Q_table_9_conv6, &WT_buffer_9_conv6, &bias_9_conv6, &out_9_conv6, PadStrideInfo(1,1,0,0), weights_9_conv6, precision[51], 1);
        _9_act6.configure(&out_9_conv6, &out_9_act6, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        _9_sf0.configure(&out_9_act0, &out_9_sf0); std::vector<ITensor*> v9; v9.push_back(&out_9_sf0);
        _9_sf1.configure(&out_9_act2, &out_9_sf1);v9.push_back(&out_9_sf1);
        _9_sf2.configure(&out_9_act5, &out_9_sf2);v9.push_back(&out_9_sf2);
        _9_sf3.configure(&out_9_act6, &out_9_sf3);v9.push_back(&out_9_sf3);
        _9_cat.configure(v9, &out_9_cat, 2);
        _9_fs0.configure(&out_9_cat, &out_9_fs0);  
/*e4*/
        _A_conv0.configure(&out_9_fs0, &Q_table_A_conv0, &WT_buffer_A_conv0, &bias_A_conv0, &out_A_conv0, PadStrideInfo(1,1,0,0), weights_A_conv0, precision[52], 1);
        _A_act0.configure(&out_A_conv0, &out_A_act0, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        _A_conv1.configure(&out_9_fs0, &Q_table_A_conv1, &WT_buffer_A_conv1, &bias_A_conv1, &out_A_conv1, PadStrideInfo(1,1,0,0), weights_A_conv1, precision[53], 1);
        _A_act1.configure(&out_A_conv1, &out_A_act1, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        _A_conv2.configure(&out_A_act1, &Q_table_A_conv2, &WT_buffer_A_conv2, &bias_A_conv2, &out_A_conv2, PadStrideInfo(1,1,1,1), weights_A_conv2, precision[54], 1);
        _A_act2.configure(&out_A_conv2, &out_A_act2, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        _A_conv3.configure(&out_9_fs0, &Q_table_A_conv3, &WT_buffer_A_conv3, &bias_A_conv3, &out_A_conv3, PadStrideInfo(1,1,0,0), weights_A_conv3, precision[55], 1);
        _A_act3.configure(&out_A_conv3, &out_A_act3, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        _A_conv4.configure(&out_A_act3, &Q_table_A_conv4, &WT_buffer_A_conv4, &bias_A_conv4, &out_A_conv4, PadStrideInfo(1,1,1,1), weights_A_conv4, precision[56], 1);
        _A_act4.configure(&out_A_conv4, &out_A_act4, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        _A_conv5.configure(&out_A_act4, &Q_table_A_conv5, &WT_buffer_A_conv5, &bias_A_conv5, &out_A_conv5, PadStrideInfo(1,1,1,1), weights_A_conv5, precision[57], 1);
        _A_act5.configure(&out_A_conv5, &out_A_act5, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        _A_conv6.configure(&out_9_fs0, &Q_table_A_conv6, &WT_buffer_A_conv6, &bias_A_conv6, &out_A_conv6, PadStrideInfo(1,1,0,0), weights_A_conv6, precision[58], 1);
        _A_act6.configure(&out_A_conv6, &out_A_act6, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        _A_sf0.configure(&out_A_act0, &out_A_sf0); std::vector<ITensor*> vA; vA.push_back(&out_A_sf0);
        _A_sf1.configure(&out_A_act2, &out_A_sf1);vA.push_back(&out_A_sf1);
        _A_sf2.configure(&out_A_act5, &out_A_sf2);vA.push_back(&out_A_sf2);
        _A_sf3.configure(&out_A_act6, &out_A_sf3);vA.push_back(&out_A_sf3);
        _A_cat.configure(vA, &out_A_cat, 2);
        _A_fs0.configure(&out_A_cat, &out_A_fs0);         
/*maxpool*/
        _B_sf0.configure(&out_A_fs0, &out_B_sf0);
        _B_pool0.configure(&out_B_sf0, &out_B_pool0,PoolingLayerInfo(PoolingType::MAX, 2, PadStrideInfo(2, 2, 0, 0)));
        _B_fs0.configure(&out_B_pool0, &out_B_fs0);
/*a5*/
        _C_conv0.configure(&out_B_fs0, &Q_table_C_conv0, &WT_buffer_C_conv0, &bias_C_conv0, &out_C_conv0, PadStrideInfo(1,1,0,0), weights_C_conv0, precision[59], 1);
        _C_act0.configure(&out_C_conv0, &out_C_act0, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        _C_conv1.configure(&out_B_fs0, &Q_table_C_conv1, &WT_buffer_C_conv1, &bias_C_conv1, &out_C_conv1, PadStrideInfo(1,1,0,0), weights_C_conv1, precision[60], 1);
        _C_act1.configure(&out_C_conv1, &out_C_act1, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        _C_conv2.configure(&out_C_act1, &Q_table_C_conv2, &WT_buffer_C_conv2, &bias_C_conv2, &out_C_conv2, PadStrideInfo(1,1,1,1), weights_C_conv2, precision[61], 1);
        _C_act2.configure(&out_C_conv2, &out_C_act2, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        _C_conv3.configure(&out_B_fs0, &Q_table_C_conv3, &WT_buffer_C_conv3, &bias_C_conv3, &out_C_conv3, PadStrideInfo(1,1,0,0), weights_C_conv3, precision[62], 1);
        _C_act3.configure(&out_C_conv3, &out_C_act3, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        _C_conv4.configure(&out_C_act3, &Q_table_C_conv4, &WT_buffer_C_conv4, &bias_C_conv4, &out_C_conv4, PadStrideInfo(1,1,1,1), weights_C_conv4, precision[63], 1);
        _C_act4.configure(&out_C_conv4, &out_C_act4, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        _C_conv5.configure(&out_C_act4, &Q_table_C_conv5, &WT_buffer_C_conv5, &bias_C_conv5, &out_C_conv5, PadStrideInfo(1,1,1,1), weights_C_conv5, precision[64], 1);
        _C_act5.configure(&out_C_conv5, &out_C_act5, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        _C_conv6.configure(&out_B_fs0, &Q_table_C_conv6, &WT_buffer_C_conv6, &bias_C_conv6, &out_C_conv6, PadStrideInfo(1,1,0,0), weights_C_conv6, precision[65], 1);
        _C_act6.configure(&out_C_conv6, &out_C_act6, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        _C_sf0.configure(&out_C_act0, &out_C_sf0); std::vector<ITensor*> vC; vC.push_back(&out_C_sf0);
        _C_sf1.configure(&out_C_act2, &out_C_sf1);vC.push_back(&out_C_sf1);
        _C_sf2.configure(&out_C_act5, &out_C_sf2);vC.push_back(&out_C_sf2);
        _C_sf3.configure(&out_C_act6, &out_C_sf3);vC.push_back(&out_C_sf3);
        _C_cat.configure(vC, &out_C_cat, 2);
        _C_fs0.configure(&out_C_cat, &out_C_fs0);         
/*b5*/
        _D_conv0.configure(&out_C_fs0, &Q_table_D_conv0, &WT_buffer_D_conv0, &bias_D_conv0, &out_D_conv0, PadStrideInfo(1,1,0,0), weights_D_conv0, precision[66], 1);
        _D_act0.configure(&out_D_conv0, &out_D_act0, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        _D_conv1.configure(&out_C_fs0, &Q_table_D_conv1, &WT_buffer_D_conv1, &bias_D_conv1, &out_D_conv1, PadStrideInfo(1,1,0,0), weights_D_conv1, precision[67], 1);
        _D_act1.configure(&out_D_conv1, &out_D_act1, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        _D_conv2.configure(&out_D_act1, &Q_table_D_conv2, &WT_buffer_D_conv2, &bias_D_conv2, &out_D_conv2, PadStrideInfo(1,1,1,1), weights_D_conv2, precision[68], 1);
        _D_act2.configure(&out_D_conv2, &out_D_act2, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        _D_conv3.configure(&out_C_fs0, &Q_table_D_conv3, &WT_buffer_D_conv3, &bias_D_conv3, &out_D_conv3, PadStrideInfo(1,1,0,0), weights_D_conv3, precision[69], 1);
        _D_act3.configure(&out_D_conv3, &out_D_act3, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        _D_conv4.configure(&out_D_act3, &Q_table_D_conv4, &WT_buffer_D_conv4, &bias_D_conv4, &out_D_conv4, PadStrideInfo(1,1,1,1), weights_D_conv4, precision[70], 1);
        _D_act4.configure(&out_D_conv4, &out_D_act4, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        _D_conv5.configure(&out_D_act4, &Q_table_D_conv5, &WT_buffer_D_conv5, &bias_D_conv5, &out_D_conv5, PadStrideInfo(1,1,1,1), weights_D_conv5, precision[71], 1);
        _D_act5.configure(&out_D_conv5, &out_D_act5, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        _D_conv6.configure(&out_C_fs0, &Q_table_D_conv6, &WT_buffer_D_conv6, &bias_D_conv6, &out_D_conv6, PadStrideInfo(1,1,0,0), weights_D_conv6, precision[72], 1);
        _D_act6.configure(&out_D_conv6, &out_D_act6, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        _D_sf0.configure(&out_D_act0, &out_D_sf0); std::vector<ITensor*> vD; vD.push_back(&out_D_sf0);
        _D_sf1.configure(&out_D_act2, &out_D_sf1);vD.push_back(&out_D_sf1);
        _D_sf2.configure(&out_D_act5, &out_D_sf2);vD.push_back(&out_D_sf2);
        _D_sf3.configure(&out_D_act6, &out_D_sf3);vD.push_back(&out_D_sf3);
        _D_cat.configure(vD, &out_D_cat, 2);
        _D_fs0.configure(&out_D_cat, &out_D_fs0);        
/*avgpool*/
        _E_sf0.configure(&out_D_fs0, &out_E_sf0);
        _E_pool0.configure(&out_E_sf0, &out_E_pool0, PoolingLayerInfo(PoolingType::AVG, 7, PadStrideInfo(1, 1, 0, 0)));
        _E_fs0.configure(&out_E_pool0, &out_E_fs0);
/*fc*/
        _F_conv0.configure(&out_E_fs0, &Q_table_F_conv0, &WT_buffer_F_conv0, &bias_F_conv0, &out_F_conv0, PadStrideInfo(1,1,0,0), weights_F_conv0, precision[73], 1);

/**************************************allocate start********************************/
        src.allocator()->allocate(); 
        /*****************pre-layers=1***************/
        Q_table_1_conv0.allocator()->allocate(); WT_buffer_1_conv0.allocator()->allocate(); bias_1_conv0.allocator()->allocate(); 
        Q_table_1_conv1.allocator()->allocate(); WT_buffer_1_conv1.allocator()->allocate(); bias_1_conv1.allocator()->allocate();
        Q_table_1_conv2.allocator()->allocate(); WT_buffer_1_conv2.allocator()->allocate(); bias_1_conv2.allocator()->allocate();
        out_1_conv0.allocator()->allocate(); out_1_conv1.allocator()->allocate(); out_1_conv2.allocator()->allocate();
        out_1_pool0.allocator()->allocate(); out_1_pool1.allocator()->allocate();
        out_1_act0.allocator()->allocate();
        out_1_sf0.allocator()->allocate(); out_1_sf1.allocator()->allocate();
        out_1_fs0.allocator()->allocate(); out_1_fs1.allocator()->allocate();

        /***********************a3=2******************/
        /********a3-b1************/
        Q_table_2_conv0.allocator()->allocate(); WT_buffer_2_conv0.allocator()->allocate(); bias_2_conv0.allocator()->allocate();
        out_2_conv0.allocator()->allocate();
        out_2_act0.allocator()->allocate();
        /*******a3-b2************/
        Q_table_2_conv1.allocator()->allocate(); WT_buffer_2_conv1.allocator()->allocate(); bias_2_conv1.allocator()->allocate();
        Q_table_2_conv2.allocator()->allocate(); WT_buffer_2_conv2.allocator()->allocate(); bias_2_conv2.allocator()->allocate();
        out_2_conv1.allocator()->allocate(); out_2_conv2.allocator()->allocate();
        out_2_act1.allocator()->allocate();     out_2_act2.allocator()->allocate();
        /*******a3-b3*************/
        Q_table_2_conv3.allocator()->allocate(); WT_buffer_2_conv3.allocator()->allocate(); bias_2_conv3.allocator()->allocate();
        Q_table_2_conv4.allocator()->allocate(); WT_buffer_2_conv4.allocator()->allocate(); bias_2_conv4.allocator()->allocate();
        Q_table_2_conv5.allocator()->allocate(); WT_buffer_2_conv5.allocator()->allocate(); bias_2_conv5.allocator()->allocate();
        out_2_conv3.allocator()->allocate(); out_2_conv4.allocator()->allocate(); out_2_conv5.allocator()->allocate();
        out_2_act3.allocator()->allocate();     out_2_act4.allocator()->allocate();     out_2_act5.allocator()->allocate();
        /*******a3-b4*************/
        Q_table_2_conv6.allocator()->allocate(); WT_buffer_2_conv6.allocator()->allocate(); bias_2_conv6.allocator()->allocate();
        out_2_conv6.allocator()->allocate(); 
        out_2_act6.allocator()->allocate();
        out_2_cat.allocator()->allocate();
        out_2_sf0.allocator()->allocate(), out_2_sf1.allocator()->allocate(), out_2_sf2.allocator()->allocate(), out_2_sf3.allocator()->allocate();out_2_fs0.allocator()->allocate();

        /***********************b3=3******************/
        /********b3-b1************/
        Q_table_3_conv0.allocator()->allocate(); WT_buffer_3_conv0.allocator()->allocate(); bias_3_conv0.allocator()->allocate();
        out_3_conv0.allocator()->allocate();
        out_3_act0.allocator()->allocate();
        /*******b3-b2************/
        Q_table_3_conv1.allocator()->allocate(); WT_buffer_3_conv1.allocator()->allocate(); bias_3_conv1.allocator()->allocate();
        Q_table_3_conv2.allocator()->allocate(); WT_buffer_3_conv2.allocator()->allocate(); bias_3_conv2.allocator()->allocate();
        out_3_conv1.allocator()->allocate(); out_3_conv2.allocator()->allocate();
        out_3_act1.allocator()->allocate();     out_3_act2.allocator()->allocate();
        /*******b3-b3*************/
        Q_table_3_conv3.allocator()->allocate(); WT_buffer_3_conv3.allocator()->allocate(); bias_3_conv3.allocator()->allocate();
        Q_table_3_conv4.allocator()->allocate(); WT_buffer_3_conv4.allocator()->allocate(); bias_3_conv4.allocator()->allocate();
        Q_table_3_conv5.allocator()->allocate(); WT_buffer_3_conv5.allocator()->allocate(); bias_3_conv5.allocator()->allocate();
        out_3_conv3.allocator()->allocate(); out_3_conv4.allocator()->allocate(); out_3_conv5.allocator()->allocate();
        out_3_act3.allocator()->allocate();     out_3_act4.allocator()->allocate();     out_3_act5.allocator()->allocate();
        /*******b3-b4*************/
        Q_table_3_conv6.allocator()->allocate(); WT_buffer_3_conv6.allocator()->allocate(); bias_3_conv6.allocator()->allocate();
        out_3_conv6.allocator()->allocate(); 
        out_3_act6.allocator()->allocate();
        out_3_cat.allocator()->allocate();
        out_3_sf0.allocator()->allocate(), out_3_sf1.allocator()->allocate(), out_3_sf2.allocator()->allocate(), out_3_sf3.allocator()->allocate();out_3_fs0.allocator()->allocate();

        /***********************c3=4******************/
        /********c3-b1************/
        Q_table_4_conv0.allocator()->allocate(); WT_buffer_4_conv0.allocator()->allocate(); bias_4_conv0.allocator()->allocate();
        out_4_conv0.allocator()->allocate();
        out_4_act0.allocator()->allocate();
        /*******c3-b2************/
        Q_table_4_conv1.allocator()->allocate(); WT_buffer_4_conv1.allocator()->allocate(); bias_4_conv1.allocator()->allocate();
        Q_table_4_conv2.allocator()->allocate(); WT_buffer_4_conv2.allocator()->allocate(); bias_4_conv2.allocator()->allocate();
        out_4_conv1.allocator()->allocate(); out_4_conv2.allocator()->allocate();
        out_4_act1.allocator()->allocate();     out_4_act2.allocator()->allocate();
        /*******c3-b3*************/
        Q_table_4_conv3.allocator()->allocate(); WT_buffer_4_conv3.allocator()->allocate(); bias_4_conv3.allocator()->allocate();
        Q_table_4_conv4.allocator()->allocate(); WT_buffer_4_conv4.allocator()->allocate(); bias_4_conv4.allocator()->allocate();
        Q_table_4_conv5.allocator()->allocate(); WT_buffer_4_conv5.allocator()->allocate(); bias_4_conv5.allocator()->allocate();
        out_4_conv3.allocator()->allocate(); out_4_conv4.allocator()->allocate(); out_4_conv5.allocator()->allocate();
        out_4_act3.allocator()->allocate();     out_4_act4.allocator()->allocate();     out_4_act5.allocator()->allocate();
        /*******c3-b4*************/
        Q_table_4_conv6.allocator()->allocate(); WT_buffer_4_conv6.allocator()->allocate(); bias_4_conv6.allocator()->allocate();
        out_4_conv6.allocator()->allocate(); 
        out_4_act6.allocator()->allocate();
        out_4_cat.allocator()->allocate();
        out_4_sf0.allocator()->allocate(), out_4_sf1.allocator()->allocate(), out_4_sf2.allocator()->allocate(), out_4_sf3.allocator()->allocate();out_4_fs0.allocator()->allocate();

        /***********************maxpool******************/
        out_5_pool0.allocator()->allocate();
        out_5_sf0.allocator()->allocate(); out_5_fs0.allocator()->allocate();

        /***********************a4=6******************/
        /********a4-b1************/
        Q_table_6_conv0.allocator()->allocate(); WT_buffer_6_conv0.allocator()->allocate(); bias_6_conv0.allocator()->allocate();
        out_6_conv0.allocator()->allocate();
        out_6_act0.allocator()->allocate();
        /*******a4-b2************/
        Q_table_6_conv1.allocator()->allocate(); WT_buffer_6_conv1.allocator()->allocate(); bias_6_conv1.allocator()->allocate();
        Q_table_6_conv2.allocator()->allocate(); WT_buffer_6_conv2.allocator()->allocate(); bias_6_conv2.allocator()->allocate();
        out_6_conv1.allocator()->allocate(); out_6_conv2.allocator()->allocate();
        out_6_act1.allocator()->allocate();     out_6_act2.allocator()->allocate();
        /*******a4-b3*************/
        Q_table_6_conv3.allocator()->allocate(); WT_buffer_6_conv3.allocator()->allocate(); bias_6_conv3.allocator()->allocate();
        Q_table_6_conv4.allocator()->allocate(); WT_buffer_6_conv4.allocator()->allocate(); bias_6_conv4.allocator()->allocate();
        Q_table_6_conv5.allocator()->allocate(); WT_buffer_6_conv5.allocator()->allocate(); bias_6_conv5.allocator()->allocate();
        out_6_conv3.allocator()->allocate(); out_6_conv4.allocator()->allocate(); out_6_conv5.allocator()->allocate();
        out_6_act3.allocator()->allocate();     out_6_act4.allocator()->allocate();     out_6_act5.allocator()->allocate();
        /*******a4-b4*************/
        Q_table_6_conv6.allocator()->allocate(); WT_buffer_6_conv6.allocator()->allocate(); bias_6_conv6.allocator()->allocate();
        out_6_conv6.allocator()->allocate(); 
        out_6_act6.allocator()->allocate();
        out_6_cat.allocator()->allocate();
        out_6_sf0.allocator()->allocate(), out_6_sf1.allocator()->allocate(), out_6_sf2.allocator()->allocate(), out_6_sf3.allocator()->allocate();out_6_fs0.allocator()->allocate();

        /***********************b4=7******************/
        /********b4-b1************/
        Q_table_7_conv0.allocator()->allocate(); WT_buffer_7_conv0.allocator()->allocate(); bias_7_conv0.allocator()->allocate();
        out_7_conv0.allocator()->allocate();
        out_7_act0.allocator()->allocate();
        /*******b4-b2************/
        Q_table_7_conv1.allocator()->allocate(); WT_buffer_7_conv1.allocator()->allocate(); bias_7_conv1.allocator()->allocate();
        Q_table_7_conv2.allocator()->allocate(); WT_buffer_7_conv2.allocator()->allocate(); bias_7_conv2.allocator()->allocate();
        out_7_conv1.allocator()->allocate(); out_7_conv2.allocator()->allocate();
        out_7_act1.allocator()->allocate();     out_7_act2.allocator()->allocate();
        /*******b4-b3*************/
        Q_table_7_conv3.allocator()->allocate(); WT_buffer_7_conv3.allocator()->allocate(); bias_7_conv3.allocator()->allocate();
        Q_table_7_conv4.allocator()->allocate(); WT_buffer_7_conv4.allocator()->allocate(); bias_7_conv4.allocator()->allocate();
        Q_table_7_conv5.allocator()->allocate(); WT_buffer_7_conv5.allocator()->allocate(); bias_7_conv5.allocator()->allocate();
        out_7_conv3.allocator()->allocate(); out_7_conv4.allocator()->allocate(); out_7_conv5.allocator()->allocate();
        out_7_act3.allocator()->allocate();     out_7_act4.allocator()->allocate();     out_7_act5.allocator()->allocate();
        /*******b4-b4*************/
        Q_table_7_conv6.allocator()->allocate(); WT_buffer_7_conv6.allocator()->allocate(); bias_7_conv6.allocator()->allocate();
        out_7_conv6.allocator()->allocate(); 
        out_7_act6.allocator()->allocate();
        out_7_cat.allocator()->allocate();
        out_7_sf0.allocator()->allocate(), out_7_sf1.allocator()->allocate(), out_7_sf2.allocator()->allocate(), out_7_sf3.allocator()->allocate();out_7_fs0.allocator()->allocate();

        /***********************c4=8******************/
        /********c4-b1************/
        Q_table_8_conv0.allocator()->allocate(); WT_buffer_8_conv0.allocator()->allocate(); bias_8_conv0.allocator()->allocate();
        out_8_conv0.allocator()->allocate();
        out_8_act0.allocator()->allocate();
        /*******c4-b2************/
        Q_table_8_conv1.allocator()->allocate(); WT_buffer_8_conv1.allocator()->allocate(); bias_8_conv1.allocator()->allocate();
        Q_table_8_conv2.allocator()->allocate(); WT_buffer_8_conv2.allocator()->allocate(); bias_8_conv2.allocator()->allocate();
        out_8_conv1.allocator()->allocate(); out_8_conv2.allocator()->allocate();
        out_8_act1.allocator()->allocate();     out_8_act2.allocator()->allocate();
        /*******c4-b3*************/
        Q_table_8_conv3.allocator()->allocate(); WT_buffer_8_conv3.allocator()->allocate(); bias_8_conv3.allocator()->allocate();
        Q_table_8_conv4.allocator()->allocate(); WT_buffer_8_conv4.allocator()->allocate(); bias_8_conv4.allocator()->allocate();
        Q_table_8_conv5.allocator()->allocate(); WT_buffer_8_conv5.allocator()->allocate(); bias_8_conv5.allocator()->allocate();
        out_8_conv3.allocator()->allocate(); out_8_conv4.allocator()->allocate(); out_8_conv5.allocator()->allocate();
        out_8_act3.allocator()->allocate();     out_8_act4.allocator()->allocate();     out_8_act5.allocator()->allocate();
        /*******c4-b4*************/
        Q_table_8_conv6.allocator()->allocate(); WT_buffer_8_conv6.allocator()->allocate(); bias_8_conv6.allocator()->allocate();
        out_8_conv6.allocator()->allocate(); 
        out_8_act6.allocator()->allocate();
        out_8_cat.allocator()->allocate();
        out_8_sf0.allocator()->allocate(), out_8_sf1.allocator()->allocate(), out_8_sf2.allocator()->allocate(), out_8_sf3.allocator()->allocate();out_8_fs0.allocator()->allocate();

        /***********************d4=9******************/
        /********d4-b1************/
        Q_table_9_conv0.allocator()->allocate(); WT_buffer_9_conv0.allocator()->allocate(); bias_9_conv0.allocator()->allocate();
        out_9_conv0.allocator()->allocate();
        out_9_act0.allocator()->allocate();
        /*******d4-b2************/
        Q_table_9_conv1.allocator()->allocate(); WT_buffer_9_conv1.allocator()->allocate(); bias_9_conv1.allocator()->allocate();
        Q_table_9_conv2.allocator()->allocate(); WT_buffer_9_conv2.allocator()->allocate(); bias_9_conv2.allocator()->allocate();
        out_9_conv1.allocator()->allocate(); out_9_conv2.allocator()->allocate();
        out_9_act1.allocator()->allocate();     out_9_act2.allocator()->allocate();
        /*******d4-b3*************/
        Q_table_9_conv3.allocator()->allocate(); WT_buffer_9_conv3.allocator()->allocate(); bias_9_conv3.allocator()->allocate();
        Q_table_9_conv4.allocator()->allocate(); WT_buffer_9_conv4.allocator()->allocate(); bias_9_conv4.allocator()->allocate();
        Q_table_9_conv5.allocator()->allocate(); WT_buffer_9_conv5.allocator()->allocate(); bias_9_conv5.allocator()->allocate();
        out_9_conv3.allocator()->allocate(); out_9_conv4.allocator()->allocate(); out_9_conv5.allocator()->allocate();
        out_9_act3.allocator()->allocate();     out_9_act4.allocator()->allocate();     out_9_act5.allocator()->allocate();
        /*******d4-b4*************/
        Q_table_9_conv6.allocator()->allocate(); WT_buffer_9_conv6.allocator()->allocate(); bias_9_conv6.allocator()->allocate();
        out_9_conv6.allocator()->allocate(); 
        out_9_act6.allocator()->allocate();
        out_9_cat.allocator()->allocate();
        out_9_sf0.allocator()->allocate(), out_9_sf1.allocator()->allocate(), out_9_sf2.allocator()->allocate(), out_9_sf3.allocator()->allocate();out_9_fs0.allocator()->allocate();

        /***********************e4=A******************/
        /********e4-b1************/
        Q_table_A_conv0.allocator()->allocate(); WT_buffer_A_conv0.allocator()->allocate(); bias_A_conv0.allocator()->allocate();
        out_A_conv0.allocator()->allocate();
        out_A_act0.allocator()->allocate();
        /*******e4-b2************/
        Q_table_A_conv1.allocator()->allocate(); WT_buffer_A_conv1.allocator()->allocate(); bias_A_conv1.allocator()->allocate();
        Q_table_A_conv2.allocator()->allocate(); WT_buffer_A_conv2.allocator()->allocate(); bias_A_conv2.allocator()->allocate();
        out_A_conv1.allocator()->allocate(); out_A_conv2.allocator()->allocate();
        out_A_act1.allocator()->allocate();     out_A_act2.allocator()->allocate();
        /*******e4-b3*************/
        Q_table_A_conv3.allocator()->allocate(); WT_buffer_A_conv3.allocator()->allocate(); bias_A_conv3.allocator()->allocate();
        Q_table_A_conv4.allocator()->allocate(); WT_buffer_A_conv4.allocator()->allocate(); bias_A_conv4.allocator()->allocate();
        Q_table_A_conv5.allocator()->allocate(); WT_buffer_A_conv5.allocator()->allocate(); bias_A_conv5.allocator()->allocate();
        out_A_conv3.allocator()->allocate(); out_A_conv4.allocator()->allocate(); out_A_conv5.allocator()->allocate();
        out_A_act3.allocator()->allocate();     out_A_act4.allocator()->allocate();     out_A_act5.allocator()->allocate();
        /*******e4-b4*************/
        Q_table_A_conv6.allocator()->allocate(); WT_buffer_A_conv6.allocator()->allocate(); bias_A_conv6.allocator()->allocate();
        out_A_conv6.allocator()->allocate(); 
        out_A_act6.allocator()->allocate();
        out_A_cat.allocator()->allocate();
        out_A_sf0.allocator()->allocate(), out_A_sf1.allocator()->allocate(), out_A_sf2.allocator()->allocate(), out_A_sf3.allocator()->allocate();out_A_fs0.allocator()->allocate();

        /**********************maxpool*******************/
        out_B_pool0.allocator()->allocate();
        out_B_sf0.allocator()->allocate(); out_B_fs0.allocator()->allocate();
        /***********************a5=C******************/
        /********a5-b1************/
        Q_table_C_conv0.allocator()->allocate(); WT_buffer_C_conv0.allocator()->allocate(); bias_C_conv0.allocator()->allocate();
        out_C_conv0.allocator()->allocate();
        out_C_act0.allocator()->allocate();
        /*******a5-b2************/
        Q_table_C_conv1.allocator()->allocate(); WT_buffer_C_conv1.allocator()->allocate(); bias_C_conv1.allocator()->allocate();
        Q_table_C_conv2.allocator()->allocate(); WT_buffer_C_conv2.allocator()->allocate(); bias_C_conv2.allocator()->allocate();
        out_C_conv1.allocator()->allocate(); out_C_conv2.allocator()->allocate();
        out_C_act1.allocator()->allocate();     out_C_act2.allocator()->allocate();
        /*******a5-b3*************/
        Q_table_C_conv3.allocator()->allocate(); WT_buffer_C_conv3.allocator()->allocate(); bias_C_conv3.allocator()->allocate();
        Q_table_C_conv4.allocator()->allocate(); WT_buffer_C_conv4.allocator()->allocate(); bias_C_conv4.allocator()->allocate();
        Q_table_C_conv5.allocator()->allocate(); WT_buffer_C_conv5.allocator()->allocate(); bias_C_conv5.allocator()->allocate();
        out_C_conv3.allocator()->allocate(); out_C_conv4.allocator()->allocate(); out_C_conv5.allocator()->allocate();
        out_C_act3.allocator()->allocate();     out_C_act4.allocator()->allocate();     out_C_act5.allocator()->allocate();
        /*******a5-b4*************/
        Q_table_C_conv6.allocator()->allocate(); WT_buffer_C_conv6.allocator()->allocate(); bias_C_conv6.allocator()->allocate();
        out_C_conv6.allocator()->allocate(); 
        out_C_act6.allocator()->allocate();
        out_C_cat.allocator()->allocate();
        out_C_sf0.allocator()->allocate(), out_C_sf1.allocator()->allocate(), out_C_sf2.allocator()->allocate(), out_C_sf3.allocator()->allocate();out_C_fs0.allocator()->allocate();

        /***********************b5=D******************/
        /********b5-b1************/
        Q_table_D_conv0.allocator()->allocate(); WT_buffer_D_conv0.allocator()->allocate(); bias_D_conv0.allocator()->allocate();
        out_D_conv0.allocator()->allocate();
        out_D_act0.allocator()->allocate();
        /*******b5-b2************/
        Q_table_D_conv1.allocator()->allocate(); WT_buffer_D_conv1.allocator()->allocate(); bias_D_conv1.allocator()->allocate();
        Q_table_D_conv2.allocator()->allocate(); WT_buffer_D_conv2.allocator()->allocate(); bias_D_conv2.allocator()->allocate();
        out_D_conv1.allocator()->allocate(); out_D_conv2.allocator()->allocate();
        out_D_act1.allocator()->allocate();     out_D_act2.allocator()->allocate();
        /*******b5-b3*************/
        Q_table_D_conv3.allocator()->allocate(); WT_buffer_D_conv3.allocator()->allocate(); bias_D_conv3.allocator()->allocate();
        Q_table_D_conv4.allocator()->allocate(); WT_buffer_D_conv4.allocator()->allocate(); bias_D_conv4.allocator()->allocate();
        Q_table_D_conv5.allocator()->allocate(); WT_buffer_D_conv5.allocator()->allocate(); bias_D_conv5.allocator()->allocate();
        out_D_conv3.allocator()->allocate(); out_D_conv4.allocator()->allocate(); out_D_conv5.allocator()->allocate();
        out_D_act3.allocator()->allocate();     out_D_act4.allocator()->allocate();     out_D_act5.allocator()->allocate();
        /*******b5-b4*************/
        Q_table_D_conv6.allocator()->allocate(); WT_buffer_D_conv6.allocator()->allocate(); bias_D_conv6.allocator()->allocate();
        out_D_conv6.allocator()->allocate(); 
        out_D_act6.allocator()->allocate();
        out_D_cat.allocator()->allocate();
        out_D_sf0.allocator()->allocate(), out_D_sf1.allocator()->allocate(), out_D_sf2.allocator()->allocate(), out_D_sf3.allocator()->allocate();out_D_fs0.allocator()->allocate();

        /***********************avgpool******************/
        out_E_pool0.allocator()->allocate();
        out_E_sf0.allocator()->allocate(); out_E_fs0.allocator()->allocate();

        /**********************fc**************************/
        Q_table_F_conv0.allocator()->allocate(); WT_buffer_F_conv0.allocator()->allocate(); bias_F_conv0.allocator()->allocate();
        out_F_conv0.allocator()->allocate();

/********************************numpy fill start**************************/
        npy_input.fill_tensor2(src);
        /*pre_layers*/
        npy_1_conv0_q.fill_tensor2(Q_table_1_conv0);
        npy_1_conv0_wt.fill_tensor2(WT_buffer_1_conv0);
        npy_1_conv0_b.fill_tensor2(bias_1_conv0);
        npy_1_conv1_q.fill_tensor2(Q_table_1_conv1);
        npy_1_conv1_wt.fill_tensor2(WT_buffer_1_conv1);
        npy_1_conv1_b.fill_tensor2(bias_1_conv1);
        npy_1_conv2_q.fill_tensor2(Q_table_1_conv2);
        npy_1_conv2_wt.fill_tensor2(WT_buffer_1_conv2);
        npy_1_conv2_b.fill_tensor2(bias_1_conv2);
        /*a3*/
        npy_2_conv0_q.fill_tensor2(Q_table_2_conv0);
        npy_2_conv0_wt.fill_tensor2(WT_buffer_2_conv0);
        npy_2_conv0_b.fill_tensor2(bias_2_conv0);
        npy_2_conv1_q.fill_tensor2(Q_table_2_conv1);
        npy_2_conv1_wt.fill_tensor2(WT_buffer_2_conv1);
        npy_2_conv1_b.fill_tensor2(bias_2_conv1);
        npy_2_conv2_q.fill_tensor2(Q_table_2_conv2);
        npy_2_conv2_wt.fill_tensor2(WT_buffer_2_conv2);
        npy_2_conv2_b.fill_tensor2(bias_2_conv2);
        npy_2_conv3_q.fill_tensor2(Q_table_2_conv3);
        npy_2_conv3_wt.fill_tensor2(WT_buffer_2_conv3);
        npy_2_conv3_b.fill_tensor2(bias_2_conv3);
        npy_2_conv4_q.fill_tensor2(Q_table_2_conv4);
        npy_2_conv4_wt.fill_tensor2(WT_buffer_2_conv4);
        npy_2_conv4_b.fill_tensor2(bias_2_conv4);
        npy_2_conv5_q.fill_tensor2(Q_table_2_conv5);
        npy_2_conv5_wt.fill_tensor2(WT_buffer_2_conv5);
        npy_2_conv5_b.fill_tensor2(bias_2_conv5);
        npy_2_conv6_q.fill_tensor2(Q_table_2_conv6);
        npy_2_conv6_wt.fill_tensor2(WT_buffer_2_conv6);
        npy_2_conv6_b.fill_tensor2(bias_2_conv6);
        /*b3*/
        npy_3_conv0_q.fill_tensor2(Q_table_3_conv0);
        npy_3_conv0_wt.fill_tensor2(WT_buffer_3_conv0);
        npy_3_conv0_b.fill_tensor2(bias_3_conv0);
        npy_3_conv1_q.fill_tensor2(Q_table_3_conv1);
        npy_3_conv1_wt.fill_tensor2(WT_buffer_3_conv1);
        npy_3_conv1_b.fill_tensor2(bias_3_conv1);
        npy_3_conv2_q.fill_tensor2(Q_table_3_conv2);
        npy_3_conv2_wt.fill_tensor2(WT_buffer_3_conv2);
        npy_3_conv2_b.fill_tensor2(bias_3_conv2);
        npy_3_conv3_q.fill_tensor2(Q_table_3_conv3);
        npy_3_conv3_wt.fill_tensor2(WT_buffer_3_conv3);
        npy_3_conv3_b.fill_tensor2(bias_3_conv3);
        npy_3_conv4_q.fill_tensor2(Q_table_3_conv4);
        npy_3_conv4_wt.fill_tensor2(WT_buffer_3_conv4);
        npy_3_conv4_b.fill_tensor2(bias_3_conv4);
        npy_3_conv5_q.fill_tensor2(Q_table_3_conv5);
        npy_3_conv5_wt.fill_tensor2(WT_buffer_3_conv5);
        npy_3_conv5_b.fill_tensor2(bias_3_conv5);
        npy_3_conv6_q.fill_tensor2(Q_table_3_conv6);
        npy_3_conv6_wt.fill_tensor2(WT_buffer_3_conv6);
        npy_3_conv6_b.fill_tensor2(bias_3_conv6);
        /*c3*/
        npy_4_conv0_q.fill_tensor2(Q_table_4_conv0);
        npy_4_conv0_wt.fill_tensor2(WT_buffer_4_conv0);
        npy_4_conv0_b.fill_tensor2(bias_4_conv0);
        npy_4_conv1_q.fill_tensor2(Q_table_4_conv1);
        npy_4_conv1_wt.fill_tensor2(WT_buffer_4_conv1);
        npy_4_conv1_b.fill_tensor2(bias_4_conv1);
        npy_4_conv2_q.fill_tensor2(Q_table_4_conv2);
        npy_4_conv2_wt.fill_tensor2(WT_buffer_4_conv2);
        npy_4_conv2_b.fill_tensor2(bias_4_conv2);
        npy_4_conv3_q.fill_tensor2(Q_table_4_conv3);
        npy_4_conv3_wt.fill_tensor2(WT_buffer_4_conv3);
        npy_4_conv3_b.fill_tensor2(bias_4_conv3);
        npy_4_conv4_q.fill_tensor2(Q_table_4_conv4);
        npy_4_conv4_wt.fill_tensor2(WT_buffer_4_conv4);
        npy_4_conv4_b.fill_tensor2(bias_4_conv4);
        npy_4_conv5_q.fill_tensor2(Q_table_4_conv5);
        npy_4_conv5_wt.fill_tensor2(WT_buffer_4_conv5);
        npy_4_conv5_b.fill_tensor2(bias_4_conv5);
        npy_4_conv6_q.fill_tensor2(Q_table_4_conv6);
        npy_4_conv6_wt.fill_tensor2(WT_buffer_4_conv6);
        npy_4_conv6_b.fill_tensor2(bias_4_conv6);
        /*a4*/
        npy_6_conv0_q.fill_tensor2(Q_table_6_conv0);
        npy_6_conv0_wt.fill_tensor2(WT_buffer_6_conv0);
        npy_6_conv0_b.fill_tensor2(bias_6_conv0);
        npy_6_conv1_q.fill_tensor2(Q_table_6_conv1);
        npy_6_conv1_wt.fill_tensor2(WT_buffer_6_conv1);
        npy_6_conv1_b.fill_tensor2(bias_6_conv1);
        npy_6_conv2_q.fill_tensor2(Q_table_6_conv2);
        npy_6_conv2_wt.fill_tensor2(WT_buffer_6_conv2);
        npy_6_conv2_b.fill_tensor2(bias_6_conv2);
        npy_6_conv3_q.fill_tensor2(Q_table_6_conv3);
        npy_6_conv3_wt.fill_tensor2(WT_buffer_6_conv3);
        npy_6_conv3_b.fill_tensor2(bias_6_conv3);
        npy_6_conv4_q.fill_tensor2(Q_table_6_conv4);
        npy_6_conv4_wt.fill_tensor2(WT_buffer_6_conv4);
        npy_6_conv4_b.fill_tensor2(bias_6_conv4);
        npy_6_conv5_q.fill_tensor2(Q_table_6_conv5);
        npy_6_conv5_wt.fill_tensor2(WT_buffer_6_conv5);
        npy_6_conv5_b.fill_tensor2(bias_6_conv5);
        npy_6_conv6_q.fill_tensor2(Q_table_6_conv6);
        npy_6_conv6_wt.fill_tensor2(WT_buffer_6_conv6);
        npy_6_conv6_b.fill_tensor2(bias_6_conv6);
        /*b4*/
        npy_7_conv0_q.fill_tensor2(Q_table_7_conv0);
        npy_7_conv0_wt.fill_tensor2(WT_buffer_7_conv0);
        npy_7_conv0_b.fill_tensor2(bias_7_conv0);
        npy_7_conv1_q.fill_tensor2(Q_table_7_conv1);
        npy_7_conv1_wt.fill_tensor2(WT_buffer_7_conv1);
        npy_7_conv1_b.fill_tensor2(bias_7_conv1);
        npy_7_conv2_q.fill_tensor2(Q_table_7_conv2);
        npy_7_conv2_wt.fill_tensor2(WT_buffer_7_conv2);
        npy_7_conv2_b.fill_tensor2(bias_7_conv2);
        npy_7_conv3_q.fill_tensor2(Q_table_7_conv3);
        npy_7_conv3_wt.fill_tensor2(WT_buffer_7_conv3);
        npy_7_conv3_b.fill_tensor2(bias_7_conv3);
        npy_7_conv4_q.fill_tensor2(Q_table_7_conv4);
        npy_7_conv4_wt.fill_tensor2(WT_buffer_7_conv4);
        npy_7_conv4_b.fill_tensor2(bias_7_conv4);
        npy_7_conv5_q.fill_tensor2(Q_table_7_conv5);
        npy_7_conv5_wt.fill_tensor2(WT_buffer_7_conv5);
        npy_7_conv5_b.fill_tensor2(bias_7_conv5);
        npy_7_conv6_q.fill_tensor2(Q_table_7_conv6);
        npy_7_conv6_wt.fill_tensor2(WT_buffer_7_conv6);
        npy_7_conv6_b.fill_tensor2(bias_7_conv6);
        /*c4*/
        npy_8_conv0_q.fill_tensor2(Q_table_8_conv0);
        npy_8_conv0_wt.fill_tensor2(WT_buffer_8_conv0);
        npy_8_conv0_b.fill_tensor2(bias_8_conv0);
        npy_8_conv1_q.fill_tensor2(Q_table_8_conv1);
        npy_8_conv1_wt.fill_tensor2(WT_buffer_8_conv1);
        npy_8_conv1_b.fill_tensor2(bias_8_conv1);
        npy_8_conv2_q.fill_tensor2(Q_table_8_conv2);
        npy_8_conv2_wt.fill_tensor2(WT_buffer_8_conv2);
        npy_8_conv2_b.fill_tensor2(bias_8_conv2);
        npy_8_conv3_q.fill_tensor2(Q_table_8_conv3);
        npy_8_conv3_wt.fill_tensor2(WT_buffer_8_conv3);
        npy_8_conv3_b.fill_tensor2(bias_8_conv3);
        npy_8_conv4_q.fill_tensor2(Q_table_8_conv4);
        npy_8_conv4_wt.fill_tensor2(WT_buffer_8_conv4);
        npy_8_conv4_b.fill_tensor2(bias_8_conv4);
        npy_8_conv5_q.fill_tensor2(Q_table_8_conv5);
        npy_8_conv5_wt.fill_tensor2(WT_buffer_8_conv5);
        npy_8_conv5_b.fill_tensor2(bias_8_conv5);
        npy_8_conv6_q.fill_tensor2(Q_table_8_conv6);
        npy_8_conv6_wt.fill_tensor2(WT_buffer_8_conv6);
        npy_8_conv6_b.fill_tensor2(bias_8_conv6);
        /*d4*/
        npy_9_conv0_q.fill_tensor2(Q_table_9_conv0);
        npy_9_conv0_wt.fill_tensor2(WT_buffer_9_conv0);
        npy_9_conv0_b.fill_tensor2(bias_9_conv0);
        npy_9_conv1_q.fill_tensor2(Q_table_9_conv1);
        npy_9_conv1_wt.fill_tensor2(WT_buffer_9_conv1);
        npy_9_conv1_b.fill_tensor2(bias_9_conv1);
        npy_9_conv2_q.fill_tensor2(Q_table_9_conv2);
        npy_9_conv2_wt.fill_tensor2(WT_buffer_9_conv2);
        npy_9_conv2_b.fill_tensor2(bias_9_conv2);
        npy_9_conv3_q.fill_tensor2(Q_table_9_conv3);
        npy_9_conv3_wt.fill_tensor2(WT_buffer_9_conv3);
        npy_9_conv3_b.fill_tensor2(bias_9_conv3);
        npy_9_conv4_q.fill_tensor2(Q_table_9_conv4);
        npy_9_conv4_wt.fill_tensor2(WT_buffer_9_conv4);
        npy_9_conv4_b.fill_tensor2(bias_9_conv4);
        npy_9_conv5_q.fill_tensor2(Q_table_9_conv5);
        npy_9_conv5_wt.fill_tensor2(WT_buffer_9_conv5);
        npy_9_conv5_b.fill_tensor2(bias_9_conv5);
        npy_9_conv6_q.fill_tensor2(Q_table_9_conv6);
        npy_9_conv6_wt.fill_tensor2(WT_buffer_9_conv6);
        npy_9_conv6_b.fill_tensor2(bias_9_conv6);
        /*e4*/
        npy_A_conv0_q.fill_tensor2(Q_table_A_conv0);
        npy_A_conv0_wt.fill_tensor2(WT_buffer_A_conv0);
        npy_A_conv0_b.fill_tensor2(bias_A_conv0);
        npy_A_conv1_q.fill_tensor2(Q_table_A_conv1);
        npy_A_conv1_wt.fill_tensor2(WT_buffer_A_conv1);
        npy_A_conv1_b.fill_tensor2(bias_A_conv1);
        npy_A_conv2_q.fill_tensor2(Q_table_A_conv2);
        npy_A_conv2_wt.fill_tensor2(WT_buffer_A_conv2);
        npy_A_conv2_b.fill_tensor2(bias_A_conv2);
        npy_A_conv3_q.fill_tensor2(Q_table_A_conv3);
        npy_A_conv3_wt.fill_tensor2(WT_buffer_A_conv3);
        npy_A_conv3_b.fill_tensor2(bias_A_conv3);
        npy_A_conv4_q.fill_tensor2(Q_table_A_conv4);
        npy_A_conv4_wt.fill_tensor2(WT_buffer_A_conv4);
        npy_A_conv4_b.fill_tensor2(bias_A_conv4);
        npy_A_conv5_q.fill_tensor2(Q_table_A_conv5);
        npy_A_conv5_wt.fill_tensor2(WT_buffer_A_conv5);
        npy_A_conv5_b.fill_tensor2(bias_A_conv5);
        npy_A_conv6_q.fill_tensor2(Q_table_A_conv6);
        npy_A_conv6_wt.fill_tensor2(WT_buffer_A_conv6);
        npy_A_conv6_b.fill_tensor2(bias_A_conv6);
        /*a5*/
        npy_C_conv0_q.fill_tensor2(Q_table_C_conv0);
        npy_C_conv0_wt.fill_tensor2(WT_buffer_C_conv0);
        npy_C_conv0_b.fill_tensor2(bias_C_conv0);
        npy_C_conv1_q.fill_tensor2(Q_table_C_conv1);
        npy_C_conv1_wt.fill_tensor2(WT_buffer_C_conv1);
        npy_C_conv1_b.fill_tensor2(bias_C_conv1);
        npy_C_conv2_q.fill_tensor2(Q_table_C_conv2);
        npy_C_conv2_wt.fill_tensor2(WT_buffer_C_conv2);
        npy_C_conv2_b.fill_tensor2(bias_C_conv2);
        npy_C_conv3_q.fill_tensor2(Q_table_C_conv3);
        npy_C_conv3_wt.fill_tensor2(WT_buffer_C_conv3);
        npy_C_conv3_b.fill_tensor2(bias_C_conv3);
        npy_C_conv4_q.fill_tensor2(Q_table_C_conv4);
        npy_C_conv4_wt.fill_tensor2(WT_buffer_C_conv4);
        npy_C_conv4_b.fill_tensor2(bias_C_conv4);
        npy_C_conv5_q.fill_tensor2(Q_table_C_conv5);
        npy_C_conv5_wt.fill_tensor2(WT_buffer_C_conv5);
        npy_C_conv5_b.fill_tensor2(bias_C_conv5);
        npy_C_conv6_q.fill_tensor2(Q_table_C_conv6);
        npy_C_conv6_wt.fill_tensor2(WT_buffer_C_conv6);
        npy_C_conv6_b.fill_tensor2(bias_C_conv6);
        /*b5*/
        npy_D_conv0_q.fill_tensor2(Q_table_D_conv0);
        npy_D_conv0_wt.fill_tensor2(WT_buffer_D_conv0);
        npy_D_conv0_b.fill_tensor2(bias_D_conv0);
        npy_D_conv1_q.fill_tensor2(Q_table_D_conv1);
        npy_D_conv1_wt.fill_tensor2(WT_buffer_D_conv1);
        npy_D_conv1_b.fill_tensor2(bias_D_conv1);
        npy_D_conv2_q.fill_tensor2(Q_table_D_conv2);
        npy_D_conv2_wt.fill_tensor2(WT_buffer_D_conv2);
        npy_D_conv2_b.fill_tensor2(bias_D_conv2);
        npy_D_conv3_q.fill_tensor2(Q_table_D_conv3);
        npy_D_conv3_wt.fill_tensor2(WT_buffer_D_conv3);
        npy_D_conv3_b.fill_tensor2(bias_D_conv3);
        npy_D_conv4_q.fill_tensor2(Q_table_D_conv4);
        npy_D_conv4_wt.fill_tensor2(WT_buffer_D_conv4);
        npy_D_conv4_b.fill_tensor2(bias_D_conv4);
        npy_D_conv5_q.fill_tensor2(Q_table_D_conv5);
        npy_D_conv5_wt.fill_tensor2(WT_buffer_D_conv5);
        npy_D_conv5_b.fill_tensor2(bias_D_conv5);
        npy_D_conv6_q.fill_tensor2(Q_table_D_conv6);
        npy_D_conv6_wt.fill_tensor2(WT_buffer_D_conv6);
        npy_D_conv6_b.fill_tensor2(bias_D_conv6);

        npy_F_conv0_q.fill_tensor2(Q_table_F_conv0);
        npy_F_conv0_wt.fill_tensor2(WT_buffer_F_conv0);
        npy_F_conv0_b.fill_tensor2(bias_F_conv0);

        is_fortran      = npy_input.is_fortran();

        return true;
    }
void do_run() override
    {

        /**************************************layers' time***********************************/
        double time_1_conv0=0, time_1_conv1=0, time_1_conv2=0, time_1_pool0=0, time_1_pool1=0, time_1_act0=0;
        double time_2_conv0=0, time_2_conv1=0, time_2_conv2=0, time_2_conv3=0, time_2_conv4=0, time_2_conv5=0, time_2_conv6=0;
        double time_2_act0=0, time_2_act1=0, time_2_act2=0, time_2_act3=0, time_2_act4=0, time_2_act5=0, time_2_act6=0, time_2_cat=0;
        double time_3_conv0=0, time_3_conv1=0, time_3_conv2=0, time_3_conv3=0, time_3_conv4=0, time_3_conv5=0, time_3_conv6=0;
        double time_3_act0=0, time_3_act1=0, time_3_act2=0, time_3_act3=0, time_3_act4=0, time_3_act5=0, time_3_act6=0, time_3_cat=0;
        double time_4_conv0=0, time_4_conv1=0, time_4_conv2=0, time_4_conv3=0, time_4_conv4=0, time_4_conv5=0, time_4_conv6=0;
        double time_4_act0=0, time_4_act1=0, time_4_act2=0, time_4_act3=0, time_4_act4=0, time_4_act5=0, time_4_act6=0, time_4_cat=0;
        double time_5_pool0=0;
        double time_6_conv0=0, time_6_conv1=0, time_6_conv2=0, time_6_conv3=0, time_6_conv4=0, time_6_conv5=0, time_6_conv6=0;
        double time_6_act0=0, time_6_act1=0, time_6_act2=0, time_6_act3=0, time_6_act4=0, time_6_act5=0, time_6_act6=0, time_6_cat=0;
        double time_7_conv0=0, time_7_conv1=0, time_7_conv2=0, time_7_conv3=0, time_7_conv4=0, time_7_conv5=0, time_7_conv6=0;
        double time_7_act0=0, time_7_act1=0, time_7_act2=0, time_7_act3=0, time_7_act4=0, time_7_act5=0, time_7_act6=0, time_7_cat=0;
        double time_8_conv0=0, time_8_conv1=0, time_8_conv2=0, time_8_conv3=0, time_8_conv4=0, time_8_conv5=0, time_8_conv6=0;
        double time_8_act0=0, time_8_act1=0, time_8_act2=0, time_8_act3=0, time_8_act4=0, time_8_act5=0, time_8_act6=0, time_8_cat=0;
        double time_9_conv0=0, time_9_conv1=0, time_9_conv2=0, time_9_conv3=0, time_9_conv4=0, time_9_conv5=0, time_9_conv6=0;
        double time_9_act0=0, time_9_act1=0, time_9_act2=0, time_9_act3=0, time_9_act4=0, time_9_act5=0, time_9_act6=0, time_9_cat=0;
        double time_A_conv0=0, time_A_conv1=0, time_A_conv2=0, time_A_conv3=0, time_A_conv4=0, time_A_conv5=0, time_A_conv6=0;
        double time_A_act0=0, time_A_act1=0, time_A_act2=0, time_A_act3=0, time_A_act4=0, time_A_act5=0, time_A_act6=0, time_A_cat=0;
        double time_B_pool0=0;
        double time_C_conv0=0, time_C_conv1=0, time_C_conv2=0, time_C_conv3=0, time_C_conv4=0, time_C_conv5=0, time_C_conv6=0;
        double time_C_act0=0, time_C_act1=0, time_C_act2=0, time_C_act3=0, time_C_act4=0, time_C_act5=0, time_C_act6=0, time_C_cat=0;
        double time_D_conv0=0, time_D_conv1=0, time_D_conv2=0, time_D_conv3=0, time_D_conv4=0, time_D_conv5=0, time_D_conv6=0;
        double time_D_act0=0, time_D_act1=0, time_D_act2=0, time_D_act3=0, time_D_act4=0, time_D_act5=0, time_D_act6=0, time_D_cat=0;
        double time_E_pool0=0;
        double time_F_conv0=0;
        /*****************************************************Layer Run***********************************/
        double total_time=0, time=0;

        std::string base_path = "/media/sdcard/ComputeLibrary";
        std::string output_file_path = "/model.csv";
        ofstream out(base_path+output_file_path, ios::out | ios::app);
        out<<"GoogLeNet ABM"<<std::endl;
        int cycles = 101;
        for(int i=0; i<cycles; i++)
        {
            auto start = std::chrono::high_resolution_clock::now();
            /*****pre_layers********/
            /*std::cout<<"pre_layers run~"<<std::endl;*/
            _1_conv0.run();auto end10 = std::chrono::high_resolution_clock::now();
            _1_sf0.run();auto end11 = std::chrono::high_resolution_clock::now();
            _1_pool0.run();auto end12 = std::chrono::high_resolution_clock::now();
            _1_fs0.run();auto end13 = std::chrono::high_resolution_clock::now();
            _1_conv1.run();auto end14 = std::chrono::high_resolution_clock::now();
            _1_conv2.run();auto end15 = std::chrono::high_resolution_clock::now();
            _1_act0.run();auto end16 = std::chrono::high_resolution_clock::now();
            _1_sf1.run();auto end17 = std::chrono::high_resolution_clock::now();
            _1_pool1.run();auto end18 = std::chrono::high_resolution_clock::now();
            _1_fs1.run();auto end19 = std::chrono::high_resolution_clock::now();
            /**********a3*****/
            /*std::cout<<"a3 run~"<<std::endl;*/
            _2_conv0.run();auto end20 = std::chrono::high_resolution_clock::now();
            _2_act0.run();auto end21 = std::chrono::high_resolution_clock::now();
            _2_conv1.run();auto end22 = std::chrono::high_resolution_clock::now();
            _2_act1.run();auto end23 = std::chrono::high_resolution_clock::now();
            _2_conv2.run();auto end24 = std::chrono::high_resolution_clock::now();
            _2_act2.run();auto end25 = std::chrono::high_resolution_clock::now();
            _2_conv3.run();auto end26 = std::chrono::high_resolution_clock::now();
            _2_act3.run();auto end27 = std::chrono::high_resolution_clock::now();
            _2_conv4.run();auto end28 = std::chrono::high_resolution_clock::now();
            _2_act4.run();auto end29 = std::chrono::high_resolution_clock::now();
            _2_conv5.run();auto end2A = std::chrono::high_resolution_clock::now();
            _2_act5.run();auto end2B = std::chrono::high_resolution_clock::now();
            _2_conv6.run();auto end2C = std::chrono::high_resolution_clock::now();
            _2_act6.run();auto end2D = std::chrono::high_resolution_clock::now();
            _2_sf0.run();_2_sf1.run(); _2_sf2.run(); _2_sf3.run(); 
            end2D = std::chrono::high_resolution_clock::now();
            _2_cat.run();auto end2E = std::chrono::high_resolution_clock::now();
            _2_fs0.run();
            end2E = std::chrono::high_resolution_clock::now();
            /*********b3******/
            /*std::cout<<"b3 run~"<<std::endl;*/
            _3_conv0.run();auto end30 = std::chrono::high_resolution_clock::now();
            _3_act0.run();auto end31 = std::chrono::high_resolution_clock::now();
            _3_conv1.run();auto end32 = std::chrono::high_resolution_clock::now();
            _3_act1.run();auto end33 = std::chrono::high_resolution_clock::now();
            _3_conv2.run();auto end34 = std::chrono::high_resolution_clock::now();
            _3_act2.run();auto end35 = std::chrono::high_resolution_clock::now();
            _3_conv3.run();auto end36 = std::chrono::high_resolution_clock::now();
            _3_act3.run();auto end37 = std::chrono::high_resolution_clock::now();
            _3_conv4.run();auto end38 = std::chrono::high_resolution_clock::now();
            _3_act4.run();auto end39 = std::chrono::high_resolution_clock::now();
            _3_conv5.run();auto end3A = std::chrono::high_resolution_clock::now();
            _3_act5.run();auto end3B = std::chrono::high_resolution_clock::now();
            _3_conv6.run();auto end3C = std::chrono::high_resolution_clock::now();
            _3_act6.run();auto end3D = std::chrono::high_resolution_clock::now();
            _3_sf0.run();_3_sf1.run(); _3_sf2.run(); _3_sf3.run(); 
            end3D = std::chrono::high_resolution_clock::now();
            _3_cat.run();auto end3E = std::chrono::high_resolution_clock::now();
            _3_fs0.run();
            end3E = std::chrono::high_resolution_clock::now();
            /**********c3******/
            /*std::cout<<"c3 run~"<<std::endl;*/
            _4_conv0.run();auto end40 = std::chrono::high_resolution_clock::now();
            _4_act0.run();auto end41 = std::chrono::high_resolution_clock::now();
            _4_conv1.run();auto end42 = std::chrono::high_resolution_clock::now();
            _4_act1.run();auto end43 = std::chrono::high_resolution_clock::now();
            _4_conv2.run();auto end44 = std::chrono::high_resolution_clock::now();
            _4_act2.run();auto end45 = std::chrono::high_resolution_clock::now();
            _4_conv3.run();auto end46 = std::chrono::high_resolution_clock::now();
            _4_act3.run();auto end47 = std::chrono::high_resolution_clock::now();
            _4_conv4.run();auto end48 = std::chrono::high_resolution_clock::now();
            _4_act4.run();auto end49 = std::chrono::high_resolution_clock::now();
            _4_conv5.run();auto end4A = std::chrono::high_resolution_clock::now();
            _4_act5.run();auto end4B = std::chrono::high_resolution_clock::now();
            _4_conv6.run();auto end4C = std::chrono::high_resolution_clock::now();
            _4_act6.run();auto end4D = std::chrono::high_resolution_clock::now();
            _4_sf0.run();_4_sf1.run(); _4_sf2.run(); _4_sf3.run(); 
            end4D = std::chrono::high_resolution_clock::now();
            _4_cat.run();auto end4E = std::chrono::high_resolution_clock::now();
            _4_fs0.run();
            end4E = std::chrono::high_resolution_clock::now();
            /*********maxpool*********/
            /*std::cout<<"maxpool run~"<<std::endl;*/
            _5_sf0.run();auto end50 = std::chrono::high_resolution_clock::now();
            _5_pool0.run();auto end51 = std::chrono::high_resolution_clock::now();
            _5_fs0.run();auto end52 = std::chrono::high_resolution_clock::now();
            /*****a4***********/
            /*std::cout<<"a4 run~"<<std::endl;*/
            _6_conv0.run();auto end60 = std::chrono::high_resolution_clock::now();
            _6_act0.run();auto end61 = std::chrono::high_resolution_clock::now();
            _6_conv1.run();auto end62 = std::chrono::high_resolution_clock::now();
            _6_act1.run();auto end63 = std::chrono::high_resolution_clock::now();
            _6_conv2.run();auto end64 = std::chrono::high_resolution_clock::now();
            _6_act2.run();auto end65 = std::chrono::high_resolution_clock::now();
            _6_conv3.run();auto end66 = std::chrono::high_resolution_clock::now();
            _6_act3.run();auto end67 = std::chrono::high_resolution_clock::now();
            _6_conv4.run();auto end68 = std::chrono::high_resolution_clock::now();
            _6_act4.run();auto end69 = std::chrono::high_resolution_clock::now();
            _6_conv5.run();auto end6A = std::chrono::high_resolution_clock::now();
            _6_act5.run();auto end6B = std::chrono::high_resolution_clock::now();
            _6_conv6.run();auto end6C = std::chrono::high_resolution_clock::now();
            _6_act6.run();auto end6D = std::chrono::high_resolution_clock::now();
            _6_sf0.run();_6_sf1.run(); _6_sf2.run(); _6_sf3.run(); 
            end6D = std::chrono::high_resolution_clock::now();
            _6_cat.run();auto end6E = std::chrono::high_resolution_clock::now();
            _6_fs0.run();
            end6E = std::chrono::high_resolution_clock::now();
            /********b4*********/
            /*std::cout<<"b4 run~"<<std::endl;*/
            _7_conv0.run();auto end70 = std::chrono::high_resolution_clock::now();
            _7_act0.run();auto end71 = std::chrono::high_resolution_clock::now();
            _7_conv1.run();auto end72 = std::chrono::high_resolution_clock::now();
            _7_act1.run();auto end73 = std::chrono::high_resolution_clock::now();
            _7_conv2.run();auto end74 = std::chrono::high_resolution_clock::now();
            _7_act2.run();auto end75 = std::chrono::high_resolution_clock::now();
            _7_conv3.run();auto end76 = std::chrono::high_resolution_clock::now();
            _7_act3.run();auto end77 = std::chrono::high_resolution_clock::now();
            _7_conv4.run();auto end78 = std::chrono::high_resolution_clock::now();
            _7_act4.run();auto end79 = std::chrono::high_resolution_clock::now();
            _7_conv5.run();auto end7A = std::chrono::high_resolution_clock::now();
            _7_act5.run();auto end7B = std::chrono::high_resolution_clock::now();
            _7_conv6.run();auto end7C = std::chrono::high_resolution_clock::now();
            _7_act6.run();auto end7D = std::chrono::high_resolution_clock::now();
            _7_sf0.run();_7_sf1.run(); _7_sf2.run(); _7_sf3.run(); 
            end7D = std::chrono::high_resolution_clock::now();
            _7_cat.run();auto end7E = std::chrono::high_resolution_clock::now();
            _7_fs0.run();
            end7E = std::chrono::high_resolution_clock::now();
            /**********c4********/
            /*std::cout<<"c4 run~"<<std::endl;*/
            _8_conv0.run();auto end80 = std::chrono::high_resolution_clock::now();
            _8_act0.run();auto end81 = std::chrono::high_resolution_clock::now();
            _8_conv1.run();auto end82 = std::chrono::high_resolution_clock::now();
            _8_act1.run();auto end83 = std::chrono::high_resolution_clock::now();
            _8_conv2.run();auto end84 = std::chrono::high_resolution_clock::now();
            _8_act2.run();auto end85 = std::chrono::high_resolution_clock::now();
            _8_conv3.run();auto end86 = std::chrono::high_resolution_clock::now();
            _8_act3.run();auto end87 = std::chrono::high_resolution_clock::now();
            _8_conv4.run();auto end88 = std::chrono::high_resolution_clock::now();
            _8_act4.run();auto end89 = std::chrono::high_resolution_clock::now();
            _8_conv5.run();auto end8A = std::chrono::high_resolution_clock::now();
            _8_act5.run();auto end8B = std::chrono::high_resolution_clock::now();
            _8_conv6.run();auto end8C = std::chrono::high_resolution_clock::now();
            _8_act6.run();auto end8D = std::chrono::high_resolution_clock::now();
            _8_sf0.run();_8_sf1.run(); _8_sf2.run(); _8_sf3.run(); 
            end8D = std::chrono::high_resolution_clock::now();
            _8_cat.run();auto end8E = std::chrono::high_resolution_clock::now();
            _8_fs0.run();
            end8E = std::chrono::high_resolution_clock::now();
            /********d4********/
            /*std::cout<<"d4 run~"<<std::endl;*/
            _9_conv0.run();auto end90 = std::chrono::high_resolution_clock::now();
            _9_act0.run();auto end91 = std::chrono::high_resolution_clock::now();
            _9_conv1.run();auto end92 = std::chrono::high_resolution_clock::now();
            _9_act1.run();auto end93 = std::chrono::high_resolution_clock::now();
            _9_conv2.run();auto end94 = std::chrono::high_resolution_clock::now();
            _9_act2.run();auto end95 = std::chrono::high_resolution_clock::now();
            _9_conv3.run();auto end96 = std::chrono::high_resolution_clock::now();
            _9_act3.run();auto end97 = std::chrono::high_resolution_clock::now();
            _9_conv4.run();auto end98 = std::chrono::high_resolution_clock::now();
            _9_act4.run();auto end99 = std::chrono::high_resolution_clock::now();
            _9_conv5.run();auto end9A = std::chrono::high_resolution_clock::now();
            _9_act5.run();auto end9B = std::chrono::high_resolution_clock::now();
            _9_conv6.run();auto end9C = std::chrono::high_resolution_clock::now();
            _9_act6.run();auto end9D = std::chrono::high_resolution_clock::now();
            _9_sf0.run();_9_sf1.run(); _9_sf2.run(); _9_sf3.run(); 
            end9D = std::chrono::high_resolution_clock::now();
            _9_cat.run();auto end9E = std::chrono::high_resolution_clock::now();
            _9_fs0.run();
            end9E = std::chrono::high_resolution_clock::now();
            /*************e4*****/
            /*std::cout<<"e4 run~"<<std::endl;*/
            _A_conv0.run();auto endA0 = std::chrono::high_resolution_clock::now();
            _A_act0.run();auto endA1 = std::chrono::high_resolution_clock::now();
            _A_conv1.run();auto endA2 = std::chrono::high_resolution_clock::now();
            _A_act1.run();auto endA3 = std::chrono::high_resolution_clock::now();
            _A_conv2.run();auto endA4 = std::chrono::high_resolution_clock::now();
            _A_act2.run();auto endA5 = std::chrono::high_resolution_clock::now();
            _A_conv3.run();auto endA6 = std::chrono::high_resolution_clock::now();
            _A_act3.run();auto endA7 = std::chrono::high_resolution_clock::now();
            _A_conv4.run();auto endA8 = std::chrono::high_resolution_clock::now();
            _A_act4.run();auto endA9 = std::chrono::high_resolution_clock::now();
            _A_conv5.run();auto endAA = std::chrono::high_resolution_clock::now();
            _A_act5.run();auto endAB = std::chrono::high_resolution_clock::now();
            _A_conv6.run();auto endAC = std::chrono::high_resolution_clock::now();
            _A_act6.run();auto endAD = std::chrono::high_resolution_clock::now();
            _A_sf0.run();_A_sf1.run(); _A_sf2.run(); _A_sf3.run(); 
            endAD = std::chrono::high_resolution_clock::now();
            _A_cat.run();auto endAE = std::chrono::high_resolution_clock::now();
            _A_fs0.run();
            endAE = std::chrono::high_resolution_clock::now();
            /**********maxpool**/
            /*std::cout<<"maxpool run~"<<std::endl;*/
            _B_sf0.run();auto endB0 = std::chrono::high_resolution_clock::now();
            _B_pool0.run();auto endB1 = std::chrono::high_resolution_clock::now();
            _B_fs0.run();auto endB2 = std::chrono::high_resolution_clock::now();
            /************a5*********/
            /*std::cout<<"a5 run~"<<std::endl;*/
            _C_conv0.run();auto endC0 = std::chrono::high_resolution_clock::now();
            _C_act0.run();auto endC1 = std::chrono::high_resolution_clock::now();
            _C_conv1.run();auto endC2 = std::chrono::high_resolution_clock::now();
            _C_act1.run();auto endC3 = std::chrono::high_resolution_clock::now();
            _C_conv2.run();auto endC4 = std::chrono::high_resolution_clock::now();
            _C_act2.run();auto endC5 = std::chrono::high_resolution_clock::now();
            _C_conv3.run();auto endC6 = std::chrono::high_resolution_clock::now();
            _C_act3.run();auto endC7 = std::chrono::high_resolution_clock::now();
            _C_conv4.run();auto endC8 = std::chrono::high_resolution_clock::now();
            _C_act4.run();auto endC9 = std::chrono::high_resolution_clock::now();
            _C_conv5.run();auto endCA = std::chrono::high_resolution_clock::now();
            _C_act5.run();auto endCB = std::chrono::high_resolution_clock::now();
            _C_conv6.run();auto endCC = std::chrono::high_resolution_clock::now();
            _C_act6.run();auto endCD = std::chrono::high_resolution_clock::now();
            _C_sf0.run();_C_sf1.run(); _C_sf2.run(); _C_sf3.run(); 
            endCD = std::chrono::high_resolution_clock::now();
            _C_cat.run();auto endCE = std::chrono::high_resolution_clock::now();
            _C_fs0.run();
            endCE = std::chrono::high_resolution_clock::now();
            /*****************b5******/
            /*std::cout<<"b5 run~"<<std::endl;*/
            _D_conv0.run();auto endD0 = std::chrono::high_resolution_clock::now();
            _D_act0.run();auto endD1 = std::chrono::high_resolution_clock::now();
            _D_conv1.run();auto endD2 = std::chrono::high_resolution_clock::now();
            _D_act1.run();auto endD3 = std::chrono::high_resolution_clock::now();
            _D_conv2.run();auto endD4 = std::chrono::high_resolution_clock::now();
            _D_act2.run();auto endD5 = std::chrono::high_resolution_clock::now();
            _D_conv3.run();auto endD6 = std::chrono::high_resolution_clock::now();
            _D_act3.run();auto endD7 = std::chrono::high_resolution_clock::now();
            _D_conv4.run();auto endD8 = std::chrono::high_resolution_clock::now();
            _D_act4.run();auto endD9 = std::chrono::high_resolution_clock::now();
            _D_conv5.run();auto endDA = std::chrono::high_resolution_clock::now();
            _D_act5.run();auto endDB = std::chrono::high_resolution_clock::now();
            _D_conv6.run();auto endDC = std::chrono::high_resolution_clock::now();
            _D_act6.run();auto endDD = std::chrono::high_resolution_clock::now();
            _D_sf0.run();_D_sf1.run(); _D_sf2.run(); _D_sf3.run(); 
            endDD = std::chrono::high_resolution_clock::now();
            _D_cat.run();auto endDE = std::chrono::high_resolution_clock::now();
            _D_fs0.run();
            endDE = std::chrono::high_resolution_clock::now();
            /**********avgpool*******/
            /*std::cout<<"avgpool run~"<<std::endl;*/
            _E_sf0.run();auto endE0 = std::chrono::high_resolution_clock::now();
            _E_pool0.run();auto endE1 = std::chrono::high_resolution_clock::now();
            _E_fs0.run();auto endE2 = std::chrono::high_resolution_clock::now();
            /********fc***********/
            /*std::cout<<"fc run~"<<std::endl;*/
            _F_conv0.run();auto endF0 = std::chrono::high_resolution_clock::now();

            if(i>0)
            {
                double one_runtime=0;
                /***************pre_layers**************/
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end10 - start).count();time_1_conv0+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end12 - end11).count();time_1_pool0+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end14 - end13).count();time_1_conv1+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end15 - end14).count();time_1_conv2+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end16 - end15).count();time_1_act0+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end18 - end17).count();time_1_pool1+=time; one_runtime+=time;
                /************a3*******************/
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end20 - end19).count();time_2_conv0+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end21 - end20).count();time_2_act0+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end22 - end21).count();time_2_conv1+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end23 - end22).count();time_2_act1+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end24 - end23).count();time_2_conv2+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end25 - end24).count();time_2_act2+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end26 - end25).count();time_2_conv3+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end27 - end26).count();time_2_act3+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end28 - end27).count();time_2_conv4+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end29 - end28).count();time_2_act4+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end2A - end29).count();time_2_conv5+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end2B - end2A).count();time_2_act5+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end2C - end2B).count();time_2_conv6+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end2D - end2C).count();time_2_act6+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end2E - end2D).count();time_2_cat+=time; one_runtime+=time;
                /****************b3*********************/
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end30 - end2E).count();time_3_conv0+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end31 - end30).count();time_3_act0+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end32 - end31).count();time_3_conv1+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end33 - end32).count();time_3_act1+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end34 - end33).count();time_3_conv2+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end35 - end34).count();time_3_act2+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end36 - end35).count();time_3_conv3+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end37 - end36).count();time_3_act3+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end38 - end37).count();time_3_conv4+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end39 - end38).count();time_3_act4+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end3A - end39).count();time_3_conv5+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end3B - end3A).count();time_3_act5+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end3C - end3B).count();time_3_conv6+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end3D - end3C).count();time_3_act6+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end3E - end3D).count();time_3_cat+=time; one_runtime+=time;
                /*******************c3*********************/
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end40 - end3E).count();time_4_conv0+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end41 - end40).count();time_4_act0+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end42 - end41).count();time_4_conv1+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end43 - end42).count();time_4_act1+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end44 - end43).count();time_4_conv2+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end45 - end44).count();time_4_act2+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end46 - end45).count();time_4_conv3+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end47 - end46).count();time_4_act3+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end48 - end47).count();time_4_conv4+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end49 - end48).count();time_4_act4+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end4A - end49).count();time_4_conv5+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end4B - end4A).count();time_4_act5+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end4C - end4B).count();time_4_conv6+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end4D - end4C).count();time_4_act6+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end4E - end4D).count();time_4_cat+=time; one_runtime+=time;
                /**************************maxpool*****************/
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end51 - end50).count();time_5_pool0+=time; one_runtime+=time;
                /**************************a4*********************/
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end60 - end52).count();time_6_conv0+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end61 - end60).count();time_6_act0+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end62 - end61).count();time_6_conv1+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end63 - end62).count();time_6_act1+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end64 - end63).count();time_6_conv2+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end65 - end64).count();time_6_act2+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end66 - end65).count();time_6_conv3+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end67 - end66).count();time_6_act3+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end68 - end67).count();time_6_conv4+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end69 - end68).count();time_6_act4+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end6A - end69).count();time_6_conv5+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end6B - end6A).count();time_6_act5+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end6C - end6B).count();time_6_conv6+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end6D - end6C).count();time_6_act6+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end6E - end6D).count();time_6_cat+=time; one_runtime+=time;
                /***********************b4**********************/
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end70 - end6E).count();time_7_conv0+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end71 - end70).count();time_7_act0+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end72 - end71).count();time_7_conv1+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end73 - end72).count();time_7_act1+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end74 - end73).count();time_7_conv2+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end75 - end74).count();time_7_act2+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end76 - end75).count();time_7_conv3+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end77 - end76).count();time_7_act3+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end78 - end77).count();time_7_conv4+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end79 - end78).count();time_7_act4+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end7A - end79).count();time_7_conv5+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end7B - end7A).count();time_7_act5+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end7C - end7B).count();time_7_conv6+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end7D - end7C).count();time_7_act6+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end7E - end7D).count();time_7_cat+=time; one_runtime+=time;
                /*******************c4******************************/
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end80 - end7E).count();time_8_conv0+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end81 - end80).count();time_8_act0+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end82 - end81).count();time_8_conv1+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end83 - end82).count();time_8_act1+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end84 - end83).count();time_8_conv2+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end85 - end84).count();time_8_act2+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end86 - end85).count();time_8_conv3+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end87 - end86).count();time_8_act3+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end88 - end87).count();time_8_conv4+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end89 - end88).count();time_8_act4+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end8A - end89).count();time_8_conv5+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end8B - end8A).count();time_8_act5+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end8C - end8B).count();time_8_conv6+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end8D - end8C).count();time_8_act6+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end8E - end8D).count();time_8_cat+=time; one_runtime+=time;
                /*****************d4**************************************/
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end90 - end8E).count();time_9_conv0+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end91 - end90).count();time_9_act0+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end92 - end91).count();time_9_conv1+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end93 - end92).count();time_9_act1+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end94 - end93).count();time_9_conv2+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end95 - end94).count();time_9_act2+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end96 - end95).count();time_9_conv3+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end97 - end96).count();time_9_act3+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end98 - end97).count();time_9_conv4+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end99 - end98).count();time_9_act4+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end9A - end99).count();time_9_conv5+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end9B - end9A).count();time_9_act5+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end9C - end9B).count();time_9_conv6+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end9D - end9C).count();time_9_act6+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(end9E - end9D).count();time_9_cat+=time; one_runtime+=time;
                /*********************e4******************************/
                time = std::chrono::duration_cast<std::chrono::duration<double>>(endA0 - end9E).count();time_A_conv0+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(endA1 - endA0).count();time_A_act0+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(endA2 - endA1).count();time_A_conv1+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(endA3 - endA2).count();time_A_act1+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(endA4 - endA3).count();time_A_conv2+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(endA5 - endA4).count();time_A_act2+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(endA6 - endA5).count();time_A_conv3+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(endA7 - endA6).count();time_A_act3+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(endA8 - endA7).count();time_A_conv4+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(endA9 - endA8).count();time_A_act4+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(endAA - endA9).count();time_A_conv5+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(endAB - endAA).count();time_A_act5+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(endAC - endAB).count();time_A_conv6+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(endAD - endAC).count();time_A_act6+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(endAE - endAD).count();time_A_cat+=time; one_runtime+=time;
                /**********************maxpool******************************/
                time = std::chrono::duration_cast<std::chrono::duration<double>>(endB1 - endB0).count();time_B_pool0+=time; one_runtime+=time;
                /*****************************a5*********************************/
                time = std::chrono::duration_cast<std::chrono::duration<double>>(endC0 - endB2).count();time_C_conv0+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(endC1 - endC0).count();time_C_act0+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(endC2 - endC1).count();time_C_conv1+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(endC3 - endC2).count();time_C_act1+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(endC4 - endC3).count();time_C_conv2+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(endC5 - endC4).count();time_C_act2+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(endC6 - endC5).count();time_C_conv3+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(endC7 - endC6).count();time_C_act3+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(endC8 - endC7).count();time_C_conv4+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(endC9 - endC8).count();time_C_act4+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(endCA - endC9).count();time_C_conv5+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(endCB - endCA).count();time_C_act5+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(endCC - endCB).count();time_C_conv6+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(endCD - endCC).count();time_C_act6+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(endCE - endCD).count();time_C_cat+=time; one_runtime+=time;
                /******************************B5*******************************/
                time = std::chrono::duration_cast<std::chrono::duration<double>>(endD0 - endCE).count();time_D_conv0+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(endD1 - endD0).count();time_D_act0+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(endD2 - endD1).count();time_D_conv1+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(endD3 - endD2).count();time_D_act1+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(endD4 - endD3).count();time_D_conv2+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(endD5 - endD4).count();time_D_act2+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(endD6 - endD5).count();time_D_conv3+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(endD7 - endD6).count();time_D_act3+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(endD8 - endD7).count();time_D_conv4+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(endD9 - endD8).count();time_D_act4+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(endDA - endD9).count();time_D_conv5+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(endDB - endDA).count();time_D_act5+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(endDC - endDB).count();time_D_conv6+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(endDD - endDC).count();time_D_act6+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(endDE - endDD).count();time_D_cat+=time; one_runtime+=time;
                /************************avgpool****************************/
                time = std::chrono::duration_cast<std::chrono::duration<double>>(endE1 - endE0).count();time_E_pool0+=time; one_runtime+=time;
                /************************fc**************************/
                time = std::chrono::duration_cast<std::chrono::duration<double>>(endF0 - endE2).count();time_F_conv0+=time; one_runtime+=time;
                if(i>0){
                        std::cout<<i<<"---run:"<<std::endl;
                        std::cout<<"time="<<one_runtime*1000<<"ms"<<std::endl;
                        out<<"one run time"<<","<<one_runtime*1000<<std::endl;
                        total_time+=one_runtime;  
		}
		if(i==0){
                        std::cout<<"First run: "<<std::endl;
                        std::cout<<"time="<<one_runtime*1000<<"ms"<<std::endl;
		}
            }  
        }
        
        arm_compute::utils::NPYLoader save;
        save.save_to_npy2(out_F_conv0, output_filename, is_fortran);
        /************************************************print layer execution time*************/  
        out<<"average time=   "<<","<<total_time*1000/(cycles-1)<<"ms"<<std::endl;
        out<<"Pre-Layers: "<<std::endl;
        out<<"conv0: "<<","<<time_1_conv0*1000/(cycles-1)<<std::endl;
        out<<"pool0: "<<","<<time_1_pool0*1000/(cycles-1)<<std::endl;
        out<<"conv1: "<<","<<time_1_conv1*1000/(cycles-1)<<std::endl;
        out<<"conv2: "<<","<<time_1_conv2*1000/(cycles-1)<<std::endl;
        out<<"relu: "<<","<<time_1_act0*1000/(cycles-1)<<std::endl;
        out<<"pool1: "<<","<<time_1_pool1*1000/(cycles-1)<<std::endl;

        out<<"a3: "<<std::endl;
        out<<"conv0: "<<","<<time_2_conv0*1000/(cycles-1)<<std::endl;;
        out<<"act0: "<<","<<time_2_act0*1000/(cycles-1)<<std::endl;
        out<<"conv1: "<<","<<time_2_conv1*1000/(cycles-1)<<std::endl;;
        out<<"act1: "<<","<<time_2_act1*1000/(cycles-1)<<std::endl;
        out<<"conv2: "<<","<<time_2_conv2*1000/(cycles-1)<<std::endl;;
        out<<"act2: "<<","<<time_2_act2*1000/(cycles-1)<<std::endl;
        out<<"conv3: "<<","<<time_2_conv3*1000/(cycles-1)<<std::endl;;
        out<<"act3: "<<","<<time_2_act3*1000/(cycles-1)<<std::endl;
        out<<"conv4: "<<","<<time_2_conv4*1000/(cycles-1)<<std::endl;;
        out<<"act4: "<<","<<time_2_act4*1000/(cycles-1)<<std::endl;
        out<<"conv5: "<<","<<time_2_conv5*1000/(cycles-1)<<std::endl;;
        out<<"act5: "<<","<<time_2_act5*1000/(cycles-1)<<std::endl;
        out<<"conv6: "<<","<<time_2_conv6*1000/(cycles-1)<<std::endl;;
        out<<"act6: "<<","<<time_2_act6*1000/(cycles-1)<<std::endl;
        out<<"cat: "<<","<<time_2_cat*1000/(cycles-1)<<std::endl;

        out<<"b3: "<<std::endl;
        out<<"conv0: "<<","<<time_3_conv0*1000/(cycles-1)<<std::endl;;
        out<<"act0: "<<","<<time_3_act0*1000/(cycles-1)<<std::endl;
        out<<"conv1: "<<","<<time_3_conv1*1000/(cycles-1)<<std::endl;;
        out<<"act1: "<<","<<time_3_act1*1000/(cycles-1)<<std::endl;
        out<<"conv2: "<<","<<time_3_conv2*1000/(cycles-1)<<std::endl;;
        out<<"act2: "<<","<<time_3_act2*1000/(cycles-1)<<std::endl;
        out<<"conv3: "<<","<<time_3_conv3*1000/(cycles-1)<<std::endl;;
        out<<"act3: "<<","<<time_3_act3*1000/(cycles-1)<<std::endl;
        out<<"conv4: "<<","<<time_3_conv4*1000/(cycles-1)<<std::endl;;
        out<<"act4: "<<","<<time_3_act4*1000/(cycles-1)<<std::endl;
        out<<"conv5: "<<","<<time_3_conv5*1000/(cycles-1)<<std::endl;;
        out<<"act5: "<<","<<time_3_act5*1000/(cycles-1)<<std::endl;
        out<<"conv6: "<<","<<time_3_conv6*1000/(cycles-1)<<std::endl;;
        out<<"act6: "<<","<<time_3_act6*1000/(cycles-1)<<std::endl;
        out<<"cat: "<<","<<time_3_cat*1000/(cycles-1)<<std::endl;

        out<<"c3: "<<std::endl;
        out<<"conv0: "<<","<<time_4_conv0*1000/(cycles-1)<<std::endl;;
        out<<"act0: "<<","<<time_4_act0*1000/(cycles-1)<<std::endl;
        out<<"conv1: "<<","<<time_4_conv1*1000/(cycles-1)<<std::endl;;
        out<<"act1: "<<","<<time_4_act1*1000/(cycles-1)<<std::endl;
        out<<"conv2: "<<","<<time_4_conv2*1000/(cycles-1)<<std::endl;;
        out<<"act2: "<<","<<time_4_act2*1000/(cycles-1)<<std::endl;
        out<<"conv3: "<<","<<time_4_conv3*1000/(cycles-1)<<std::endl;;
        out<<"act3: "<<","<<time_4_act3*1000/(cycles-1)<<std::endl;
        out<<"conv4: "<<","<<time_4_conv4*1000/(cycles-1)<<std::endl;;
        out<<"act4: "<<","<<time_4_act4*1000/(cycles-1)<<std::endl;
        out<<"conv5: "<<","<<time_4_conv5*1000/(cycles-1)<<std::endl;;
        out<<"act5: "<<","<<time_4_act5*1000/(cycles-1)<<std::endl;
        out<<"conv6: "<<","<<time_4_conv6*1000/(cycles-1)<<std::endl;;
        out<<"act6: "<<","<<time_4_act6*1000/(cycles-1)<<std::endl;
        out<<"cat: "<<","<<time_4_cat*1000/(cycles-1)<<std::endl;

        out<<"maxpool: "<<std::endl;
        out<<"pool: "<<","<<time_5_pool0*1000/(cycles-1)<<std::endl;

        out<<"a4: "<<std::endl;
        out<<"conv0: "<<","<<time_6_conv0*1000/(cycles-1)<<std::endl;;
        out<<"act0: "<<","<<time_6_act0*1000/(cycles-1)<<std::endl;
        out<<"conv1: "<<","<<time_6_conv1*1000/(cycles-1)<<std::endl;;
        out<<"act1: "<<","<<time_6_act1*1000/(cycles-1)<<std::endl;
        out<<"conv2: "<<","<<time_6_conv2*1000/(cycles-1)<<std::endl;;
        out<<"act2: "<<","<<time_6_act2*1000/(cycles-1)<<std::endl;
        out<<"conv3: "<<","<<time_6_conv3*1000/(cycles-1)<<std::endl;;
        out<<"act3: "<<","<<time_6_act3*1000/(cycles-1)<<std::endl;
        out<<"conv4: "<<","<<time_6_conv4*1000/(cycles-1)<<std::endl;;
        out<<"act4: "<<","<<time_6_act4*1000/(cycles-1)<<std::endl;
        out<<"conv5: "<<","<<time_6_conv5*1000/(cycles-1)<<std::endl;;
        out<<"act5: "<<","<<time_6_act5*1000/(cycles-1)<<std::endl;
        out<<"conv6: "<<","<<time_6_conv6*1000/(cycles-1)<<std::endl;;
        out<<"act6: "<<","<<time_6_act6*1000/(cycles-1)<<std::endl;
        out<<"cat: "<<","<<time_6_cat*1000/(cycles-1)<<std::endl;

        out<<"b4: "<<std::endl;
        out<<"conv0: "<<","<<time_7_conv0*1000/(cycles-1)<<std::endl;;
        out<<"act0: "<<","<<time_7_act0*1000/(cycles-1)<<std::endl;
        out<<"conv1: "<<","<<time_7_conv1*1000/(cycles-1)<<std::endl;;
        out<<"act1: "<<","<<time_7_act1*1000/(cycles-1)<<std::endl;
        out<<"conv2: "<<","<<time_7_conv2*1000/(cycles-1)<<std::endl;;
        out<<"act2: "<<","<<time_7_act2*1000/(cycles-1)<<std::endl;
        out<<"conv3: "<<","<<time_7_conv3*1000/(cycles-1)<<std::endl;;
        out<<"act3: "<<","<<time_7_act3*1000/(cycles-1)<<std::endl;
        out<<"conv4: "<<","<<time_7_conv4*1000/(cycles-1)<<std::endl;;
        out<<"act4: "<<","<<time_7_act4*1000/(cycles-1)<<std::endl;
        out<<"conv5: "<<","<<time_7_conv5*1000/(cycles-1)<<std::endl;;
        out<<"act5: "<<","<<time_7_act5*1000/(cycles-1)<<std::endl;
        out<<"conv6: "<<","<<time_7_conv6*1000/(cycles-1)<<std::endl;;
        out<<"act6: "<<","<<time_7_act6*1000/(cycles-1)<<std::endl;
        out<<"cat: "<<","<<time_7_cat*1000/(cycles-1)<<std::endl;

        out<<"c4: "<<std::endl;
        out<<"conv0: "<<","<<time_8_conv0*1000/(cycles-1)<<std::endl;;
        out<<"act0: "<<","<<time_8_act0*1000/(cycles-1)<<std::endl;
        out<<"conv1: "<<","<<time_8_conv1*1000/(cycles-1)<<std::endl;;
        out<<"act1: "<<","<<time_8_act1*1000/(cycles-1)<<std::endl;
        out<<"conv2: "<<","<<time_8_conv2*1000/(cycles-1)<<std::endl;;
        out<<"act2: "<<","<<time_8_act2*1000/(cycles-1)<<std::endl;
        out<<"conv3: "<<","<<time_8_conv3*1000/(cycles-1)<<std::endl;;
        out<<"act3: "<<","<<time_8_act3*1000/(cycles-1)<<std::endl;
        out<<"conv4: "<<","<<time_8_conv4*1000/(cycles-1)<<std::endl;;
        out<<"act4: "<<","<<time_8_act4*1000/(cycles-1)<<std::endl;
        out<<"conv5: "<<","<<time_8_conv5*1000/(cycles-1)<<std::endl;;
        out<<"act5: "<<","<<time_8_act5*1000/(cycles-1)<<std::endl;
        out<<"conv6: "<<","<<time_8_conv6*1000/(cycles-1)<<std::endl;;
        out<<"act6: "<<","<<time_8_act6*1000/(cycles-1)<<std::endl;
        out<<"cat: "<<","<<time_8_cat*1000/(cycles-1)<<std::endl;

        out<<"d4: "<<std::endl;
        out<<"conv0: "<<","<<time_9_conv0*1000/(cycles-1)<<std::endl;;
        out<<"act0: "<<","<<time_9_act0*1000/(cycles-1)<<std::endl;
        out<<"conv1: "<<","<<time_9_conv1*1000/(cycles-1)<<std::endl;;
        out<<"act1: "<<","<<time_9_act1*1000/(cycles-1)<<std::endl;
        out<<"conv2: "<<","<<time_9_conv2*1000/(cycles-1)<<std::endl;;
        out<<"act2: "<<","<<time_9_act2*1000/(cycles-1)<<std::endl;
        out<<"conv3: "<<","<<time_9_conv3*1000/(cycles-1)<<std::endl;;
        out<<"act3: "<<","<<time_9_act3*1000/(cycles-1)<<std::endl;
        out<<"conv4: "<<","<<time_9_conv4*1000/(cycles-1)<<std::endl;;
        out<<"act4: "<<","<<time_9_act4*1000/(cycles-1)<<std::endl;
        out<<"conv5: "<<","<<time_9_conv5*1000/(cycles-1)<<std::endl;;
        out<<"act5: "<<","<<time_9_act5*1000/(cycles-1)<<std::endl;
        out<<"conv6: "<<","<<time_9_conv6*1000/(cycles-1)<<std::endl;;
        out<<"act6: "<<","<<time_9_act6*1000/(cycles-1)<<std::endl;
        out<<"cat: "<<","<<time_9_cat*1000/(cycles-1)<<std::endl;

        out<<"e4: "<<std::endl;
        out<<"conv0: "<<","<<time_A_conv0*1000/(cycles-1)<<std::endl;;
        out<<"act0: "<<","<<time_A_act0*1000/(cycles-1)<<std::endl;
        out<<"conv1: "<<","<<time_A_conv1*1000/(cycles-1)<<std::endl;;
        out<<"act1: "<<","<<time_A_act1*1000/(cycles-1)<<std::endl;
        out<<"conv2: "<<","<<time_A_conv2*1000/(cycles-1)<<std::endl;;
        out<<"act2: "<<","<<time_A_act2*1000/(cycles-1)<<std::endl;
        out<<"conv3: "<<","<<time_A_conv3*1000/(cycles-1)<<std::endl;;
        out<<"act3: "<<","<<time_A_act3*1000/(cycles-1)<<std::endl;
        out<<"conv4: "<<","<<time_A_conv4*1000/(cycles-1)<<std::endl;;
        out<<"act4: "<<","<<time_A_act4*1000/(cycles-1)<<std::endl;
        out<<"conv5: "<<","<<time_A_conv5*1000/(cycles-1)<<std::endl;;
        out<<"act5: "<<","<<time_A_act5*1000/(cycles-1)<<std::endl;
        out<<"conv6: "<<","<<time_A_conv6*1000/(cycles-1)<<std::endl;;
        out<<"act6: "<<","<<time_A_act6*1000/(cycles-1)<<std::endl;
        out<<"cat: "<<","<<time_A_cat*1000/(cycles-1)<<std::endl;

        out<<"maxpool: "<<std::endl;
        out<<"pool: "<<","<<time_B_pool0*1000/(cycles-1)<<std::endl;

        out<<"a5: "<<std::endl;
        out<<"conv0: "<<","<<time_C_conv0*1000/(cycles-1)<<std::endl;;
        out<<"act0: "<<","<<time_C_act0*1000/(cycles-1)<<std::endl;
        out<<"conv1: "<<","<<time_C_conv1*1000/(cycles-1)<<std::endl;;
        out<<"act1: "<<","<<time_C_act1*1000/(cycles-1)<<std::endl;
        out<<"conv2: "<<","<<time_C_conv2*1000/(cycles-1)<<std::endl;;
        out<<"act2: "<<","<<time_C_act2*1000/(cycles-1)<<std::endl;
        out<<"conv3: "<<","<<time_C_conv3*1000/(cycles-1)<<std::endl;;
        out<<"act3: "<<","<<time_C_act3*1000/(cycles-1)<<std::endl;
        out<<"conv4: "<<","<<time_C_conv4*1000/(cycles-1)<<std::endl;;
        out<<"act4: "<<","<<time_C_act4*1000/(cycles-1)<<std::endl;
        out<<"conv5: "<<","<<time_C_conv5*1000/(cycles-1)<<std::endl;;
        out<<"act5: "<<","<<time_C_act5*1000/(cycles-1)<<std::endl;
        out<<"conv6: "<<","<<time_C_conv6*1000/(cycles-1)<<std::endl;;
        out<<"act6: "<<","<<time_C_act6*1000/(cycles-1)<<std::endl;
        out<<"cat: "<<","<<time_C_cat*1000/(cycles-1)<<std::endl;

        out<<"b5: "<<std::endl;
        out<<"conv0: "<<","<<time_D_conv0*1000/(cycles-1)<<std::endl;;
        out<<"act0: "<<","<<time_D_act0*1000/(cycles-1)<<std::endl;
        out<<"conv1: "<<","<<time_D_conv1*1000/(cycles-1)<<std::endl;;
        out<<"act1: "<<","<<time_D_act1*1000/(cycles-1)<<std::endl;
        out<<"conv2: "<<","<<time_D_conv2*1000/(cycles-1)<<std::endl;;
        out<<"act2: "<<","<<time_D_act2*1000/(cycles-1)<<std::endl;
        out<<"conv3: "<<","<<time_D_conv3*1000/(cycles-1)<<std::endl;;
        out<<"act3: "<<","<<time_D_act3*1000/(cycles-1)<<std::endl;
        out<<"conv4: "<<","<<time_D_conv4*1000/(cycles-1)<<std::endl;;
        out<<"act4: "<<","<<time_D_act4*1000/(cycles-1)<<std::endl;
        out<<"conv5: "<<","<<time_D_conv5*1000/(cycles-1)<<std::endl;;
        out<<"act5: "<<","<<time_D_act5*1000/(cycles-1)<<std::endl;
        out<<"conv6: "<<","<<time_D_conv6*1000/(cycles-1)<<std::endl;;
        out<<"act6: "<<","<<time_D_act6*1000/(cycles-1)<<std::endl;
        out<<"cat: "<<","<<time_D_cat*1000/(cycles-1)<<std::endl;

        out<<"avgpool: "<<std::endl;
        out<<"pool: "<<","<<time_E_pool0*1000/(cycles-1)<<std::endl;

        out<<"fully-connected: "<<std::endl;
        out<<"fc: "<<","<<time_F_conv0*1000/(cycles-1)<<std::endl;




            std::cout<<"average time=          "<<total_time*1000/(cycles-1)<<"ms"<<std::endl;
            std::cout<<"Pre-Layers: "<<std::endl;
            std::cout<<"conv0: "<<time_1_conv0*1000/(cycles-1)<<"ms"<<std::endl;
            std::cout<<"pool0: "<<time_1_pool0*1000/(cycles-1)<<"ms"<<std::endl;
            std::cout<<"conv1: "<<time_1_conv1*1000/(cycles-1)<<"ms"<<std::endl;
            std::cout<<"conv2: "<<time_1_conv2*1000/(cycles-1)<<"ms"<<std::endl;
            std::cout<<"relu: "<<time_1_act0*1000/(cycles-1)<<"ms"<<std::endl;
            std::cout<<"pool1: "<<time_1_pool1*1000/(cycles-1)<<"ms"<<std::endl;

            std::cout<<"a3: "<<std::endl;
            std::cout<<"conv0: "<<time_2_conv0*1000/(cycles-1)<<"ms"<<std::endl;;
            std::cout<<"act0: "<<time_2_act0*1000/(cycles-1)<<"ms"<<std::endl;
            std::cout<<"conv1: "<<time_2_conv1*1000/(cycles-1)<<"ms"<<std::endl;;
            std::cout<<"act1: "<<time_2_act1*1000/(cycles-1)<<"ms"<<std::endl;
            std::cout<<"conv2: "<<time_2_conv2*1000/(cycles-1)<<"ms"<<std::endl;;
            std::cout<<"act2: "<<time_2_act2*1000/(cycles-1)<<"ms"<<std::endl;
            std::cout<<"conv3: "<<time_2_conv3*1000/(cycles-1)<<"ms"<<std::endl;;
            std::cout<<"act3: "<<time_2_act3*1000/(cycles-1)<<"ms"<<std::endl;
            std::cout<<"conv4: "<<time_2_conv4*1000/(cycles-1)<<"ms"<<std::endl;;
            std::cout<<"act4: "<<time_2_act4*1000/(cycles-1)<<"ms"<<std::endl;
            std::cout<<"conv5: "<<time_2_conv5*1000/(cycles-1)<<"ms"<<std::endl;;
            std::cout<<"act5: "<<time_2_act5*1000/(cycles-1)<<"ms"<<std::endl;
            std::cout<<"conv6: "<<time_2_conv6*1000/(cycles-1)<<"ms"<<std::endl;;
            std::cout<<"act6: "<<time_2_act6*1000/(cycles-1)<<"ms"<<std::endl;
            std::cout<<"cat: "<<time_2_cat*1000/(cycles-1)<<"ms"<<std::endl;

            std::cout<<"b3: "<<std::endl;
            std::cout<<"conv0: "<<time_3_conv0*1000/(cycles-1)<<"ms"<<std::endl;;
            std::cout<<"act0: "<<time_3_act0*1000/(cycles-1)<<"ms"<<std::endl;
            std::cout<<"conv1: "<<time_3_conv1*1000/(cycles-1)<<"ms"<<std::endl;;
            std::cout<<"act1: "<<time_3_act1*1000/(cycles-1)<<"ms"<<std::endl;
            std::cout<<"conv2: "<<time_3_conv2*1000/(cycles-1)<<"ms"<<std::endl;;
            std::cout<<"act2: "<<time_3_act2*1000/(cycles-1)<<"ms"<<std::endl;
            std::cout<<"conv3: "<<time_3_conv3*1000/(cycles-1)<<"ms"<<std::endl;;
            std::cout<<"act3: "<<time_3_act3*1000/(cycles-1)<<"ms"<<std::endl;
            std::cout<<"conv4: "<<time_3_conv4*1000/(cycles-1)<<"ms"<<std::endl;;
            std::cout<<"act4: "<<time_3_act4*1000/(cycles-1)<<"ms"<<std::endl;
            std::cout<<"conv5: "<<time_3_conv5*1000/(cycles-1)<<"ms"<<std::endl;;
            std::cout<<"act5: "<<time_3_act5*1000/(cycles-1)<<"ms"<<std::endl;
            std::cout<<"conv6: "<<time_3_conv6*1000/(cycles-1)<<"ms"<<std::endl;;
            std::cout<<"act6: "<<time_3_act6*1000/(cycles-1)<<"ms"<<std::endl;
            std::cout<<"cat: "<<time_3_cat*1000/(cycles-1)<<"ms"<<std::endl;

            std::cout<<"c3: "<<std::endl;
            std::cout<<"conv0: "<<time_4_conv0*1000/(cycles-1)<<"ms"<<std::endl;;
            std::cout<<"act0: "<<time_4_act0*1000/(cycles-1)<<"ms"<<std::endl;
            std::cout<<"conv1: "<<time_4_conv1*1000/(cycles-1)<<"ms"<<std::endl;;
            std::cout<<"act1: "<<time_4_act1*1000/(cycles-1)<<"ms"<<std::endl;
            std::cout<<"conv2: "<<time_4_conv2*1000/(cycles-1)<<"ms"<<std::endl;;
            std::cout<<"act2: "<<time_4_act2*1000/(cycles-1)<<"ms"<<std::endl;
            std::cout<<"conv3: "<<time_4_conv3*1000/(cycles-1)<<"ms"<<std::endl;;
            std::cout<<"act3: "<<time_4_act3*1000/(cycles-1)<<"ms"<<std::endl;
            std::cout<<"conv4: "<<time_4_conv4*1000/(cycles-1)<<"ms"<<std::endl;;
            std::cout<<"act4: "<<time_4_act4*1000/(cycles-1)<<"ms"<<std::endl;
            std::cout<<"conv5: "<<time_4_conv5*1000/(cycles-1)<<"ms"<<std::endl;;
            std::cout<<"act5: "<<time_4_act5*1000/(cycles-1)<<"ms"<<std::endl;
            std::cout<<"conv6: "<<time_4_conv6*1000/(cycles-1)<<"ms"<<std::endl;;
            std::cout<<"act6: "<<time_4_act6*1000/(cycles-1)<<"ms"<<std::endl;
            std::cout<<"cat: "<<time_4_cat*1000/(cycles-1)<<"ms"<<std::endl;

            std::cout<<"maxpool: "<<std::endl;
            std::cout<<"pool: "<<time_5_pool0*1000/(cycles-1)<<"ms"<<std::endl;

            std::cout<<"a4: "<<std::endl;
            std::cout<<"conv0: "<<time_6_conv0*1000/(cycles-1)<<"ms"<<std::endl;;
            std::cout<<"act0: "<<time_6_act0*1000/(cycles-1)<<"ms"<<std::endl;
            std::cout<<"conv1: "<<time_6_conv1*1000/(cycles-1)<<"ms"<<std::endl;;
            std::cout<<"act1: "<<time_6_act1*1000/(cycles-1)<<"ms"<<std::endl;
            std::cout<<"conv2: "<<time_6_conv2*1000/(cycles-1)<<"ms"<<std::endl;;
            std::cout<<"act2: "<<time_6_act2*1000/(cycles-1)<<"ms"<<std::endl;
            std::cout<<"conv3: "<<time_6_conv3*1000/(cycles-1)<<"ms"<<std::endl;;
            std::cout<<"act3: "<<time_6_act3*1000/(cycles-1)<<"ms"<<std::endl;
            std::cout<<"conv4: "<<time_6_conv4*1000/(cycles-1)<<"ms"<<std::endl;;
            std::cout<<"act4: "<<time_6_act4*1000/(cycles-1)<<"ms"<<std::endl;
            std::cout<<"conv5: "<<time_6_conv5*1000/(cycles-1)<<"ms"<<std::endl;;
            std::cout<<"act5: "<<time_6_act5*1000/(cycles-1)<<"ms"<<std::endl;
            std::cout<<"conv6: "<<time_6_conv6*1000/(cycles-1)<<"ms"<<std::endl;;
            std::cout<<"act6: "<<time_6_act6*1000/(cycles-1)<<"ms"<<std::endl;
            std::cout<<"cat: "<<time_6_cat*1000/(cycles-1)<<"ms"<<std::endl;

            std::cout<<"b4: "<<std::endl;
            std::cout<<"conv0: "<<time_7_conv0*1000/(cycles-1)<<"ms"<<std::endl;;
            std::cout<<"act0: "<<time_7_act0*1000/(cycles-1)<<"ms"<<std::endl;
            std::cout<<"conv1: "<<time_7_conv1*1000/(cycles-1)<<"ms"<<std::endl;;
            std::cout<<"act1: "<<time_7_act1*1000/(cycles-1)<<"ms"<<std::endl;
            std::cout<<"conv2: "<<time_7_conv2*1000/(cycles-1)<<"ms"<<std::endl;;
            std::cout<<"act2: "<<time_7_act2*1000/(cycles-1)<<"ms"<<std::endl;
            std::cout<<"conv3: "<<time_7_conv3*1000/(cycles-1)<<"ms"<<std::endl;;
            std::cout<<"act3: "<<time_7_act3*1000/(cycles-1)<<"ms"<<std::endl;
            std::cout<<"conv4: "<<time_7_conv4*1000/(cycles-1)<<"ms"<<std::endl;;
            std::cout<<"act4: "<<time_7_act4*1000/(cycles-1)<<"ms"<<std::endl;
            std::cout<<"conv5: "<<time_7_conv5*1000/(cycles-1)<<"ms"<<std::endl;;
            std::cout<<"act5: "<<time_7_act5*1000/(cycles-1)<<"ms"<<std::endl;
            std::cout<<"conv6: "<<time_7_conv6*1000/(cycles-1)<<"ms"<<std::endl;;
            std::cout<<"act6: "<<time_7_act6*1000/(cycles-1)<<"ms"<<std::endl;
            std::cout<<"cat: "<<time_7_cat*1000/(cycles-1)<<"ms"<<std::endl;

            std::cout<<"c4: "<<std::endl;
            std::cout<<"conv0: "<<time_8_conv0*1000/(cycles-1)<<"ms"<<std::endl;;
            std::cout<<"act0: "<<time_8_act0*1000/(cycles-1)<<"ms"<<std::endl;
            std::cout<<"conv1: "<<time_8_conv1*1000/(cycles-1)<<"ms"<<std::endl;;
            std::cout<<"act1: "<<time_8_act1*1000/(cycles-1)<<"ms"<<std::endl;
            std::cout<<"conv2: "<<time_8_conv2*1000/(cycles-1)<<"ms"<<std::endl;;
            std::cout<<"act2: "<<time_8_act2*1000/(cycles-1)<<"ms"<<std::endl;
            std::cout<<"conv3: "<<time_8_conv3*1000/(cycles-1)<<"ms"<<std::endl;;
            std::cout<<"act3: "<<time_8_act3*1000/(cycles-1)<<"ms"<<std::endl;
            std::cout<<"conv4: "<<time_8_conv4*1000/(cycles-1)<<"ms"<<std::endl;;
            std::cout<<"act4: "<<time_8_act4*1000/(cycles-1)<<"ms"<<std::endl;
            std::cout<<"conv5: "<<time_8_conv5*1000/(cycles-1)<<"ms"<<std::endl;;
            std::cout<<"act5: "<<time_8_act5*1000/(cycles-1)<<"ms"<<std::endl;
            std::cout<<"conv6: "<<time_8_conv6*1000/(cycles-1)<<"ms"<<std::endl;;
            std::cout<<"act6: "<<time_8_act6*1000/(cycles-1)<<"ms"<<std::endl;
            std::cout<<"cat: "<<time_8_cat*1000/(cycles-1)<<"ms"<<std::endl;

            std::cout<<"d4: "<<std::endl;
            std::cout<<"conv0: "<<time_9_conv0*1000/(cycles-1)<<"ms"<<std::endl;;
            std::cout<<"act0: "<<time_9_act0*1000/(cycles-1)<<"ms"<<std::endl;
            std::cout<<"conv1: "<<time_9_conv1*1000/(cycles-1)<<"ms"<<std::endl;;
            std::cout<<"act1: "<<time_9_act1*1000/(cycles-1)<<"ms"<<std::endl;
            std::cout<<"conv2: "<<time_9_conv2*1000/(cycles-1)<<"ms"<<std::endl;;
            std::cout<<"act2: "<<time_9_act2*1000/(cycles-1)<<"ms"<<std::endl;
            std::cout<<"conv3: "<<time_9_conv3*1000/(cycles-1)<<"ms"<<std::endl;;
            std::cout<<"act3: "<<time_9_act3*1000/(cycles-1)<<"ms"<<std::endl;
            std::cout<<"conv4: "<<time_9_conv4*1000/(cycles-1)<<"ms"<<std::endl;;
            std::cout<<"act4: "<<time_9_act4*1000/(cycles-1)<<"ms"<<std::endl;
            std::cout<<"conv5: "<<time_9_conv5*1000/(cycles-1)<<"ms"<<std::endl;;
            std::cout<<"act5: "<<time_9_act5*1000/(cycles-1)<<"ms"<<std::endl;
            std::cout<<"conv6: "<<time_9_conv6*1000/(cycles-1)<<"ms"<<std::endl;;
            std::cout<<"act6: "<<time_9_act6*1000/(cycles-1)<<"ms"<<std::endl;
            std::cout<<"cat: "<<time_9_cat*1000/(cycles-1)<<"ms"<<std::endl;

            std::cout<<"e4: "<<std::endl;
            std::cout<<"conv0: "<<time_A_conv0*1000/(cycles-1)<<"ms"<<std::endl;;
            std::cout<<"act0: "<<time_A_act0*1000/(cycles-1)<<"ms"<<std::endl;
            std::cout<<"conv1: "<<time_A_conv1*1000/(cycles-1)<<"ms"<<std::endl;;
            std::cout<<"act1: "<<time_A_act1*1000/(cycles-1)<<"ms"<<std::endl;
            std::cout<<"conv2: "<<time_A_conv2*1000/(cycles-1)<<"ms"<<std::endl;;
            std::cout<<"act2: "<<time_A_act2*1000/(cycles-1)<<"ms"<<std::endl;
            std::cout<<"conv3: "<<time_A_conv3*1000/(cycles-1)<<"ms"<<std::endl;;
            std::cout<<"act3: "<<time_A_act3*1000/(cycles-1)<<"ms"<<std::endl;
            std::cout<<"conv4: "<<time_A_conv4*1000/(cycles-1)<<"ms"<<std::endl;;
            std::cout<<"act4: "<<time_A_act4*1000/(cycles-1)<<"ms"<<std::endl;
            std::cout<<"conv5: "<<time_A_conv5*1000/(cycles-1)<<"ms"<<std::endl;;
            std::cout<<"act5: "<<time_A_act5*1000/(cycles-1)<<"ms"<<std::endl;
            std::cout<<"conv6: "<<time_A_conv6*1000/(cycles-1)<<"ms"<<std::endl;;
            std::cout<<"act6: "<<time_A_act6*1000/(cycles-1)<<"ms"<<std::endl;
            std::cout<<"cat: "<<time_A_cat*1000/(cycles-1)<<"ms"<<std::endl;

            std::cout<<"maxpool: "<<std::endl;
            std::cout<<"pool: "<<time_B_pool0*1000/(cycles-1)<<"ms"<<std::endl;

            std::cout<<"a5: "<<std::endl;
            std::cout<<"conv0: "<<time_C_conv0*1000/(cycles-1)<<"ms"<<std::endl;;
            std::cout<<"act0: "<<time_C_act0*1000/(cycles-1)<<"ms"<<std::endl;
            std::cout<<"conv1: "<<time_C_conv1*1000/(cycles-1)<<"ms"<<std::endl;;
            std::cout<<"act1: "<<time_C_act1*1000/(cycles-1)<<"ms"<<std::endl;
            std::cout<<"conv2: "<<time_C_conv2*1000/(cycles-1)<<"ms"<<std::endl;;
            std::cout<<"act2: "<<time_C_act2*1000/(cycles-1)<<"ms"<<std::endl;
            std::cout<<"conv3: "<<time_C_conv3*1000/(cycles-1)<<"ms"<<std::endl;;
            std::cout<<"act3: "<<time_C_act3*1000/(cycles-1)<<"ms"<<std::endl;
            std::cout<<"conv4: "<<time_C_conv4*1000/(cycles-1)<<"ms"<<std::endl;;
            std::cout<<"act4: "<<time_C_act4*1000/(cycles-1)<<"ms"<<std::endl;
            std::cout<<"conv5: "<<time_C_conv5*1000/(cycles-1)<<"ms"<<std::endl;;
            std::cout<<"act5: "<<time_C_act5*1000/(cycles-1)<<"ms"<<std::endl;
            std::cout<<"conv6: "<<time_C_conv6*1000/(cycles-1)<<"ms"<<std::endl;;
            std::cout<<"act6: "<<time_C_act6*1000/(cycles-1)<<"ms"<<std::endl;
            std::cout<<"cat: "<<time_C_cat*1000/(cycles-1)<<"ms"<<std::endl;

            std::cout<<"b5: "<<std::endl;
            std::cout<<"conv0: "<<time_D_conv0*1000/(cycles-1)<<"ms"<<std::endl;;
            std::cout<<"act0: "<<time_D_act0*1000/(cycles-1)<<"ms"<<std::endl;
            std::cout<<"conv1: "<<time_D_conv1*1000/(cycles-1)<<"ms"<<std::endl;;
            std::cout<<"act1: "<<time_D_act1*1000/(cycles-1)<<"ms"<<std::endl;
            std::cout<<"conv2: "<<time_D_conv2*1000/(cycles-1)<<"ms"<<std::endl;;
            std::cout<<"act2: "<<time_D_act2*1000/(cycles-1)<<"ms"<<std::endl;
            std::cout<<"conv3: "<<time_D_conv3*1000/(cycles-1)<<"ms"<<std::endl;;
            std::cout<<"act3: "<<time_D_act3*1000/(cycles-1)<<"ms"<<std::endl;
            std::cout<<"conv4: "<<time_D_conv4*1000/(cycles-1)<<"ms"<<std::endl;;
            std::cout<<"act4: "<<time_D_act4*1000/(cycles-1)<<"ms"<<std::endl;
            std::cout<<"conv5: "<<time_D_conv5*1000/(cycles-1)<<"ms"<<std::endl;;
            std::cout<<"act5: "<<time_D_act5*1000/(cycles-1)<<"ms"<<std::endl;
            std::cout<<"conv6: "<<time_D_conv6*1000/(cycles-1)<<"ms"<<std::endl;;
            std::cout<<"act6: "<<time_D_act6*1000/(cycles-1)<<"ms"<<std::endl;
            std::cout<<"cat: "<<time_D_cat*1000/(cycles-1)<<"ms"<<std::endl;

            std::cout<<"avgpool: "<<std::endl;
            std::cout<<"pool: "<<time_E_pool0*1000/(cycles-1)<<"ms"<<std::endl;

            std::cout<<"fully-connected: "<<std::endl;
            std::cout<<"fc: "<<time_F_conv0*1000/(cycles-1)<<"ms"<<std::endl;

    }
private:
    unsigned int precision[74][4]={
        {7, 5, 5, 5},{6, 8, 5, 5},{10, 7, 5, 5},{7, 7, 5, 5},{7, 6, 5, 5},
        {3, 6, 5, 5},{7, 8, 5, 5},{7, 6, 5, 5},{8, 7, 5, 5},{8, 7, 5, 5},
        {7, 7, 5, 5},{5, 6, 5, 5},{8, 6, 5, 5},{6, 6, 5, 5},{8, 6, 5, 5},
        {6, 8, 5, 5},{7, 6, 5, 5},{8, 7, 5, 5},{7, 4, 5, 5},{9, 8, 5, 5},
        {8, 8, 5, 5},{7, 5, 5, 5},{6, 6, 5, 5},{6, 7, 5, 5},{8, 5, 5, 5},
        {7, 5, 5, 5},{7, 7, 5, 5},{8, 8, 5, 5},{7, 5, 5, 5},{7, 8, 5, 5},
        {6, 6, 5, 5},{7, 5, 5, 5},{5, 6, 5, 5},{9, 5, 5, 5},{8, 6, 5, 5},
        {8, 7, 5, 5},{7, 7, 5, 5},{8, 8, 5, 5},{7, 5, 5, 5},{8, 6, 5, 5},
        {7, 7, 5, 5},{6, 6, 5, 5},{6, 7, 5, 5},{6, 6, 5, 5},{7, 6, 5, 5},
        {7, 6, 5, 5},{8, 5, 5, 5},{8, 7, 5, 5},{7, 7, 5, 5},{8, 6, 5, 5},
        {8, 7, 5, 5},{7, 8, 5, 5},{8, 9, 5, 5},{8, 8, 5, 5},{5, 8, 5, 5},
        {8, 8, 5, 5},{8, 6, 5, 5},{8, 6, 5, 5},{7, 6, 5, 5},{6, 7, 5, 5},
        {8, 9, 5, 5},{9, 8, 5, 5},{7, 8, 5, 5},{8, 6, 5, 5},{8, 7, 5, 5},
        {6, 8, 5, 5},{2, 2, 5, 5},{8, 7, 5, 5},{2, 5, 5, 5},{8, 6, 5, 5},
        {9, 8, 5, 5},{2, 6, 5, 5},{2, 5, 5, 5},{8, 8, 5, 5}
    };
    string Q_table_datapath="/media/sdcard/ComputeLibrary/data/neon_googlenet/Q_table/";
    string WT_buffer_datapath="/media/sdcard/ComputeLibrary/data/neon_googlenet/WT_buffer/";
    string bias_datapath="/media/sdcard/ComputeLibrary/data/neon_googlenet/Bias/";
    string Q_table_name[74]={
        "module.pre_layers.1.weight.Q_table.npy",
        "module.pre_layers.4.weight.Q_table.npy",
        "module.pre_layers.6.weight.Q_table.npy",
        "module.a3.b1.conv1.weight.Q_table.npy",
        "module.a3.b2.conv1.weight.Q_table.npy",
        "module.a3.b2.conv2.weight.Q_table.npy",
        "module.a3.b3.conv1.weight.Q_table.npy",
        "module.a3.b3.conv2.weight.Q_table.npy",
        "module.a3.b3.conv3.weight.Q_table.npy",
        "module.a3.b4.conv1.weight.Q_table.npy",
        "module.b3.b1.conv1.weight.Q_table.npy",
        "module.b3.b2.conv1.weight.Q_table.npy",
        "module.b3.b2.conv2.weight.Q_table.npy",
        "module.b3.b3.conv1.weight.Q_table.npy",
        "module.b3.b3.conv2.weight.Q_table.npy",
        "module.b3.b3.conv3.weight.Q_table.npy",
        "module.b3.b4.conv1.weight.Q_table.npy",
        "module.c3.b1.conv1.weight.Q_table.npy",
        "module.c3.b2.conv1.weight.Q_table.npy",
        "module.c3.b2.conv2.weight.Q_table.npy",
        "module.c3.b3.conv1.weight.Q_table.npy",
        "module.c3.b3.conv2.weight.Q_table.npy",
        "module.c3.b3.conv3.weight.Q_table.npy",
        "module.c3.b4.conv1.weight.Q_table.npy",
        "module.a4.b1.conv1.weight.Q_table.npy",
        "module.a4.b2.conv1.weight.Q_table.npy",
        "module.a4.b2.conv2.weight.Q_table.npy",
        "module.a4.b3.conv1.weight.Q_table.npy",
        "module.a4.b3.conv2.weight.Q_table.npy",
        "module.a4.b3.conv3.weight.Q_table.npy",
        "module.a4.b4.conv1.weight.Q_table.npy",
        "module.b4.b1.conv1.weight.Q_table.npy",
        "module.b4.b2.conv1.weight.Q_table.npy",
        "module.b4.b2.conv2.weight.Q_table.npy",
        "module.b4.b3.conv1.weight.Q_table.npy",
        "module.b4.b3.conv2.weight.Q_table.npy",
        "module.b4.b3.conv3.weight.Q_table.npy",
        "module.b4.b4.conv1.weight.Q_table.npy",
        "module.c4.b1.conv1.weight.Q_table.npy",
        "module.c4.b2.conv1.weight.Q_table.npy",
        "module.c4.b2.conv2.weight.Q_table.npy",
        "module.c4.b3.conv1.weight.Q_table.npy",
        "module.c4.b3.conv2.weight.Q_table.npy",
        "module.c4.b3.conv3.weight.Q_table.npy",
        "module.c4.b4.conv1.weight.Q_table.npy",
        "module.d4.b1.conv1.weight.Q_table.npy",
        "module.d4.b2.conv1.weight.Q_table.npy",
        "module.d4.b2.conv2.weight.Q_table.npy",
        "module.d4.b3.conv1.weight.Q_table.npy",
        "module.d4.b3.conv2.weight.Q_table.npy",
        "module.d4.b3.conv3.weight.Q_table.npy",
        "module.d4.b4.conv1.weight.Q_table.npy",
        "module.e4.b1.conv1.weight.Q_table.npy",
        "module.e4.b2.conv1.weight.Q_table.npy",
        "module.e4.b2.conv2.weight.Q_table.npy",
        "module.e4.b3.conv1.weight.Q_table.npy",
        "module.e4.b3.conv2.weight.Q_table.npy",
        "module.e4.b3.conv3.weight.Q_table.npy",
        "module.e4.b4.conv1.weight.Q_table.npy",
        "module.a5.b1.conv1.weight.Q_table.npy",
        "module.a5.b2.conv1.weight.Q_table.npy",
        "module.a5.b2.conv2.weight.Q_table.npy",
        "module.a5.b3.conv1.weight.Q_table.npy",
        "module.a5.b3.conv2.weight.Q_table.npy",
        "module.a5.b3.conv3.weight.Q_table.npy",
        "module.a5.b4.conv1.weight.Q_table.npy",
        "module.b5.b1.conv1.weight.Q_table.npy",
        "module.b5.b2.conv1.weight.Q_table.npy",
        "module.b5.b2.conv2.weight.Q_table.npy",
        "module.b5.b3.conv1.weight.Q_table.npy",
        "module.b5.b3.conv2.weight.Q_table.npy",
        "module.b5.b3.conv3.weight.Q_table.npy",
        "module.b5.b4.conv1.weight.Q_table.npy",
        "module.linear.weight.Q_table.npy"
    };
    string WT_buffer_name[74]={
             "module.pre_layers.1.weight.WT_buffer.npy",
        "module.pre_layers.4.weight.WT_buffer.npy",
        "module.pre_layers.6.weight.WT_buffer.npy",
        "module.a3.b1.conv1.weight.WT_buffer.npy",
        "module.a3.b2.conv1.weight.WT_buffer.npy",
        "module.a3.b2.conv2.weight.WT_buffer.npy",
        "module.a3.b3.conv1.weight.WT_buffer.npy",
        "module.a3.b3.conv2.weight.WT_buffer.npy",
        "module.a3.b3.conv3.weight.WT_buffer.npy",
        "module.a3.b4.conv1.weight.WT_buffer.npy",
        "module.b3.b1.conv1.weight.WT_buffer.npy",
        "module.b3.b2.conv1.weight.WT_buffer.npy",
        "module.b3.b2.conv2.weight.WT_buffer.npy",
        "module.b3.b3.conv1.weight.WT_buffer.npy",
        "module.b3.b3.conv2.weight.WT_buffer.npy",
        "module.b3.b3.conv3.weight.WT_buffer.npy",
        "module.b3.b4.conv1.weight.WT_buffer.npy",
        "module.c3.b1.conv1.weight.WT_buffer.npy",
        "module.c3.b2.conv1.weight.WT_buffer.npy",
        "module.c3.b2.conv2.weight.WT_buffer.npy",
        "module.c3.b3.conv1.weight.WT_buffer.npy",
        "module.c3.b3.conv2.weight.WT_buffer.npy",
        "module.c3.b3.conv3.weight.WT_buffer.npy",
        "module.c3.b4.conv1.weight.WT_buffer.npy",
        "module.a4.b1.conv1.weight.WT_buffer.npy",
        "module.a4.b2.conv1.weight.WT_buffer.npy",
        "module.a4.b2.conv2.weight.WT_buffer.npy",
        "module.a4.b3.conv1.weight.WT_buffer.npy",
        "module.a4.b3.conv2.weight.WT_buffer.npy",
        "module.a4.b3.conv3.weight.WT_buffer.npy",
        "module.a4.b4.conv1.weight.WT_buffer.npy",
        "module.b4.b1.conv1.weight.WT_buffer.npy",
        "module.b4.b2.conv1.weight.WT_buffer.npy",
        "module.b4.b2.conv2.weight.WT_buffer.npy",
        "module.b4.b3.conv1.weight.WT_buffer.npy",
        "module.b4.b3.conv2.weight.WT_buffer.npy",
        "module.b4.b3.conv3.weight.WT_buffer.npy",
        "module.b4.b4.conv1.weight.WT_buffer.npy",
        "module.c4.b1.conv1.weight.WT_buffer.npy",
        "module.c4.b2.conv1.weight.WT_buffer.npy",
        "module.c4.b2.conv2.weight.WT_buffer.npy",
        "module.c4.b3.conv1.weight.WT_buffer.npy",
        "module.c4.b3.conv2.weight.WT_buffer.npy",
        "module.c4.b3.conv3.weight.WT_buffer.npy",
        "module.c4.b4.conv1.weight.WT_buffer.npy",
        "module.d4.b1.conv1.weight.WT_buffer.npy",
        "module.d4.b2.conv1.weight.WT_buffer.npy",
        "module.d4.b2.conv2.weight.WT_buffer.npy",
        "module.d4.b3.conv1.weight.WT_buffer.npy",
        "module.d4.b3.conv2.weight.WT_buffer.npy",
        "module.d4.b3.conv3.weight.WT_buffer.npy",
        "module.d4.b4.conv1.weight.WT_buffer.npy",
        "module.e4.b1.conv1.weight.WT_buffer.npy",
        "module.e4.b2.conv1.weight.WT_buffer.npy",
        "module.e4.b2.conv2.weight.WT_buffer.npy",
        "module.e4.b3.conv1.weight.WT_buffer.npy",
        "module.e4.b3.conv2.weight.WT_buffer.npy",
        "module.e4.b3.conv3.weight.WT_buffer.npy",
        "module.e4.b4.conv1.weight.WT_buffer.npy",
        "module.a5.b1.conv1.weight.WT_buffer.npy",
        "module.a5.b2.conv1.weight.WT_buffer.npy",
        "module.a5.b2.conv2.weight.WT_buffer.npy",
        "module.a5.b3.conv1.weight.WT_buffer.npy",
        "module.a5.b3.conv2.weight.WT_buffer.npy",
        "module.a5.b3.conv3.weight.WT_buffer.npy",
        "module.a5.b4.conv1.weight.WT_buffer.npy",
        "module.b5.b1.conv1.weight.WT_buffer.npy",
        "module.b5.b2.conv1.weight.WT_buffer.npy",
        "module.b5.b2.conv2.weight.WT_buffer.npy",
        "module.b5.b3.conv1.weight.WT_buffer.npy",
        "module.b5.b3.conv2.weight.WT_buffer.npy",
        "module.b5.b3.conv3.weight.WT_buffer.npy",
        "module.b5.b4.conv1.weight.WT_buffer.npy",
        "module.linear.weight.WT_buffer.npy"
    };
    string bias_name[74]={
             "module.pre_layers.1.bias.npy",
        "module.pre_layers.4.bias.npy",
        "module.pre_layers.6.bias.npy",
        "module.a3.b1.conv1.bias.npy",
        "module.a3.b2.conv1.bias.npy",
        "module.a3.b2.conv2.bias.npy",
        "module.a3.b3.conv1.bias.npy",
        "module.a3.b3.conv2.bias.npy",
        "module.a3.b3.conv3.bias.npy",
        "module.a3.b4.conv1.bias.npy",
        "module.b3.b1.conv1.bias.npy",
        "module.b3.b2.conv1.bias.npy",
        "module.b3.b2.conv2.bias.npy",
        "module.b3.b3.conv1.bias.npy",
        "module.b3.b3.conv2.bias.npy",
        "module.b3.b3.conv3.bias.npy",
        "module.b3.b4.conv1.bias.npy",
        "module.c3.b1.conv1.bias.npy",
        "module.c3.b2.conv1.bias.npy",
        "module.c3.b2.conv2.bias.npy",
        "module.c3.b3.conv1.bias.npy",
        "module.c3.b3.conv2.bias.npy",
        "module.c3.b3.conv3.bias.npy",
        "module.c3.b4.conv1.bias.npy",
        "module.a4.b1.conv1.bias.npy",
        "module.a4.b2.conv1.bias.npy",
        "module.a4.b2.conv2.bias.npy",
        "module.a4.b3.conv1.bias.npy",
        "module.a4.b3.conv2.bias.npy",
        "module.a4.b3.conv3.bias.npy",
        "module.a4.b4.conv1.bias.npy",
        "module.b4.b1.conv1.bias.npy",
        "module.b4.b2.conv1.bias.npy",
        "module.b4.b2.conv2.bias.npy",
        "module.b4.b3.conv1.bias.npy",
        "module.b4.b3.conv2.bias.npy",
        "module.b4.b3.conv3.bias.npy",
        "module.b4.b4.conv1.bias.npy",
        "module.c4.b1.conv1.bias.npy",
        "module.c4.b2.conv1.bias.npy",
        "module.c4.b2.conv2.bias.npy",
        "module.c4.b3.conv1.bias.npy",
        "module.c4.b3.conv2.bias.npy",
        "module.c4.b3.conv3.bias.npy",
        "module.c4.b4.conv1.bias.npy",
        "module.d4.b1.conv1.bias.npy",
        "module.d4.b2.conv1.bias.npy",
        "module.d4.b2.conv2.bias.npy",
        "module.d4.b3.conv1.bias.npy",
        "module.d4.b3.conv2.bias.npy",
        "module.d4.b3.conv3.bias.npy",
        "module.d4.b4.conv1.bias.npy",
        "module.e4.b1.conv1.bias.npy",
        "module.e4.b2.conv1.bias.npy",
        "module.e4.b2.conv2.bias.npy",
        "module.e4.b3.conv1.bias.npy",
        "module.e4.b3.conv2.bias.npy",
        "module.e4.b3.conv3.bias.npy",
        "module.e4.b4.conv1.bias.npy",
        "module.a5.b1.conv1.bias.npy",
        "module.a5.b2.conv1.bias.npy",
        "module.a5.b2.conv2.bias.npy",
        "module.a5.b3.conv1.bias.npy",
        "module.a5.b3.conv2.bias.npy",
        "module.a5.b3.conv3.bias.npy",
        "module.a5.b4.conv1.bias.npy",
        "module.b5.b1.conv1.bias.npy",
        "module.b5.b2.conv1.bias.npy",
        "module.b5.b2.conv2.bias.npy",
        "module.b5.b3.conv1.bias.npy",
        "module.b5.b3.conv2.bias.npy",
        "module.b5.b3.conv3.bias.npy",
        "module.b5.b4.conv1.bias.npy",
        "module.linear.bias.npy"
    };
    /**********************************************Tensor and Layer Defination******************************************/
    /***********************************************BatchNorm2D Not Defination***************************************/
        bool is_fortran{};
        string output_filename="/media/sdcard/ComputeLibrary/data/neon_googlenet/output.npy";
        Tensor src{}; 
        /*****************pre-layers=1***************/
        Tensor Q_table_1_conv0{}; Tensor WT_buffer_1_conv0{}; Tensor bias_1_conv0{}; 
        Tensor Q_table_1_conv1{}; Tensor WT_buffer_1_conv1{}; Tensor bias_1_conv1{};
        Tensor Q_table_1_conv2{}; Tensor WT_buffer_1_conv2{}; Tensor bias_1_conv2{};
        Tensor out_1_conv0{}; Tensor out_1_conv1{}; Tensor out_1_conv2{};
        Tensor out_1_pool0{}; Tensor out_1_pool1{};
        Tensor out_1_act0{};
        Tensor out_1_sf0{}; Tensor out_1_sf1{};
        Tensor out_1_fs0{}; Tensor out_1_fs1{};

        NEABMConvolutionLayer _1_conv0{}; NEABMConvolutionLayer _1_conv1{}; NEABMConvolutionLayer _1_conv2{};
        NEActivationLayer _1_act0{};
        NEPoolingLayer _1_pool0{}; NEPoolingLayer _1_pool1{};
        NES8toF32Layer _1_sf0{}; NES8toF32Layer _1_sf1{};
        NEF32toS8Layer _1_fs0{}; NEF32toS8Layer _1_fs1{};

        /***********************a3=2******************/
        /********a3-b1************/
        Tensor Q_table_2_conv0{}; Tensor WT_buffer_2_conv0{}; Tensor bias_2_conv0{};
        Tensor out_2_conv0{};
        Tensor out_2_act0{};
        /*******a3-b2************/
        Tensor Q_table_2_conv1{}; Tensor WT_buffer_2_conv1{}; Tensor bias_2_conv1{};
        Tensor Q_table_2_conv2{}; Tensor WT_buffer_2_conv2{}; Tensor bias_2_conv2{};
        Tensor out_2_conv1{}; Tensor out_2_conv2{};
        Tensor out_2_act1{};     Tensor out_2_act2{};
        /*******a3-b3*************/
        Tensor Q_table_2_conv3{}; Tensor WT_buffer_2_conv3{}; Tensor bias_2_conv3{};
        Tensor Q_table_2_conv4{}; Tensor WT_buffer_2_conv4{}; Tensor bias_2_conv4{};
        Tensor Q_table_2_conv5{}; Tensor WT_buffer_2_conv5{}; Tensor bias_2_conv5{};
        Tensor out_2_conv3{}; Tensor out_2_conv4{}; Tensor out_2_conv5{};
        Tensor out_2_act3{};     Tensor out_2_act4{};     Tensor out_2_act5{};
        /*******a3-b4*************/
        Tensor Q_table_2_conv6{}; Tensor WT_buffer_2_conv6{}; Tensor bias_2_conv6{};
        Tensor out_2_conv6{}; 
        Tensor out_2_act6{};
        Tensor out_2_cat{};
        Tensor out_2_sf0{}, out_2_sf1{}, out_2_sf2{}, out_2_sf3{};Tensor out_2_fs0{};

        NEABMConvolutionLayer _2_conv0{}, _2_conv1{}, _2_conv2{}, _2_conv3{}, _2_conv4{}, _2_conv5{}, _2_conv6{};
        NEActivationLayer _2_act0{}, _2_act1{}, _2_act2{}, _2_act3{}, _2_act4{}, _2_act5{}, _2_act6{};
        NEConcatenateLayer _2_cat{};
        NES8toF32Layer _2_sf0{}, _2_sf1{}, _2_sf2{}, _2_sf3{}; NEF32toS8Layer _2_fs0{};
        /***********************b3=3******************/
        /********b3-b1************/
        Tensor Q_table_3_conv0{}; Tensor WT_buffer_3_conv0{}; Tensor bias_3_conv0{};
        Tensor out_3_conv0{};
        Tensor out_3_act0{};
        /*******b3-b2************/
        Tensor Q_table_3_conv1{}; Tensor WT_buffer_3_conv1{}; Tensor bias_3_conv1{};
        Tensor Q_table_3_conv2{}; Tensor WT_buffer_3_conv2{}; Tensor bias_3_conv2{};
        Tensor out_3_conv1{}; Tensor out_3_conv2{};
        Tensor out_3_act1{};     Tensor out_3_act2{};
        /*******b3-b3*************/
        Tensor Q_table_3_conv3{}; Tensor WT_buffer_3_conv3{}; Tensor bias_3_conv3{};
        Tensor Q_table_3_conv4{}; Tensor WT_buffer_3_conv4{}; Tensor bias_3_conv4{};
        Tensor Q_table_3_conv5{}; Tensor WT_buffer_3_conv5{}; Tensor bias_3_conv5{};
        Tensor out_3_conv3{}; Tensor out_3_conv4{}; Tensor out_3_conv5{};
        Tensor out_3_act3{};     Tensor out_3_act4{};     Tensor out_3_act5{};
        /*******b3-b4*************/
        Tensor Q_table_3_conv6{}; Tensor WT_buffer_3_conv6{}; Tensor bias_3_conv6{};
        Tensor out_3_conv6{}; 
        Tensor out_3_act6{};
        Tensor out_3_cat{};
        Tensor out_3_sf0{}, out_3_sf1{}, out_3_sf2{}, out_3_sf3{};Tensor out_3_fs0{};

        NEABMConvolutionLayer _3_conv0{}, _3_conv1{}, _3_conv2{}, _3_conv3{}, _3_conv4{}, _3_conv5{}, _3_conv6{};
        NEActivationLayer _3_act0{}, _3_act1{}, _3_act2{}, _3_act3{}, _3_act4{}, _3_act5{}, _3_act6{};
        NEConcatenateLayer _3_cat{};
        NES8toF32Layer _3_sf0{}, _3_sf1{}, _3_sf2{}, _3_sf3{}; NEF32toS8Layer _3_fs0{};
        /***********************c3=4******************/
        /********c3-b1************/
        Tensor Q_table_4_conv0{}; Tensor WT_buffer_4_conv0{}; Tensor bias_4_conv0{};
        Tensor out_4_conv0{};
        Tensor out_4_act0{};
        /*******c3-b2************/
        Tensor Q_table_4_conv1{}; Tensor WT_buffer_4_conv1{}; Tensor bias_4_conv1{};
        Tensor Q_table_4_conv2{}; Tensor WT_buffer_4_conv2{}; Tensor bias_4_conv2{};
        Tensor out_4_conv1{}; Tensor out_4_conv2{};
        Tensor out_4_act1{};     Tensor out_4_act2{};
        /*******c3-b3*************/
        Tensor Q_table_4_conv3{}; Tensor WT_buffer_4_conv3{}; Tensor bias_4_conv3{};
        Tensor Q_table_4_conv4{}; Tensor WT_buffer_4_conv4{}; Tensor bias_4_conv4{};
        Tensor Q_table_4_conv5{}; Tensor WT_buffer_4_conv5{}; Tensor bias_4_conv5{};
        Tensor out_4_conv3{}; Tensor out_4_conv4{}; Tensor out_4_conv5{};
        Tensor out_4_act3{};     Tensor out_4_act4{};     Tensor out_4_act5{};
        /*******c3-b4*************/
        Tensor Q_table_4_conv6{}; Tensor WT_buffer_4_conv6{}; Tensor bias_4_conv6{};
        Tensor out_4_conv6{}; 
        Tensor out_4_act6{};
        Tensor out_4_cat{};
        Tensor out_4_sf0{}, out_4_sf1{}, out_4_sf2{}, out_4_sf3{};Tensor out_4_fs0{};

        NEABMConvolutionLayer _4_conv0{}, _4_conv1{}, _4_conv2{}, _4_conv3{}, _4_conv4{}, _4_conv5{}, _4_conv6{};
        NEActivationLayer _4_act0{}, _4_act1{}, _4_act2{}, _4_act3{}, _4_act4{}, _4_act5{}, _4_act6{};
        NEConcatenateLayer _4_cat{};
        NES8toF32Layer _4_sf0{}, _4_sf1{}, _4_sf2{}, _4_sf3{}; NEF32toS8Layer _4_fs0{};
        /***********************maxpool******************/
        Tensor out_5_pool0{};
        Tensor out_5_sf0{}; Tensor out_5_fs0{};

        NEPoolingLayer _5_pool0{};
        NES8toF32Layer _5_sf0{}; NEF32toS8Layer _5_fs0{};
        /***********************a4=6******************/
        /********a4-b1************/
        Tensor Q_table_6_conv0{}; Tensor WT_buffer_6_conv0{}; Tensor bias_6_conv0{};
        Tensor out_6_conv0{};
        Tensor out_6_act0{};
        /*******a4-b2************/
        Tensor Q_table_6_conv1{}; Tensor WT_buffer_6_conv1{}; Tensor bias_6_conv1{};
        Tensor Q_table_6_conv2{}; Tensor WT_buffer_6_conv2{}; Tensor bias_6_conv2{};
        Tensor out_6_conv1{}; Tensor out_6_conv2{};
        Tensor out_6_act1{};     Tensor out_6_act2{};
        /*******a4-b3*************/
        Tensor Q_table_6_conv3{}; Tensor WT_buffer_6_conv3{}; Tensor bias_6_conv3{};
        Tensor Q_table_6_conv4{}; Tensor WT_buffer_6_conv4{}; Tensor bias_6_conv4{};
        Tensor Q_table_6_conv5{}; Tensor WT_buffer_6_conv5{}; Tensor bias_6_conv5{};
        Tensor out_6_conv3{}; Tensor out_6_conv4{}; Tensor out_6_conv5{};
        Tensor out_6_act3{};     Tensor out_6_act4{};     Tensor out_6_act5{};
        /*******a4-b4*************/
        Tensor Q_table_6_conv6{}; Tensor WT_buffer_6_conv6{}; Tensor bias_6_conv6{};
        Tensor out_6_conv6{}; 
        Tensor out_6_act6{};
        Tensor out_6_cat{};
        Tensor out_6_sf0{}, out_6_sf1{}, out_6_sf2{}, out_6_sf3{};Tensor out_6_fs0{};

        NEABMConvolutionLayer _6_conv0{}, _6_conv1{}, _6_conv2{}, _6_conv3{}, _6_conv4{}, _6_conv5{}, _6_conv6{};
        NEActivationLayer _6_act0{}, _6_act1{}, _6_act2{}, _6_act3{}, _6_act4{}, _6_act5{}, _6_act6{};
        NEConcatenateLayer _6_cat{};
        NES8toF32Layer _6_sf0{}, _6_sf1{}, _6_sf2{}, _6_sf3{}; NEF32toS8Layer _6_fs0{};
        /***********************b4=7******************/
        /********b4-b1************/
        Tensor Q_table_7_conv0{}; Tensor WT_buffer_7_conv0{}; Tensor bias_7_conv0{};
        Tensor out_7_conv0{};
        Tensor out_7_act0{};
        /*******b4-b2************/
        Tensor Q_table_7_conv1{}; Tensor WT_buffer_7_conv1{}; Tensor bias_7_conv1{};
        Tensor Q_table_7_conv2{}; Tensor WT_buffer_7_conv2{}; Tensor bias_7_conv2{};
        Tensor out_7_conv1{}; Tensor out_7_conv2{};
        Tensor out_7_act1{};     Tensor out_7_act2{};
        /*******b4-b3*************/
        Tensor Q_table_7_conv3{}; Tensor WT_buffer_7_conv3{}; Tensor bias_7_conv3{};
        Tensor Q_table_7_conv4{}; Tensor WT_buffer_7_conv4{}; Tensor bias_7_conv4{};
        Tensor Q_table_7_conv5{}; Tensor WT_buffer_7_conv5{}; Tensor bias_7_conv5{};
        Tensor out_7_conv3{}; Tensor out_7_conv4{}; Tensor out_7_conv5{};
        Tensor out_7_act3{};     Tensor out_7_act4{};     Tensor out_7_act5{};
        /*******b4-b4*************/
        Tensor Q_table_7_conv6{}; Tensor WT_buffer_7_conv6{}; Tensor bias_7_conv6{};
        Tensor out_7_conv6{}; 
        Tensor out_7_act6{};
        Tensor out_7_cat{};
        Tensor out_7_sf0{}, out_7_sf1{}, out_7_sf2{}, out_7_sf3{};Tensor out_7_fs0{};

        NEABMConvolutionLayer _7_conv0{}, _7_conv1{}, _7_conv2{}, _7_conv3{}, _7_conv4{}, _7_conv5{}, _7_conv6{};
        NEActivationLayer _7_act0{}, _7_act1{}, _7_act2{}, _7_act3{}, _7_act4{}, _7_act5{}, _7_act6{};
        NEConcatenateLayer _7_cat{};
        NES8toF32Layer _7_sf0{}, _7_sf1{}, _7_sf2{}, _7_sf3{}; NEF32toS8Layer _7_fs0{};
        /***********************c4=8******************/
        /********c4-b1************/
        Tensor Q_table_8_conv0{}; Tensor WT_buffer_8_conv0{}; Tensor bias_8_conv0{};
        Tensor out_8_conv0{};
        Tensor out_8_act0{};
        /*******c4-b2************/
        Tensor Q_table_8_conv1{}; Tensor WT_buffer_8_conv1{}; Tensor bias_8_conv1{};
        Tensor Q_table_8_conv2{}; Tensor WT_buffer_8_conv2{}; Tensor bias_8_conv2{};
        Tensor out_8_conv1{}; Tensor out_8_conv2{};
        Tensor out_8_act1{};     Tensor out_8_act2{};
        /*******c4-b3*************/
        Tensor Q_table_8_conv3{}; Tensor WT_buffer_8_conv3{}; Tensor bias_8_conv3{};
        Tensor Q_table_8_conv4{}; Tensor WT_buffer_8_conv4{}; Tensor bias_8_conv4{};
        Tensor Q_table_8_conv5{}; Tensor WT_buffer_8_conv5{}; Tensor bias_8_conv5{};
        Tensor out_8_conv3{}; Tensor out_8_conv4{}; Tensor out_8_conv5{};
        Tensor out_8_act3{};     Tensor out_8_act4{};     Tensor out_8_act5{};
        /*******c4-b4*************/
        Tensor Q_table_8_conv6{}; Tensor WT_buffer_8_conv6{}; Tensor bias_8_conv6{};
        Tensor out_8_conv6{}; 
        Tensor out_8_act6{};
        Tensor out_8_cat{};
        Tensor out_8_sf0{}, out_8_sf1{}, out_8_sf2{}, out_8_sf3{};Tensor out_8_fs0{};

        NEABMConvolutionLayer _8_conv0{}, _8_conv1{}, _8_conv2{}, _8_conv3{}, _8_conv4{}, _8_conv5{}, _8_conv6{};
        NEActivationLayer _8_act0{}, _8_act1{}, _8_act2{}, _8_act3{}, _8_act4{}, _8_act5{}, _8_act6{};
        NEConcatenateLayer _8_cat{};
        NES8toF32Layer _8_sf0{}, _8_sf1{}, _8_sf2{}, _8_sf3{}; NEF32toS8Layer _8_fs0{};
        /***********************d4=9******************/
        /********d4-b1************/
        Tensor Q_table_9_conv0{}; Tensor WT_buffer_9_conv0{}; Tensor bias_9_conv0{};
        Tensor out_9_conv0{};
        Tensor out_9_act0{};
        /*******d4-b2************/
        Tensor Q_table_9_conv1{}; Tensor WT_buffer_9_conv1{}; Tensor bias_9_conv1{};
        Tensor Q_table_9_conv2{}; Tensor WT_buffer_9_conv2{}; Tensor bias_9_conv2{};
        Tensor out_9_conv1{}; Tensor out_9_conv2{};
        Tensor out_9_act1{};     Tensor out_9_act2{};
        /*******d4-b3*************/
        Tensor Q_table_9_conv3{}; Tensor WT_buffer_9_conv3{}; Tensor bias_9_conv3{};
        Tensor Q_table_9_conv4{}; Tensor WT_buffer_9_conv4{}; Tensor bias_9_conv4{};
        Tensor Q_table_9_conv5{}; Tensor WT_buffer_9_conv5{}; Tensor bias_9_conv5{};
        Tensor out_9_conv3{}; Tensor out_9_conv4{}; Tensor out_9_conv5{};
        Tensor out_9_act3{};     Tensor out_9_act4{};     Tensor out_9_act5{};
        /*******d4-b4*************/
        Tensor Q_table_9_conv6{}; Tensor WT_buffer_9_conv6{}; Tensor bias_9_conv6{};
        Tensor out_9_conv6{}; 
        Tensor out_9_act6{};
        Tensor out_9_cat{};
        Tensor out_9_sf0{}, out_9_sf1{}, out_9_sf2{}, out_9_sf3{};Tensor out_9_fs0{};

        NEABMConvolutionLayer _9_conv0{}, _9_conv1{}, _9_conv2{}, _9_conv3{}, _9_conv4{}, _9_conv5{}, _9_conv6{};
        NEActivationLayer _9_act0{}, _9_act1{}, _9_act2{}, _9_act3{}, _9_act4{}, _9_act5{}, _9_act6{};
        NEConcatenateLayer _9_cat{};
        NES8toF32Layer _9_sf0{}, _9_sf1{}, _9_sf2{}, _9_sf3{}; NEF32toS8Layer _9_fs0{};
        /***********************e4=A******************/
        /********e4-b1************/
        Tensor Q_table_A_conv0{}; Tensor WT_buffer_A_conv0{}; Tensor bias_A_conv0{};
        Tensor out_A_conv0{};
        Tensor out_A_act0{};
        /*******e4-b2************/
        Tensor Q_table_A_conv1{}; Tensor WT_buffer_A_conv1{}; Tensor bias_A_conv1{};
        Tensor Q_table_A_conv2{}; Tensor WT_buffer_A_conv2{}; Tensor bias_A_conv2{};
        Tensor out_A_conv1{}; Tensor out_A_conv2{};
        Tensor out_A_act1{};     Tensor out_A_act2{};
        /*******e4-b3*************/
        Tensor Q_table_A_conv3{}; Tensor WT_buffer_A_conv3{}; Tensor bias_A_conv3{};
        Tensor Q_table_A_conv4{}; Tensor WT_buffer_A_conv4{}; Tensor bias_A_conv4{};
        Tensor Q_table_A_conv5{}; Tensor WT_buffer_A_conv5{}; Tensor bias_A_conv5{};
        Tensor out_A_conv3{}; Tensor out_A_conv4{}; Tensor out_A_conv5{};
        Tensor out_A_act3{};     Tensor out_A_act4{};     Tensor out_A_act5{};
        /*******e4-b4*************/
        Tensor Q_table_A_conv6{}; Tensor WT_buffer_A_conv6{}; Tensor bias_A_conv6{};
        Tensor out_A_conv6{}; 
        Tensor out_A_act6{};
        Tensor out_A_cat{};
        Tensor out_A_sf0{}, out_A_sf1{}, out_A_sf2{}, out_A_sf3{};Tensor out_A_fs0{};

        NEABMConvolutionLayer _A_conv0{}, _A_conv1{}, _A_conv2{}, _A_conv3{}, _A_conv4{}, _A_conv5{}, _A_conv6{};
        NEActivationLayer _A_act0{}, _A_act1{}, _A_act2{}, _A_act3{}, _A_act4{}, _A_act5{}, _A_act6{};
        NEConcatenateLayer _A_cat{};
        NES8toF32Layer _A_sf0{}, _A_sf1{}, _A_sf2{}, _A_sf3{}; NEF32toS8Layer _A_fs0{};
        /**********************maxpool*******************/
        Tensor out_B_pool0{};
        Tensor out_B_sf0{}; Tensor out_B_fs0{};
        NEPoolingLayer _B_pool0{};
        NES8toF32Layer _B_sf0{}; NEF32toS8Layer _B_fs0{};
        /***********************a5=C******************/
        /********a5-b1************/
        Tensor Q_table_C_conv0{}; Tensor WT_buffer_C_conv0{}; Tensor bias_C_conv0{};
        Tensor out_C_conv0{};
        Tensor out_C_act0{};
        /*******a5-b2************/
        Tensor Q_table_C_conv1{}; Tensor WT_buffer_C_conv1{}; Tensor bias_C_conv1{};
        Tensor Q_table_C_conv2{}; Tensor WT_buffer_C_conv2{}; Tensor bias_C_conv2{};
        Tensor out_C_conv1{}; Tensor out_C_conv2{};
        Tensor out_C_act1{};     Tensor out_C_act2{};
        /*******a5-b3*************/
        Tensor Q_table_C_conv3{}; Tensor WT_buffer_C_conv3{}; Tensor bias_C_conv3{};
        Tensor Q_table_C_conv4{}; Tensor WT_buffer_C_conv4{}; Tensor bias_C_conv4{};
        Tensor Q_table_C_conv5{}; Tensor WT_buffer_C_conv5{}; Tensor bias_C_conv5{};
        Tensor out_C_conv3{}; Tensor out_C_conv4{}; Tensor out_C_conv5{};
        Tensor out_C_act3{};     Tensor out_C_act4{};     Tensor out_C_act5{};
        /*******a5-b4*************/
        Tensor Q_table_C_conv6{}; Tensor WT_buffer_C_conv6{}; Tensor bias_C_conv6{};
        Tensor out_C_conv6{}; 
        Tensor out_C_act6{};
        Tensor out_C_cat{};
        Tensor out_C_sf0{}, out_C_sf1{}, out_C_sf2{}, out_C_sf3{};Tensor out_C_fs0{};

        NEABMConvolutionLayer _C_conv0{}, _C_conv1{}, _C_conv2{}, _C_conv3{}, _C_conv4{}, _C_conv5{}, _C_conv6{};
        NEActivationLayer _C_act0{}, _C_act1{}, _C_act2{}, _C_act3{}, _C_act4{}, _C_act5{}, _C_act6{};
        NEConcatenateLayer _C_cat{};
        NES8toF32Layer _C_sf0{}, _C_sf1{}, _C_sf2{}, _C_sf3{}; NEF32toS8Layer _C_fs0{};
        /***********************b5=D******************/
        /********b5-b1************/
        Tensor Q_table_D_conv0{}; Tensor WT_buffer_D_conv0{}; Tensor bias_D_conv0{};
        Tensor out_D_conv0{};
        Tensor out_D_act0{};
        /*******b5-b2************/
        Tensor Q_table_D_conv1{}; Tensor WT_buffer_D_conv1{}; Tensor bias_D_conv1{};
        Tensor Q_table_D_conv2{}; Tensor WT_buffer_D_conv2{}; Tensor bias_D_conv2{};
        Tensor out_D_conv1{}; Tensor out_D_conv2{};
        Tensor out_D_act1{};     Tensor out_D_act2{};
        /*******b5-b3*************/
        Tensor Q_table_D_conv3{}; Tensor WT_buffer_D_conv3{}; Tensor bias_D_conv3{};
        Tensor Q_table_D_conv4{}; Tensor WT_buffer_D_conv4{}; Tensor bias_D_conv4{};
        Tensor Q_table_D_conv5{}; Tensor WT_buffer_D_conv5{}; Tensor bias_D_conv5{};
        Tensor out_D_conv3{}; Tensor out_D_conv4{}; Tensor out_D_conv5{};
        Tensor out_D_act3{};     Tensor out_D_act4{};     Tensor out_D_act5{};
        /*******b5-b4*************/
        Tensor Q_table_D_conv6{}; Tensor WT_buffer_D_conv6{}; Tensor bias_D_conv6{};
        Tensor out_D_conv6{}; 
        Tensor out_D_act6{};
        Tensor out_D_cat{};
        Tensor out_D_sf0{}, out_D_sf1{}, out_D_sf2{}, out_D_sf3{};Tensor out_D_fs0{};

        NEABMConvolutionLayer _D_conv0{}, _D_conv1{}, _D_conv2{}, _D_conv3{}, _D_conv4{}, _D_conv5{}, _D_conv6{};
        NEActivationLayer _D_act0{}, _D_act1{}, _D_act2{}, _D_act3{}, _D_act4{}, _D_act5{}, _D_act6{};
        NEConcatenateLayer _D_cat{};
        NES8toF32Layer _D_sf0{}, _D_sf1{}, _D_sf2{}, _D_sf3{}; NEF32toS8Layer _D_fs0{};
        /***********************avgpool******************/
        Tensor out_E_pool0{};
        Tensor out_E_sf0{}; Tensor out_E_fs0{};

        NEPoolingLayer _E_pool0{};
        NES8toF32Layer _E_sf0{}; NEF32toS8Layer _E_fs0{};
        /**********************fc**************************/
        Tensor Q_table_F_conv0{}; Tensor WT_buffer_F_conv0{}; Tensor bias_F_conv0{};
        Tensor out_F_conv0{};
        NEABMConvolutionLayer _F_conv0{};

};
int main(int argc, char **argv)
{
	return utils::run_example<NEONGOOGLENETExample>(argc, argv);
}
