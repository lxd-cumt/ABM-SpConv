/*Support GroupConvolution*/
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/CLFunctions.h"
#include "arm_compute/runtime/NEON/NEFunctions.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "arm_compute/runtime/Allocator.h"
#include "utils/ImageLoader.h"
#include "utils/Utils.h"
#include "utils/GraphUtils.h"
#include <ctime>
#include <cstdlib>

using namespace arm_compute;
using namespace utils;

class NEALEXNETExample : public Example
{
public:
bool do_setup(int argc, char **argv) override
{
/*/////////////////////////////////////////////////////////////////////////*/
/*    validate_file_name=argv[1];
    file_path=argv[2];
    start=strtoul(argv[3], NULL, 10)-1;
    end=strtoul(argv[4], NULL, 10)-1;
*/
    /*arm_compute::graph_utils::ImageAccessor load_input;*/
    
    /*/////////////////////////////////////////////////////////////////////////////////////////////////////////// Tensor Init/////////////////////////////////////////////////////////////////*/
    /*string data_path="/media/sdcard/ComputeLibrary/data/neon_alexnet/";*/
    /*NPYLoader npy_input;npy_input.open(data_path+"input_s8.npy");npy_input.init_tensor2(src,DataType:: S8);*/
    TensorShape src_shape(227,227,3);
    src.allocator()->init(TensorInfo(src_shape, 1, DataType::F32));
    TensorShape input_shape(227,227,3);
    input.allocator()->init(TensorInfo(input_shape, 1, DataType::S8));
    /*c-a-p*/
    NPYLoader npy0_q;npy0_q.open(Q_table_datapath+Q_table_name[0]);npy0_q.init_tensor2(Q_table_conv0,DataType:: S16);
    NPYLoader npy0_wt;npy0_wt.open(WT_buffer_datapath+WT_buffer_name[0]);npy0_wt.init_tensor2(WT_buffer_conv0,DataType::U16);
    NPYLoader npy0_b;npy0_b.open(bias_datapath+bias_name[0]);npy0_b.init_tensor2(bias_conv0,DataType:: S8);
    WeightsInfo weights_conv0(false,11,11,96,false);
    const TensorShape out_shape_conv0(55, 55, 96);
    out_conv0.allocator()->init(TensorInfo(out_shape_conv0, 1, DataType:: S8));
    out_act0.allocator()->init(TensorInfo(out_shape_conv0,1,DataType:: F32));
    out_norm0.allocator()->init(TensorInfo(out_shape_conv0,1,DataType:: F32));
    TensorShape out_shape_pool0 = out_shape_conv0;
    out_shape_pool0.set(0, out_shape_pool0.x() / 2);
    out_shape_pool0.set(1, out_shape_pool0.y() / 2);
    out_pool0.allocator()->init(TensorInfo(out_shape_pool0, 1, DataType:: F32));

    /*c-a-p*/
    NPYLoader npy1_q;npy1_q.open(Q_table_datapath+Q_table_name[1]);npy1_q.init_tensor2(Q_table_conv1,DataType:: S16);
    NPYLoader npy1_wt;npy1_wt.open(WT_buffer_datapath+WT_buffer_name[1]);npy1_wt.init_tensor2(WT_buffer_conv1,DataType::U16);
    NPYLoader npy1_b;npy1_b.open(bias_datapath+bias_name[1]);npy1_b.init_tensor2(bias_conv1,DataType:: S8);
    WeightsInfo weights_conv1(false,5,5,256,false);
    const TensorShape out_shape_conv1(27, 27, 256);
    out_conv1.allocator()->init(TensorInfo(out_shape_conv1, 1, DataType:: S8));
    out_act1.allocator()->init(TensorInfo(out_shape_conv1,1,DataType:: F32));
    out_norm1.allocator()->init(TensorInfo(out_shape_conv1,1,DataType:: F32));
    TensorShape out_shape_pool1 = out_shape_conv1;
    out_shape_pool1.set(0, out_shape_pool1.x() / 2);
    out_shape_pool1.set(1, out_shape_pool1.y() / 2);
    out_pool1.allocator()->init(TensorInfo(out_shape_pool1, 1, DataType:: F32));

    /*c-a*/
    NPYLoader npy2_q;npy2_q.open(Q_table_datapath+Q_table_name[2]);npy2_q.init_tensor2(Q_table_conv2,DataType:: S16);
    NPYLoader npy2_wt;npy2_wt.open(WT_buffer_datapath+WT_buffer_name[2]);npy2_wt.init_tensor2(WT_buffer_conv2,DataType::U16);
    NPYLoader npy2_b;npy2_b.open(bias_datapath+bias_name[2]);npy2_b.init_tensor2(bias_conv2,DataType:: S8);
    WeightsInfo weights_conv2(false,3,3,384,false);
    const TensorShape out_shape_conv2(13, 13, 384);
    out_conv2.allocator()->init(TensorInfo(out_shape_conv2, 1, DataType:: S8));
    out_act2.allocator()->init(TensorInfo(out_shape_conv2,1,DataType:: F32));

    /*c-a*/
    NPYLoader npy3_q;npy3_q.open(Q_table_datapath+Q_table_name[3]);npy3_q.init_tensor2(Q_table_conv3,DataType:: S16);
    NPYLoader npy3_wt;npy3_wt.open(WT_buffer_datapath+WT_buffer_name[3]);npy3_wt.init_tensor2(WT_buffer_conv3,DataType::U16);
    NPYLoader npy3_b;npy3_b.open(bias_datapath+bias_name[3]);npy3_b.init_tensor2(bias_conv3,DataType:: S8);
    WeightsInfo weights_conv3(false,3,3,384,false);
    const TensorShape out_shape_conv3(13, 13, 384);
    out_conv3.allocator()->init(TensorInfo(out_shape_conv3, 1, DataType:: S8));
    out_act3.allocator()->init(TensorInfo(out_shape_conv3,1,DataType:: F32));

    /*c-a-p*/
    NPYLoader npy4_q;npy4_q.open(Q_table_datapath+Q_table_name[4]);npy4_q.init_tensor2(Q_table_conv4,DataType:: S16);
    NPYLoader npy4_wt;npy4_wt.open(WT_buffer_datapath+WT_buffer_name[4]);npy4_wt.init_tensor2(WT_buffer_conv4,DataType::U16);
    NPYLoader npy4_b;npy4_b.open(bias_datapath+bias_name[4]);npy4_b.init_tensor2(bias_conv4,DataType:: S8);
    WeightsInfo weights_conv4(false,3,3,256,false);
    const TensorShape out_shape_conv4(13, 13, 256);
    out_conv4.allocator()->init(TensorInfo(out_shape_conv4, 1, DataType:: S8));
    out_act4.allocator()->init(TensorInfo(out_shape_conv4,1,DataType:: F32));
    TensorShape out_shape_pool4 = out_shape_conv4;
    out_shape_pool4.set(0, out_shape_pool4.x() / 2);
    out_shape_pool4.set(1, out_shape_pool4.y() / 2);
    out_pool4.allocator()->init(TensorInfo(out_shape_pool4, 1, DataType:: F32));

    /*fc-a*/
    NPYLoader npy5_q;npy5_q.open(Q_table_datapath+Q_table_name[5]);npy5_q.init_tensor2(Q_table_conv5,DataType:: S16);
    NPYLoader npy5_wt;npy5_wt.open(WT_buffer_datapath+WT_buffer_name[5]);npy5_wt.init_tensor2(WT_buffer_conv5,DataType::U16);
    NPYLoader npy5_b;npy5_b.open(bias_datapath+bias_name[5]);npy5_b.init_tensor2(bias_conv5,DataType:: S8);
    WeightsInfo weights_conv5(false,6,6,4096,false);
    const TensorShape out_shape_conv5(1, 1, 4096);
    out_conv5.allocator()->init(TensorInfo(out_shape_conv5, 1, DataType:: S8));
    out_act5.allocator()->init(TensorInfo(out_shape_conv5,1,DataType:: F32));

    /*fc-a*/
    NPYLoader npy6_q;npy6_q.open(Q_table_datapath+Q_table_name[6]);npy6_q.init_tensor2(Q_table_conv6,DataType:: S16);
    NPYLoader npy6_wt;npy6_wt.open(WT_buffer_datapath+WT_buffer_name[6]);npy6_wt.init_tensor2(WT_buffer_conv6,DataType::U16);
    NPYLoader npy6_b;npy6_b.open(bias_datapath+bias_name[6]);npy6_b.init_tensor2(bias_conv6,DataType:: S8);
    WeightsInfo weights_conv6(false,1,1,4096,false);
    const TensorShape out_shape_conv6(1, 1, 4096);
    out_conv6.allocator()->init(TensorInfo(out_shape_conv6, 1, DataType:: S8));
    out_act6.allocator()->init(TensorInfo(out_shape_conv6,1,DataType:: F32));

    /*fc*/
    NPYLoader npy7_q;npy7_q.open(Q_table_datapath+Q_table_name[7]);npy7_q.init_tensor2(Q_table_conv7,DataType:: S16);
    NPYLoader npy7_wt;npy7_wt.open(WT_buffer_datapath+WT_buffer_name[7]);npy7_wt.init_tensor2(WT_buffer_conv7,DataType::U16);
    NPYLoader npy7_b;npy7_b.open(bias_datapath+bias_name[7]);npy7_b.init_tensor2(bias_conv7,DataType:: S8);
    WeightsInfo weights_conv7(false,1,1,1000,false);
    const TensorShape out_shape_conv7(1, 1, 1000);
    out_conv7.allocator()->init(TensorInfo(out_shape_conv7, 1, DataType:: S8));
    /*const TensorShape out_shape_class(out_shape_conv7.x()*out_shape_conv7.y()*out_shape_conv7.z());
    out_class.allocator()->init(TensorInfo(out_shape_class, 1, DataType::F32));*/
    /*out_softmax.allocator()->init(TensorInfo(out_shape_conv7, 1, DataType:: F32));*/

    /*///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// Layer Cofigure////////////////////////////////////////////////////////////////*/
    /*c-a-p*/
    input_convert.configure(&src, &input);
    conv0.configure(&input, &Q_table_conv0, &WT_buffer_conv0, &bias_conv0, &out_conv0, PadStrideInfo(4,4,0,0), weights_conv0, precision[0],1);
    lsfconv0.configure(&out_conv0, &sfconv0);
    act0.configure(&sfconv0, &out_act0, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
    norm0.configure(&out_act0, &out_norm0, NormalizationLayerInfo(NormType::CROSS_MAP, 5, 0.0001f, 0.75f));
    pool0.configure(&out_norm0, &out_pool0, PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 0)));
    lfspool0.configure(&out_pool0, &fspool0);

    /*c-a-p*/
    conv1.configure(&fspool0, &Q_table_conv1, &WT_buffer_conv1, &bias_conv1, &out_conv1, PadStrideInfo(1,1,2,2), weights_conv1, precision[1],2);
    lsfconv1.configure(&out_conv1, &sfconv1);
    act1.configure(&sfconv1, &out_act1, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
    norm1.configure(&out_act1, &out_norm1, NormalizationLayerInfo(NormType::CROSS_MAP, 5, 0.0001f, 0.75f));
    pool1.configure(&out_norm1, &out_pool1, PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 0)));
    lfspool1.configure(&out_pool1, &fspool1);

    /*c-a*/
    conv2.configure(&fspool1, &Q_table_conv2, &WT_buffer_conv2, &bias_conv2, &out_conv2, PadStrideInfo(1,1,1,1), weights_conv2, precision[2],1);
    lsfconv2.configure(&out_conv2, &sfconv2);
    act2.configure(&sfconv2, &out_act2, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
    lfsact2.configure(&out_act2, &fsact2);

    /*c-a*/
    conv3.configure(&fsact2, &Q_table_conv3, &WT_buffer_conv3, &bias_conv3, &out_conv3, PadStrideInfo(1,1,1,1), weights_conv3, precision[3],2);
    lsfconv3.configure(&out_conv3, &sfconv3);
    act3.configure(&sfconv3, &out_act3, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
    lfsact3.configure(&out_act3, &fsact3);    

    /*c-a-p*/
    conv4.configure(&fsact3, &Q_table_conv4, &WT_buffer_conv4, &bias_conv4, &out_conv4, PadStrideInfo(1,1,1,1), weights_conv4, precision[4],2);
    lsfconv4.configure(&out_conv4, &sfconv4);
    act4.configure(&sfconv4, &out_act4, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
    pool4.configure(&out_act4, &out_pool4, PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 0)));
    lfspool4.configure(&out_pool4, &fspool4);

    /*fc-a*/
    conv5.configure(&fspool4, &Q_table_conv5, &WT_buffer_conv5, &bias_conv5, &out_conv5, PadStrideInfo(1,1,0,0), weights_conv5, precision[5],1);
    lsfconv5.configure(&out_conv5, &sfconv5);
    act5.configure(&sfconv5, &out_act5, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
    lfsact5.configure(&out_act5, &fsact5);

    /*fc-a*/
    conv6.configure(&fsact5, &Q_table_conv6, &WT_buffer_conv6, &bias_conv6, &out_conv6, PadStrideInfo(1,1,0,0), weights_conv6, precision[6],1);
    lsfconv6.configure(&out_conv6, &sfconv6);
    act6.configure(&sfconv6, &out_act6, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
    lfsact6.configure(&out_act6, &fsact6);

    /*fc*/
    conv7.configure(&fsact6, &Q_table_conv7, &WT_buffer_conv7, &bias_conv7, &out_conv7, PadStrideInfo(1,1,0,0), weights_conv7, precision[7],1);
    lsfconv7.configure(&out_conv7, &sfconv7);
   /* softmax.configure(&sfconv7, &out_softmax);*/
    /*/////////////////////////////////////////////////////////////////////////////////////////////////////// Allocate//////////////////////////////////////////////////////////////////*/
    src.allocator()->allocate();input.allocator()->allocate();
    Q_table_conv0.allocator()->allocate(); WT_buffer_conv0.allocator()->allocate(); bias_conv0.allocator()->allocate();
    Q_table_conv1.allocator()->allocate(); WT_buffer_conv1.allocator()->allocate(); bias_conv1.allocator()->allocate();
    Q_table_conv2.allocator()->allocate(); WT_buffer_conv2.allocator()->allocate(); bias_conv2.allocator()->allocate();
    Q_table_conv3.allocator()->allocate(); WT_buffer_conv3.allocator()->allocate(); bias_conv3.allocator()->allocate();
    Q_table_conv4.allocator()->allocate(); WT_buffer_conv4.allocator()->allocate(); bias_conv4.allocator()->allocate();
    /*fully connected layer*/
    Q_table_conv5.allocator()->allocate(); WT_buffer_conv5.allocator()->allocate(); bias_conv5.allocator()->allocate();
    Q_table_conv6.allocator()->allocate(); WT_buffer_conv6.allocator()->allocate(); bias_conv6.allocator()->allocate();
    Q_table_conv7.allocator()->allocate(); WT_buffer_conv7.allocator()->allocate(); bias_conv7.allocator()->allocate();

    out_conv0.allocator()->allocate(); out_act0.allocator()->allocate(); out_norm0.allocator()->allocate(); out_pool0.allocator()->allocate();
    out_conv1.allocator()->allocate(); out_act1.allocator()->allocate(); out_norm1.allocator()->allocate(); out_pool1.allocator()->allocate();
    out_conv2.allocator()->allocate(); out_act2.allocator()->allocate();
    out_conv3.allocator()->allocate(); out_act3.allocator()->allocate();
    out_conv4.allocator()->allocate(); out_act4.allocator()->allocate(); out_pool4.allocator()->allocate();
    out_conv5.allocator()->allocate(); out_act5.allocator()->allocate();
    out_conv6.allocator()->allocate(); out_act6.allocator()->allocate();
    out_conv7.allocator()->allocate();
    /*out_class.allocator()->allocate();
    out_softmax.allocator()->allocate();*/

    /*type change tensor*/
    sfconv0.allocator()->allocate(); fspool0.allocator()->allocate();
    sfconv1.allocator()->allocate(); fspool1.allocator()->allocate();
    sfconv2.allocator()->allocate(); fsact2.allocator()->allocate();
    sfconv3.allocator()->allocate(); fsact3.allocator()->allocate();
    sfconv4.allocator()->allocate(); fspool4.allocator()->allocate();
    sfconv5.allocator()->allocate(); fsact5.allocator()->allocate();
    sfconv6.allocator()->allocate(); fsact6.allocator()->allocate();
    sfconv7.allocator()->allocate();

    /*npy_input.fill_tensor2(src);*/
    npy0_q.fill_tensor2(Q_table_conv0);
    npy1_q.fill_tensor2(Q_table_conv1);
    npy2_q.fill_tensor2(Q_table_conv2);
    npy3_q.fill_tensor2(Q_table_conv3);
    npy4_q.fill_tensor2(Q_table_conv4);
    npy5_q.fill_tensor2(Q_table_conv5);
    npy6_q.fill_tensor2(Q_table_conv6);
    npy7_q.fill_tensor2(Q_table_conv7);
    npy0_wt.fill_tensor2(WT_buffer_conv0);
    npy1_wt.fill_tensor2(WT_buffer_conv1);
    npy2_wt.fill_tensor2(WT_buffer_conv2);
    npy3_wt.fill_tensor2(WT_buffer_conv3);
    npy4_wt.fill_tensor2(WT_buffer_conv4);
    npy5_wt.fill_tensor2(WT_buffer_conv5);
    npy6_wt.fill_tensor2(WT_buffer_conv6);
    npy7_wt.fill_tensor2(WT_buffer_conv7);
    npy0_b.fill_tensor2(bias_conv0);
    npy1_b.fill_tensor2(bias_conv1);
    npy2_b.fill_tensor2(bias_conv2);
    npy3_b.fill_tensor2(bias_conv3);
    npy4_b.fill_tensor2(bias_conv4);
    npy5_b.fill_tensor2(bias_conv5);
    npy6_b.fill_tensor2(bias_conv6);
    npy7_b.fill_tensor2(bias_conv7);

    is_fortran=npy0_b.is_fortran();
    return true;
}
void do_run() override
{

    string validate_file_name{};
    string file_path{};
    unsigned int start=0;
    unsigned int end=0;
    string image_filename{};
    string image_labels{};
    unsigned int function=1;
    /*string functions{};*/
    std::cout<<"1 Validate Datasets or 2 Image Classification"<<std::endl;
    std::cin>>function;
    if(function==1)
    {
        /*std::cout<<"1 validation-labels:"<<std::endl;*/
        /*std::cin>>validate_file_name;
        validate_file_name="/media/sdcard/ComputeLibrary/data/neon_alexnet/"+validate_file_name;*/
        validate_file_name="/media/sdcard/ComputeLibrary/data/neon_alexnet/val_labels.txt";
        /*std::cout<<"2 validation-path:"<<std::endl;
        std::cin>>file_path;file_path="/media/sdcard/ComputeLibrary/data/neon_alexnet/"+file_path;*/
        file_path="/media/sdcard/ComputeLibrary/data/neon_alexnet/input_images/";
        std::cout<<"1 validation-range start"<<std::endl;
        std::cin>>start;start=start-1;
        std::cout<<"2 validation-range end"<<std::endl;
        std::cin>>end;end=end-1;
    }
    else if(function==2)
    {
        /*std::cout<<"1 image-labels:"<<std::endl;
        std::cin>>image_labels;image_labels="/media/sdcard/ComputeLibrary/data/neon_alexnet/"+image_labels;*/
        image_labels="/media/sdcard/ComputeLibrary/data/neon_alexnet/labels.txt";
        std::cout<<"1 image-filename:"<<std::endl;
        std::cin>>image_filename;image_filename="/media/sdcard/ComputeLibrary/data/neon_alexnet/"+image_filename;                
    }
    else{
        std::cout<<"Wrong Instruction!"<<std::endl;
    }


    const std::array<float, 3> mean_rgb{ { 122.68f, 116.67f, 104.01f } };
    std::unique_ptr<arm_compute::graph_utils::IPreprocessor> preprocessor = arm_compute::support::cpp14::make_unique<arm_compute::graph_utils::CaffePreproccessor>(mean_rgb);
    std::unique_ptr<arm_compute::graph_utils::IPreprocessor> preprocessor2 = arm_compute::support::cpp14::make_unique<arm_compute::graph_utils::CaffePreproccessor>(mean_rgb);
    arm_compute::graph_utils::ValidationInputAccessor load_images(validate_file_name,file_path,std::move(preprocessor),false,start,end);
    arm_compute::graph_utils::ValidationOutputAccessor test_output(validate_file_name,std::cout,start,end);
  /*  string image_filename="/media/sdcard/ComputeLibrary/data/neon_alexnet/gondola.jpeg";
    string image_labels="/media/sdcard/ComputeLibrary/data/neon_alexnet/labels.txt";*/
    arm_compute::graph_utils::ImageAccessor image_predict_input(image_filename, false, std::move(preprocessor2));
    arm_compute::graph_utils::TopNPredictionsAccessor image_predict_output(image_labels, 5);
    if(function==1){
        double act_layer=0;
        double total=0;
        double end_load_images=0;
        double end00, end01, end02=0, end03=0;
        double end10, end11, end12=0, end13=0;
        double end20, end21=0;
        double end30, end31=0;
        double end40, end41, end42=0;
        double end50, end51=0;
        double end60, end61=0;
        double end70=0;
        double end_output_prediction=0;

        std::string base_path = "/media/sdcard/ComputeLibrary";
        std::string output_file_path = "/model.csv";
        ofstream out(base_path+output_file_path, ios::out | ios::app);
        out<<"AlexNet ABM"<<std::endl;
        int cycles = end + 1 - start;
        for(unsigned int i=start; i<end+1; i++)
        {
            double time=0;
            double one_runtime=0;
            auto tbegin = std::chrono::high_resolution_clock::now();

            load_images.access_tensor(src);auto e_load_images = std::chrono::high_resolution_clock::now();
            input_convert.run();auto e_input_convert = std::chrono::high_resolution_clock::now();
            conv0.run();auto e_conv0 = std::chrono::high_resolution_clock::now();
            lsfconv0.run();auto e_sfconv0 = std::chrono::high_resolution_clock::now();
            act0.run(); auto e_act0 = std::chrono::high_resolution_clock::now();
            norm0.run(); auto e_norm0 = std::chrono::high_resolution_clock::now();
            pool0.run();auto e_pool0 = std::chrono::high_resolution_clock::now();
            lfspool0.run();auto e_fspool0 = std::chrono::high_resolution_clock::now();

            conv1.run();auto e_conv1 = std::chrono::high_resolution_clock::now();
            lsfconv1.run();auto e_sfconv1 = std::chrono::high_resolution_clock::now();
            act1.run(); auto e_act1 = std::chrono::high_resolution_clock::now();
            norm1.run(); auto e_norm1 = std::chrono::high_resolution_clock::now();
            pool1.run();auto e_pool1 = std::chrono::high_resolution_clock::now();
            lfspool1.run();auto e_fspool1 = std::chrono::high_resolution_clock::now();

            conv2.run();auto e_conv2 = std::chrono::high_resolution_clock::now();
            lsfconv2.run();auto e_sfconv2 = std::chrono::high_resolution_clock::now();
            act2.run();auto e_act2 = std::chrono::high_resolution_clock::now();
            lfsact2.run();auto e_fsact2 = std::chrono::high_resolution_clock::now();

            conv3.run();auto e_conv3 = std::chrono::high_resolution_clock::now();
            lsfconv3.run();auto e_sfconv3 = std::chrono::high_resolution_clock::now();
            act3.run();auto e_act3 = std::chrono::high_resolution_clock::now();
            lfsact3.run();auto e_fsact3 = std::chrono::high_resolution_clock::now();

            conv4.run();auto e_conv4 = std::chrono::high_resolution_clock::now();
            lsfconv4.run();auto e_sfconv4 = std::chrono::high_resolution_clock::now();
            act4.run();auto e_act4 = std::chrono::high_resolution_clock::now();
            pool4.run();auto e_pool4 = std::chrono::high_resolution_clock::now();
            lfspool4.run();auto e_fspool4 = std::chrono::high_resolution_clock::now();

            conv5.run();auto e_conv5 = std::chrono::high_resolution_clock::now();
            lsfconv5.run();auto e_sfconv5 = std::chrono::high_resolution_clock::now();
            act5.run();auto e_act5 = std::chrono::high_resolution_clock::now();
            lfsact5.run();auto e_fsact5 = std::chrono::high_resolution_clock::now();

            conv6.run();auto e_conv6 = std::chrono::high_resolution_clock::now();
            lsfconv6.run();auto e_sfconv6 = std::chrono::high_resolution_clock::now();
            act6.run();auto e_act6 = std::chrono::high_resolution_clock::now();
            lfsact6.run();auto e_fsact6 = std::chrono::high_resolution_clock::now();

            conv7.run();auto e_conv7 = std::chrono::high_resolution_clock::now();
            lsfconv7.run();auto e_sfconv7 = std::chrono::high_resolution_clock::now();

            test_output.access_tensor(sfconv7);auto e_output_prediction = std::chrono::high_resolution_clock::now();
            // auto tend = std::chrono::high_resolution_clock::now();

            /*each layer's running time*/
            if(i>start){
                time = std::chrono::duration_cast<std::chrono::duration<double>>(e_load_images - tbegin).count();end_load_images+=time; 

                time = std::chrono::duration_cast<std::chrono::duration<double>>(e_conv0 - e_input_convert).count();end00+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(e_act0 - e_sfconv0).count();end01+=time; one_runtime+=time; act_layer+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(e_norm0 - e_act0).count();end03+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(e_pool0 - e_norm0).count();end02+=time; one_runtime+=time;

                time = std::chrono::duration_cast<std::chrono::duration<double>>(e_conv1 - e_fspool0).count();end10+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(e_act1 - e_sfconv1).count();end11+=time; one_runtime+=time; act_layer+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(e_norm1 - e_act1).count();end13+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(e_pool1 - e_norm1).count();end12+=time; one_runtime+=time;

                time = std::chrono::duration_cast<std::chrono::duration<double>>(e_conv2 - e_fspool1).count();end20+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(e_act2 - e_sfconv2).count();end21+=time; one_runtime+=time; act_layer+=time;

                time = std::chrono::duration_cast<std::chrono::duration<double>>(e_conv3 - e_fsact2).count();end30+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(e_act3 - e_sfconv3).count();end31+=time; one_runtime+=time; act_layer+=time;

                time = std::chrono::duration_cast<std::chrono::duration<double>>(e_conv4 - e_fsact3).count();end40+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(e_act4 - e_sfconv4).count();end41+=time; one_runtime+=time; act_layer+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(e_pool4 - e_act4).count();end42+=time; one_runtime+=time;

                time = std::chrono::duration_cast<std::chrono::duration<double>>(e_conv5 - e_fspool4).count();end50+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(e_act5 - e_sfconv5).count();end51+=time; one_runtime+=time; act_layer+=time;

                time = std::chrono::duration_cast<std::chrono::duration<double>>(e_conv6 - e_fsact5).count();end60+=time; one_runtime+=time;
                time = std::chrono::duration_cast<std::chrono::duration<double>>(e_act6 - e_sfconv6).count();end61+=time; one_runtime+=time; act_layer+=time;

                time = std::chrono::duration_cast<std::chrono::duration<double>>(e_conv7 - e_fsact6).count();end70+=time; one_runtime+=time;

                time = std::chrono::duration_cast<std::chrono::duration<double>>(e_output_prediction - e_sfconv7).count();end_output_prediction+=time; 

                total=total+one_runtime;

                out<<"one run time"<<","<<one_runtime*1000<<std::endl;
            }
            /*total=total+one_runtime;*/
        }

        

        out<<"load_input_images"<<","<<end_load_images*1000/(cycles-1)<<std::endl;

        out<<"conv0"<<","<<end00*1000/(cycles-1)<<std::endl;
        out<<"act0"<<","<<end01*1000/(cycles-1)<<std::endl;
        out<<"norm0"<<","<<end03*1000/(cycles-1)<<std::endl;
        out<<"pool0"<<","<<end02*1000/(cycles-1)<<std::endl;

        out<<"conv1"<<","<<end10*1000/(cycles-1)<<std::endl;
        out<<"act1"<<","<<end11*1000/(cycles-1)<<std::endl;
        out<<"norm1"<<","<<end13*1000/(cycles-1)<<std::endl;
        out<<"pool1"<<","<<end12*1000/(cycles-1)<<std::endl;

        out<<"conv2"<<","<<end20*1000/(cycles-1)<<std::endl;
        out<<"act2"<<","<<end21*1000/(cycles-1)<<std::endl;

        out<<"conv3"<<","<<end30*1000/(cycles-1)<<std::endl;
        out<<"act3"<<","<<end31*1000/(cycles-1)<<std::endl;

        out<<"conv4"<<","<<end40*1000/(cycles-1)<<std::endl;
        out<<"act4"<<","<<end41*1000/(cycles-1)<<std::endl;
        out<<"pool4"<<","<<end42*1000/(cycles-1)<<std::endl;

        out<<"conv5"<<","<<end50*1000/(cycles-1)<<std::endl;
        out<<"act5"<<","<<end51*1000/(cycles-1)<<std::endl;
        
        out<<"conv6"<<","<<end60*1000/(cycles-1)<<std::endl;
        out<<"act6"<<","<<end61*1000/(cycles-1)<<std::endl;

        out<<"conv7"<<","<<end70*1000/(cycles-1)<<std::endl;

        out<<"output_prediction"<<","<<end_output_prediction*1000/(cycles-1)<<std::endl;

        out<<"avg_time"<<","<<total*1000/(cycles-1)<<std::endl;
        out<<"act_time"<<","<<act_layer*1000/(cycles-1)<<std::endl;

        std::cout<<"load_input_images"<<"       "<<end_load_images*1000/(cycles-1)<<"ms"<<std::endl;

        std::cout<<"conv0"<<"       "<<end00*1000/(cycles-1)<<"ms"<<std::endl;
        std::cout<<"act0"<<"       "<<end01*1000/(cycles-1)<<"ms"<<std::endl;
        std::cout<<"norm0"<<"       "<<end03*1000/(cycles-1)<<"ms"<<std::endl;
        std::cout<<"pool0"<<"       "<<end02*1000/(cycles-1)<<"ms"<<std::endl;

        std::cout<<"conv1"<<"       "<<end10*1000/(cycles-1)<<"ms"<<std::endl;
        std::cout<<"act1"<<"       "<<end11*1000/(cycles-1)<<"ms"<<std::endl;
        std::cout<<"norm1"<<"       "<<end13*1000/(cycles-1)<<"ms"<<std::endl;
        std::cout<<"pool1"<<"       "<<end12*1000/(cycles-1)<<"ms"<<std::endl;

        std::cout<<"conv2"<<"       "<<end20*1000/(cycles-1)<<"ms"<<std::endl;
        std::cout<<"act2"<<"       "<<end21*1000/(cycles-1)<<"ms"<<std::endl;

        std::cout<<"conv3"<<"       "<<end30*1000/(cycles-1)<<"ms"<<std::endl;
        std::cout<<"act3"<<"       "<<end31*1000/(cycles-1)<<"ms"<<std::endl;

        std::cout<<"conv4"<<"       "<<end40*1000/(cycles-1)<<"ms"<<std::endl;
        std::cout<<"act4"<<"       "<<end41*1000/(cycles-1)<<"ms"<<std::endl;
        std::cout<<"pool4"<<"       "<<end42*1000/(cycles-1)<<"ms"<<std::endl;

        std::cout<<"conv5"<<"       "<<end50*1000/(cycles-1)<<"ms"<<std::endl;
        std::cout<<"act5"<<"       "<<end51*1000/(cycles-1)<<"ms"<<std::endl;
        
        std::cout<<"conv6"<<"       "<<end60*1000/(cycles-1)<<"ms"<<std::endl;
        std::cout<<"act6"<<"       "<<end61*1000/(cycles-1)<<"ms"<<std::endl;

        std::cout<<"conv7"<<"       "<<end70*1000/(cycles-1)<<"ms"<<std::endl;

        std::cout<<"output_prediction"<<"       "<<end_output_prediction*1000/(cycles-1)<<"ms"<<std::endl;

        std::cout<<"avg_time"<<"       "<<total*1000/(cycles-1)<<"ms"<<std::endl;
        std::cout<<"act_time"<<"       "<<act_layer*1000/(cycles-1)<<"ms"<<std::endl;
    }
    else if(function==2)
    {
        image_predict_input.access_tensor(src);
        input_convert.run();
        conv0.run();lsfconv0.run();act0.run(); norm0.run(); pool0.run();lfspool0.run();
        conv1.run();lsfconv1.run();act1.run(); norm1.run(); pool1.run();lfspool1.run();
        conv2.run();lsfconv2.run();act2.run();lfsact2.run();
        conv3.run();lsfconv3.run();act3.run();lfsact3.run();
        conv4.run();lsfconv4.run();act4.run();pool4.run();lfspool4.run();
        conv5.run();lsfconv5.run();act5.run();lfsact5.run();
        conv6.run();lsfconv6.run();act6.run();lfsact6.run();
        conv7.run();lsfconv7.run();
        image_predict_output.access_tensor(sfconv7);
    }
    /*memcpy(out_class.buffer(), sfconv7.buffer(), 4*num_class);
    softmax.run();*/
    /*save_to_npy(sfconv7, output_filename, false);*/
    NPYLoader save;
    save.save_to_npy2(input, "/media/sdcard/ComputeLibrary/data/neon_alexnet/output_s8.npy",false);
    save_to_npy(src, "/media/sdcard/ComputeLibrary/data/neon_alexnet/output.npy",false);
}
private:
    unsigned int precision[8][4]={
        {7,5,5,2},
        {6,6,2,0},
        {7,9,0,0},
        {8,8,0,1},
        {8,4,1,1},
        {11,11,1,4},
        {11,11,4,5},
        {9,8,5,2}};
    string Q_table_datapath="/media/sdcard/ComputeLibrary/data/neon_alexnet/Q_table/";
    string WT_buffer_datapath="/media/sdcard/ComputeLibrary/data/neon_alexnet/WT_buffer/";
    string bias_datapath="/media/sdcard/ComputeLibrary/data/neon_alexnet/bias/";
    string Q_table_name[8]={
        "conv0_q.npy",
        "conv1_q.npy",
        "conv2_q.npy",
        "conv3_q.npy",
        "conv4_q.npy",
        "conv5_q.npy",
        "conv6_q.npy",
        "conv7_q.npy"
    };
    string WT_buffer_name[8]={
        "conv0_wt.npy",
        "conv1_wt.npy",
        "conv2_wt.npy",
        "conv3_wt.npy",
        "conv4_wt.npy",
        "conv5_wt.npy",
        "conv6_wt.npy",
        "conv7_wt.npy"
    };
    string bias_name[8]={
        "conv0_b.npy",
        "conv1_b.npy",
        "conv2_b.npy",
        "conv3_b.npy",
        "conv4_b.npy",
        "conv5_b.npy",
        "conv6_b.npy",
        "conv7_b.npy"
    };
    bool is_fortran{};
	string output_filename="/media/sdcard/ComputeLibrary/data/neon_alexnet/output.npy";

    Tensor src{};Tensor input{};
    Tensor  Q_table_conv0{}; Tensor WT_buffer_conv0{}; Tensor bias_conv0{};
    Tensor  Q_table_conv1{}; Tensor WT_buffer_conv1{}; Tensor bias_conv1{};
    Tensor  Q_table_conv2{}; Tensor WT_buffer_conv2{}; Tensor bias_conv2{};
    Tensor  Q_table_conv3{}; Tensor WT_buffer_conv3{}; Tensor bias_conv3{};
    Tensor  Q_table_conv4{}; Tensor WT_buffer_conv4{}; Tensor bias_conv4{};
    /*fully connected layer*/
    Tensor  Q_table_conv5{}; Tensor WT_buffer_conv5{}; Tensor bias_conv5{};
    Tensor  Q_table_conv6{}; Tensor WT_buffer_conv6{}; Tensor bias_conv6{};
    Tensor  Q_table_conv7{}; Tensor WT_buffer_conv7{}; Tensor bias_conv7{};

    Tensor out_conv0{}; Tensor out_act0{}; Tensor out_norm0{}; Tensor out_pool0{};
    Tensor out_conv1{}; Tensor out_act1{}; Tensor out_norm1{}; Tensor out_pool1{};
    Tensor out_conv2{}; Tensor out_act2{};
    Tensor out_conv3{}; Tensor out_act3{};
    Tensor out_conv4{}; Tensor out_act4{}; Tensor out_pool4{};
    Tensor out_conv5{}; Tensor out_act5{};
    Tensor out_conv6{}; Tensor out_act6{};
    Tensor out_conv7{};
    /*Tensor out_class{};
    Tensor out_softmax{};*/

    /*type change tensor*/
    Tensor sfconv0{}; Tensor fspool0{};
    Tensor sfconv1{}; Tensor fspool1{};
    Tensor sfconv2{}; Tensor fsact2{};
    Tensor sfconv3{}; Tensor fsact3{};
    Tensor sfconv4{}; Tensor fspool4{};
    Tensor sfconv5{}; Tensor fsact5{};
    Tensor sfconv6{}; Tensor fsact6{};
    Tensor sfconv7{};

    NEABMConvolutionLayer conv0{}; NEActivationLayer act0{}; NENormalizationLayer norm0{}; NEPoolingLayer pool0{};
    NEABMConvolutionLayer conv1{}; NEActivationLayer act1{}; NENormalizationLayer norm1{}; NEPoolingLayer pool1{};
    NEABMConvolutionLayer conv2{}; NEActivationLayer act2{};
    NEABMConvolutionLayer conv3{}; NEActivationLayer act3{};
    NEABMConvolutionLayer conv4{}; NEActivationLayer act4{}; NEPoolingLayer pool4{};
    NEABMConvolutionLayer conv5{}; NEActivationLayer act5{};
    NEABMConvolutionLayer conv6{}; NEActivationLayer act6{};
    NEABMConvolutionLayer conv7{};
    /*NESoftmaxLayer softmax{};*/

    NEF32toS8Layer input_convert{};
    NES8toF32Layer lsfconv0{}; NEF32toS8Layer lfspool0{};
    NES8toF32Layer lsfconv1{}; NEF32toS8Layer lfspool1{};
    NES8toF32Layer lsfconv2{}; NEF32toS8Layer lfsact2{};
    NES8toF32Layer lsfconv3{}; NEF32toS8Layer lfsact3{};
    NES8toF32Layer lsfconv4{}; NEF32toS8Layer lfspool4{};
    NES8toF32Layer lsfconv5{}; NEF32toS8Layer lfsact5{};
    NES8toF32Layer lsfconv6{}; NEF32toS8Layer lfsact6{};
    NES8toF32Layer lsfconv7{};


};
int main(int argc, char **argv)
{
	return utils::run_example<NEALEXNETExample>(argc, argv);
}

