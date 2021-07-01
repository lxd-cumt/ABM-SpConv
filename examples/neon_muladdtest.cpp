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

/*
using namespace arm_compute;
using namespace utils;

class NEMulAddTestExample : public Example
{
public:
    bool do_setup(int argc, char **argv) override
    {
        
            NPYLoader npy0;
            NPYLoader npy1;
            string input_a_filename="/media/sdcard/ComputeLibrary/data/neon_muladdtest/input_a.npy";
            string input_b_filename="/media/sdcard/ComputeLibrary/data/neon_muladdtest/input_b.npy";
            npy0.open(input_a_filename);
            npy0.init_tensor2(src0, DataType::S16);
            npy1.open(input_b_filename);
            npy1.init_tensor2(src1,DataType::S16);

            test.configure(&src0, &src1, &dst);

            src0.allocator()->allocate();
            src1.allocator()->allocate();
            dst.allocator()->allocate();
            if(npy0.is_open()) 
            {
                npy0.fill_tensor2(src0);
                npy1.fill_tensor2(src1);
                is_fortran      = npy0.is_fortran();
            }
        
        return true;
    }
    void do_run() override
    {
        
        double total=0;
        for(unsigned int i=0;i<300;i++)
        {
            auto tbegin = std::chrono::high_resolution_clock::now();
            test.run();
            auto tend = std::chrono::high_resolution_clock::now();
            double cost= std::chrono::duration_cast<std::chrono::duration<double>>(tend - tbegin).count();
            total=total+cost;
            std::cout << cost*1000 <<" ms"<< std::endl;
        }
            std::cout << "average: " << total*1000 /300<<" ms"<< std::endl;
        output_filename="/media/sdcard/ComputeLibrary/data/neon_muladdtest/output.npy";
        if(!output_filename.empty()) 
		{
            NPYLoader SSS;
			SSS.save_to_npy4(dst, output_filename, is_fortran);
		}
        
       
       auto tbegin = std::chrono::high_resolution_clock::now();
        for(unsigned int i=0; i<10000; i++)
        {
            for(unsigned int j=0; j<10000; j++)
            {
                int16_t ope_a=(int16_t)i;
                int16_t ope_b=(int16_t)j;
                int16_t res=ope_a+ope_b;
            }
        }
        auto tend = std::chrono::high_resolution_clock::now();
        double cost= std::chrono::duration_cast<std::chrono::duration<double>>(tend - tbegin).count();
        std::cout << cost*1000 <<" ms"<< std::endl;
        
    }

private:
    Tensor      src0{}, src1{}, dst{};
    NEMulAddTestLayer      test{};
    bool        is_fortran{};
    std::string output_filename{};
    
};

int main(int argc, char **argv)
{
    return utils::run_example<NEMulAddTestExample>(argc, argv);
}
*/

/*
using namespace std;
int main(){
    
    unsigned int running_times=20;
    double total=0, avg=0;
    long long int inner_loops=400000;
    for(unsigned int time=0; time<running_times; time++)
    {
        auto tbegin = std::chrono::high_resolution_clock::now();
        long long int sum=1;
        for(long long  int i=0; i<inner_loops; i++)
        {
            sum=i*(i+1);
        }
        auto tend = std::chrono::high_resolution_clock::now();
        double cost= std::chrono::duration_cast<std::chrono::duration<double>>(tend - tbegin).count();
        std::cout << cost*1000 <<" ms"<< std::endl;
        total+=cost;
    }
    std::cout<<"avg-time: "<<std::endl;
    avg=total/running_times;
    std::cout << avg*1000 <<" ms"<< std::endl;
    return 0;

}
*/

/* General Purpose Register Test */
/*
using namespace std;
int main()
{
    unsigned long long int  times=10000000;
    unsigned long long int  out=0;
    double sum_time=0;
    unsigned int cycles=100;
    for(unsigned int ii=0; ii<cycles; ii++)
    {
    auto tbegin = std::chrono::high_resolution_clock::now();
    asm volatile (
        "mov x1, %[times]                               \n"
        "mov x2, #0                                             \n"
        "mov x3, #2                                             \n"
        "2:                                                               \n"
        "mul x2, x2, x3                                  \n"
        "subs x1, x1, #1                                     \n"
        "bne 2b                                                    \n"
        "mov %[out], x2                                   \n"

        :[out] "+r" (out)
        :[times]"r"(times)
        :"memory", "x1", "x2", "x3"
    );
    auto tend = std::chrono::high_resolution_clock::now();
    double cost= std::chrono::duration_cast<std::chrono::duration<double>>(tend - tbegin).count();
    sum_time+=cost;
    std::cout << cost*1000 <<" ms"<< std::endl;
    }
    std::cout<<"avg_time= "<<sum_time*1000/cycles<<"ms"<<std::endl;
    return 0;
}
*/

/* NEON Register test*//* AArch64 contains instructions such as add, instead of vadd.*/
using namespace std;
int main()
{
    unsigned long long int  times=10000000000;
    unsigned long long int  out=0;
    double sum_time=0;
    unsigned int cycles=10;
    for(unsigned int ii=0; ii<cycles; ii++)
    {
    auto tbegin = std::chrono::high_resolution_clock::now();
    asm volatile (
        "mov x1, %[times]                               \n"
        "mov w2, #0                                             \n"
        "mov w3, #2                                             \n"
        // "dup v2.8h, w2                              \n"
        // "dup v3.8h, w3                              \n"
        "dup v2.4s, w2                              \n"
        "dup v3.4s, w3                              \n"


        "2:                                                               \n"
        // "mul v2.8h, v2.8h, v3.8h                                  \n"
        "fadd v2.4s, v2.4s, v3.4s                                  \n"
        "subs x1, x1, #1                                     \n"
        "bne 2b                                                    \n"
        "mov %[out], x2                                   \n"

        :[out] "+r" (out)
        :[times]"r"(times)
        :"memory", "x1", "x2", "x3"
    );
    auto tend = std::chrono::high_resolution_clock::now();
    double cost= std::chrono::duration_cast<std::chrono::duration<double>>(tend - tbegin).count();
    sum_time+=cost;
    std::cout << cost*1000 <<" ms"<< std::endl;
    }
    std::cout<<"avg_time= "<<sum_time*1000/cycles<<"ms"<<std::endl;
    return 0;
}
