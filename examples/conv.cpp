#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/NEON/NEFunctions.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "arm_compute/runtime/CPP/CPPScheduler.h"
#include "arm_compute/runtime/Allocator.h"
#include "utils/ImageLoader.h"
#include "utils/Utils.h"
#include "utils/GraphUtils.h"
#include <ctime>
#include <cstdlib>
#include <unistd.h>

#include <cstdlib>
#include <memory>
#include <iostream>
#include <string>
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
					        
					        unsigned int width_src_image  = atoi(argv[1]);
						        unsigned int height_src_image = width_src_image;
							        unsigned int ifm_src_img      = atoi(argv[2]);

								        const TensorShape src_shape(width_src_image, height_src_image, ifm_src_img);
									        src.allocator()->init(TensorInfo(src_shape, 1, DataType::F32));

										        /*
											 *         ///////////////  layer 1 
											 *                 // Initialize tensors of conv1
											 *                         */
										        unsigned int kernel_x_conv1 = atoi(argv[3]);
											        unsigned int kernel_y_conv1 = kernel_x_conv1;
												        unsigned int ofm_conv1      = atoi(argv[4]);
													        unsigned int stride         =atoi(argv[5]);
														        int pad=0;
															        unsigned int out_x_conv1    = (src_shape.x()-kernel_x_conv1 + pad*2)/stride+1;
																        unsigned int out_y_conv1    = out_x_conv1;        
																	        const TensorShape weights_shape_conv1(kernel_x_conv1, kernel_y_conv1, src_shape.z(), ofm_conv1);
																		        const TensorShape biases_shape_conv1(weights_shape_conv1[3]);
																			        const TensorShape out_shape_conv1(out_x_conv1, out_y_conv1, weights_shape_conv1[3]);
																				        weights1.allocator()->init(TensorInfo(weights_shape_conv1, 1, DataType::F32));
																					        biases1.allocator()->init(TensorInfo(biases_shape_conv1, 1, DataType::F32));
																						        out_conv1.allocator()->init(TensorInfo(out_shape_conv1, 1, DataType::F32));
																							        /* Initialize tensor of act0 */
																							        out_act1.allocator()->init(TensorInfo(out_shape_conv1, 1, DataType::F32));
																								       
																								        /* -----------------------End: [Initialize tensors] */

																								        /*-----------------BEGIN:[Configure Functions]--------------*/

																								        /* [Configure functions] */

																								        /*///layer1*/
																								        conv1.configure(&src, &weights1, &biases1, &out_conv1, PadStrideInfo(stride, stride, pad, pad));
																									        act1.configure(&out_conv1, &out_act1, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

																										        /* -----------------------End: [Configure functions] */

																										        /*---------------END:[Allocate tensors]---------------*/
																										       /* ///layer1*/
																										        out_conv1.allocator()->allocate();
																											        out_act1.allocator()->allocate();
																												       
																												        /* -----------------------End: [ Add tensors to memory manager ] */

																												        /* [Allocate tensors] */

																												        /*// Now that the padding requirements are known we can allocate all tensors*/
																												        src.allocator()->allocate();
																													        weights1.allocator()->allocate();

																														        biases1.allocator()->allocate();
																															        /* -----------------------End: [Allocate tensors] */

																															        
																															        return true;

																																    }

		        void do_run() override
				    {
					            int times =100 ;
						            for(int i=0; i<times; i++){
								                conv1.run();
										            act1.run();
											            }
							        }

	private:
			    /*// The src tensor should contain the input image*/
			    Tensor src{};

			        Tensor weights0{},weights1{};
				    Tensor biases0{},biases1{};

				        Tensor out_conv0{};
					    Tensor out_conv1{};
					        

					        Tensor out_act0{};
						    Tensor out_act1{};
						        

						        Tensor out_norm0{};
							    Tensor out_norm1{};

							        Tensor out_pool0{};
								    Tensor out_pool1{};
								        Tensor out_pool2{};

									    /*
									     *     NEConvolutionLayer          conv0{};
									     *         NEConvolutionLayer          conv1{};
									     *             */
									    NEGEMMConvolutionLayer conv0{};
									        NEGEMMConvolutionLayer conv1{};

										    NEPoolingLayer              pool0{};
										        NEPoolingLayer              pool1{};
											    NEPoolingLayer              pool2{};

											        NEActivationLayer           act0{};
												    NEActivationLayer           act1{};


};


int main(int argc, char **argv)
{

	    int cpus = 8;
	        for(int i=0;i<4;i++){
			        ofstream out("./lab3/kernel_exec_time.csv", ios::out | ios::app);
				        out<<"core= "<<(i+1)<<"s"<<",";
					        out<<atoi(argv[1])<<","<<atoi(argv[2])<<","<<atoi(argv[3])<<","<<atoi(argv[4])<<","<<atoi(argv[5])<<",";
						        out.close();
							        cpu_set_t cpuset;
								        cpu_set_t get;
									        int num = 0;

										        CPU_ZERO(&cpuset);
											        for(int j=0;j<=i;j++){
													            CPU_SET(j, &cpuset);
														            }
												        
												        int e = sched_setaffinity(getpid(), sizeof(cpuset), &cpuset);
													        if(e !=0) {
															            std::cout << "Error in setting sched_setaffinity \n";
																            }
														        CPU_ZERO(&get);
															        if (sched_getaffinity(0, sizeof(get), &get) == -1) {
																	            printf("get CPU affinity failue, ERROR:%s\n", strerror(errno));
																		                return -1; 
																				        }
																        for(int j = 0; j < cpus; j++) {
																		            if (CPU_ISSET(j, &get)) { 
																				                    num++;
																						                }    
																			            }

																	        CPPScheduler::get().set_num_threads(num);
																		        utils::run_example<NEONALEXExample>(argc, argv);
																			    }
		    
		    
		    
		    for(int i=4;i<8;i++){
			            ofstream out("./lab3/kernel_exec_time.csv", ios::out | ios::app);
				            out<<"core= "<<(i-3)<<"b"<<",";
					            out<<atoi(argv[1])<<","<<atoi(argv[2])<<","<<atoi(argv[3])<<","<<atoi(argv[4])<<","<<atoi(argv[5])<<",";
						            out.close();

							            cpu_set_t cpuset;
								            cpu_set_t get;
									            int num = 0;

										            CPU_ZERO(&cpuset);
											            for(int j=4;j<=i;j++){
													                CPU_SET(j, &cpuset);
															        }
												            
												            int e = sched_setaffinity(getpid(), sizeof(cpuset), &cpuset);
													            if(e !=0) {
															                std::cout << "Error in setting sched_setaffinity \n";
																	        }

														            CPU_ZERO(&get);
															            if (sched_getaffinity(0, sizeof(get), &get) == -1) {
																	                printf("get CPU affinity failue, ERROR:%s\n", strerror(errno));
																			            return -1; 
																				            }   
																            for(int j = 0; j < cpus; j++) {
																		                if (CPU_ISSET(j, &get)) { 
																					                num++;
																							            }    
																				        }
																	            CPPScheduler::get().set_num_threads(num);
																		            utils::run_example<NEONALEXExample>(argc, argv);
																			        }
}


