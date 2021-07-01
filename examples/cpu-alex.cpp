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

					        /* [Initialize tensors] */
					        constexpr unsigned int width_src_image  = 227;
						        constexpr unsigned int height_src_image = 227;
							        constexpr unsigned int ifm_src_img      = 3;

								        const TensorShape src_shape(width_src_image, height_src_image, ifm_src_img);
									        src.allocator()->init(TensorInfo(src_shape, 1, DataType::F32));



										        constexpr unsigned int kernel_x_conv0 = 11;
											        constexpr unsigned int kernel_y_conv0 = 11;
												        constexpr unsigned int ofm_conv0      = 96;
													        constexpr unsigned int out_x_conv0    = 55;
														        constexpr unsigned int out_y_conv0    = 55;
															        const TensorShape weights_shape_conv0(kernel_x_conv0, kernel_y_conv0, src_shape.z(), ofm_conv0);
																        const TensorShape biases_shape_conv0(weights_shape_conv0[3]);
																	        const TensorShape out_shape_conv0(out_x_conv0, out_y_conv0, weights_shape_conv0[3]);
																		        weights0.allocator()->init(TensorInfo(weights_shape_conv0, 1, DataType::F32));
																			        biases0.allocator()->init(TensorInfo(biases_shape_conv0, 1, DataType::F32));
																				        out_conv0.allocator()->init(TensorInfo(out_shape_conv0, 1, DataType::F32));

																					        out_act0.allocator()->init(TensorInfo(out_shape_conv0, 1, DataType::F32));

																						        out_norm0.allocator()->init(TensorInfo(out_shape_conv0, 1, DataType::F32));

																							        TensorShape out_shape_pool0 = out_shape_conv0;
																								        out_shape_pool0.set(0, out_shape_pool0.x() / 2); 
																									        out_shape_pool0.set(1, out_shape_pool0.y() / 2);
																										        out_pool0.allocator()->init(TensorInfo(out_shape_pool0, 1, DataType::F32));




																											        constexpr unsigned int kernel_x_conv1 = 5;
																												        constexpr unsigned int kernel_y_conv1 = 5;
																													        constexpr unsigned int ofm_conv1      = 256;
																														        constexpr unsigned int out_x_conv1    = 27;
																															        constexpr unsigned int out_y_conv1    = 27;        
																																        const TensorShape weights_shape_conv1(kernel_x_conv1, kernel_y_conv1, out_shape_pool0.z(), ofm_conv1);
																																	        const TensorShape biases_shape_conv1(weights_shape_conv1[3]);
																																		        const TensorShape out_shape_conv1(out_x_conv1, out_y_conv1, weights_shape_conv1[3]);
																																			        weights1.allocator()->init(TensorInfo(weights_shape_conv1, 1, DataType::F32));
																																				        biases1.allocator()->init(TensorInfo(biases_shape_conv1, 1, DataType::F32));
																																					        out_conv1.allocator()->init(TensorInfo(out_shape_conv1, 1, DataType::F32));

																																						        out_act1.allocator()->init(TensorInfo(out_shape_conv1, 1, DataType::F32));

																																							        out_norm1.allocator()->init(TensorInfo(out_shape_conv1, 1, DataType::F32));

																																								        TensorShape out_shape_pool1 = out_shape_conv1;
																																									        out_shape_pool1.set(0, out_shape_pool1.x() / 2);
																																										        out_shape_pool1.set(1, out_shape_pool1.y() / 2);
																																											        out_pool1.allocator()->init(TensorInfo(out_shape_pool1, 1, DataType::F32));




																																												        constexpr unsigned int kernel_x_conv2 = 3;
																																													        constexpr unsigned int kernel_y_conv2 = 3;
																																														        constexpr unsigned int ofm_conv2      = 384;
																																															        constexpr unsigned int out_x_conv2    = 13;
																																																        constexpr unsigned int out_y_conv2    = 13;  
																																																	        const TensorShape weights_shape_conv2(kernel_x_conv2, kernel_y_conv2, out_shape_pool1.z(), ofm_conv2);
																																																		        const TensorShape biases_shape_conv2(weights_shape_conv2[3]);
																																																			        const TensorShape out_shape_conv2(out_x_conv2, out_y_conv2, weights_shape_conv2[3]);
																																																				        weights2.allocator()->init(TensorInfo(weights_shape_conv2, 1, DataType::F32));
																																																					        biases2.allocator()->init(TensorInfo(biases_shape_conv2, 1, DataType::F32));
																																																						        out_conv2.allocator()->init(TensorInfo(out_shape_conv2, 1, DataType::F32));

																																																							        out_act2.allocator()->init(TensorInfo(out_shape_conv2, 1, DataType::F32));




																																																								        constexpr unsigned int kernel_x_conv3 = 3;
																																																									        constexpr unsigned int kernel_y_conv3 = 3;
																																																										        constexpr unsigned int ofm_conv3      = 384;
																																																											        constexpr unsigned int out_x_conv3    = 13;
																																																												        constexpr unsigned int out_y_conv3    = 13;
																																																													        const TensorShape weights_shape_conv3(kernel_x_conv3, kernel_y_conv3, out_shape_conv2.z(), ofm_conv3);
																																																														        const TensorShape biases_shape_conv3(weights_shape_conv3[3]);
																																																															        const TensorShape out_shape_conv3(out_x_conv3, out_y_conv3, weights_shape_conv3[3]);
																																																																        weights3.allocator()->init(TensorInfo(weights_shape_conv3, 1, DataType::F32 ));
																																																																	        biases3.allocator()->init(TensorInfo(biases_shape_conv3, 1, DataType::F32));
																																																																		        out_conv3.allocator()->init(TensorInfo(out_shape_conv3, 1, DataType::F32));

																																																																			        out_act3.allocator()->init(TensorInfo(out_shape_conv3, 1, DataType::F32));



																																																																				        constexpr unsigned int kernel_x_conv4 = 3;
																																																																					        constexpr unsigned int kernel_y_conv4 = 3;
																																																																						        constexpr unsigned int ofm_conv4      = 256;
																																																																							        constexpr unsigned int out_x_conv4    = 13;
																																																																								        constexpr unsigned int out_y_conv4    = 13;
																																																																									        const TensorShape weights_shape_conv4(kernel_x_conv4, kernel_y_conv4, out_shape_conv3.z(), ofm_conv4);
																																																																										        const TensorShape biases_shape_conv4(weights_shape_conv4[3]);
																																																																											        const TensorShape out_shape_conv4(out_x_conv4, out_y_conv4, weights_shape_conv4[3]);
																																																																												        weights4.allocator()->init(TensorInfo(weights_shape_conv4, 1, DataType::F32));
																																																																													        biases4.allocator()->init(TensorInfo(biases_shape_conv4, 1, DataType::F32));
																																																																														        out_conv4.allocator()->init(TensorInfo(out_shape_conv4, 1, DataType::F32));

																																																																															        out_act4.allocator()->init(TensorInfo(out_shape_conv4, 1, DataType::F32));

																																																																																        TensorShape out_shape_pool2 = out_shape_conv4;
																																																																																	        out_shape_pool2.set(0, out_shape_pool2.x() / 2);
																																																																																		        out_shape_pool2.set(1, out_shape_pool2.y() / 2);
																																																																																			        out_pool2.allocator()->init(TensorInfo(out_shape_pool2, 1, DataType::F32));


																																																																																				        /*unsigned int num_labels = 4096;*/

																																																																																				        const TensorShape weights_shape_fc0(out_shape_pool2.x() * out_shape_pool2.y() * out_shape_pool2.z(), 4096);
																																																																																					        const TensorShape biases_shape_fc0(4096);
																																																																																						        const TensorShape out_shape_fc0(4096);

																																																																																							        weights5.allocator()->init(TensorInfo(weights_shape_fc0, 1, DataType::F32));
																																																																																								        biases5.allocator()->init(TensorInfo(biases_shape_fc0, 1, DataType::F32));
																																																																																									        out_fc0.allocator()->init(TensorInfo(out_shape_fc0, 1, DataType::F32));


																																																																																										        out_act5.allocator()->init(TensorInfo(out_shape_fc0, 1, DataType::F32));


																																																																																											        /*num_labels = 4096; */

																																																																																											        const TensorShape weights_shape_fc1(out_shape_fc0.x() * out_shape_fc0.y() * out_shape_fc0.z(), 4096);
																																																																																												        const TensorShape biases_shape_fc1(4096);
																																																																																													        const TensorShape out_shape_fc1(4096);

																																																																																														        weights6.allocator()->init(TensorInfo(weights_shape_fc1, 1, DataType::F32));
																																																																																															        biases6.allocator()->init(TensorInfo(biases_shape_fc1, 1, DataType::F32));
																																																																																																        out_fc1.allocator()->init(TensorInfo(out_shape_fc1, 1, DataType::F32));


																																																																																																	        out_act6.allocator()->init(TensorInfo(out_shape_fc1, 1, DataType::F32));



																																																																																																		        /*num_labels = 1000;  */

																																																																																																		        const TensorShape weights_shape_fc2(out_shape_fc1.x() * out_shape_fc1.y() * out_shape_fc1.z(), 1000);
																																																																																																			        const TensorShape biases_shape_fc2(1000);
																																																																																																				        const TensorShape out_shape_fc2(1000);

																																																																																																					        weights7.allocator()->init(TensorInfo(weights_shape_fc2, 1, DataType::F32));
																																																																																																						        biases7.allocator()->init(TensorInfo(biases_shape_fc2, 1, DataType::F32));
																																																																																																							        out_fc2.allocator()->init(TensorInfo(out_shape_fc2, 1, DataType::F32));


																																																																																																								        const TensorShape out_shape_softmax(out_shape_fc2.x());
																																																																																																									        out_softmax.allocator()->init(TensorInfo(out_shape_softmax, 1, DataType::F32));

																																																																																																										        /* -----------------------End: [Initialize tensors] */

																																																																																																										        /*-----------------BEGIN:[Configure Functions]--------------*/

																																																																																																										        /* [Configure functions] */


																																																																																																										        conv0.configure(&src, &weights0, &biases0, &out_conv0, PadStrideInfo(4, 4, 0, 0));
																																																																																																											        act0.configure(&out_conv0, &out_act0, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
																																																																																																												        norm0.configure(&out_act0, &out_norm0, NormalizationLayerInfo(NormType::CROSS_MAP, 5, 0.0001f, 0.75f));
																																																																																																													        pool0.configure(&out_norm0, &out_pool0, PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 0)));


																																																																																																														        conv1.configure(&out_pool0, &weights1, &biases1, &out_conv1, PadStrideInfo(1, 1, 2, 2));
																																																																																																															        act1.configure(&out_conv1, &out_act1, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
																																																																																																																        norm1.configure(&out_act1, &out_norm1, NormalizationLayerInfo(NormType::CROSS_MAP, 5, 0.0001f, 0.75f));
																																																																																																																	        pool1.configure(&out_norm1, &out_pool1, PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 0)));


																																																																																																																		        conv2.configure(&out_pool1, &weights2, &biases2, &out_conv2, PadStrideInfo(1, 1, 1, 1));
																																																																																																																			        act2.configure(&out_conv2, &out_act2, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));


																																																																																																																				        conv3.configure(&out_act2, &weights3, &biases3, &out_conv3, PadStrideInfo(1, 1, 1, 1));
																																																																																																																					        act3.configure(&out_conv3, &out_act3, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));


																																																																																																																						        conv4.configure(&out_act3, &weights4, &biases4, &out_conv4, PadStrideInfo(1, 1, 1, 1));
																																																																																																																							        act4.configure(&out_conv4, &out_act4, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
																																																																																																																								        pool2.configure(&out_act4, &out_pool2, PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 0)));


																																																																																																																									        fc0.configure(&out_pool2, &weights5, &biases5, &out_fc0);
																																																																																																																										        act5.configure(&out_fc0, &out_act5, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

																																																																																																																											        fc1.configure(&out_act5, &weights6, &biases6, &out_fc1);
																																																																																																																												        act6.configure(&out_fc1, &out_act6, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

																																																																																																																													        fc2.configure(&out_act6, &weights7, &biases7, &out_fc2);
																																																																																																																														        softmax.configure(&out_fc2, &out_softmax);

																																																																																																																															        /* -----------------------End: [Configure functions] */

																																																																																																																															        /*---------------END:[Allocate tensors]---------------*/

																																																																																																																															        out_conv0.allocator()->allocate();
																																																																																																																																        out_act0.allocator()->allocate();
																																																																																																																																	        out_norm0.allocator()->allocate();
																																																																																																																																		        out_pool0.allocator()->allocate();

																																																																																																																																			        out_conv1.allocator()->allocate();
																																																																																																																																				        out_act1.allocator()->allocate();
																																																																																																																																					        out_norm1.allocator()->allocate();
																																																																																																																																						        out_pool1.allocator()->allocate();

																																																																																																																																							        out_conv2.allocator()->allocate();
																																																																																																																																								        out_act2.allocator()->allocate();

																																																																																																																																									        out_conv3.allocator()->allocate();
																																																																																																																																										        out_act3.allocator()->allocate();

																																																																																																																																											        out_conv4.allocator()->allocate();
																																																																																																																																												        out_act4.allocator()->allocate();
																																																																																																																																													        out_pool2.allocator()->allocate();


																																																																																																																																														        out_fc0.allocator()->allocate();
																																																																																																																																															        out_act5.allocator()->allocate();

																																																																																																																																																        out_fc1.allocator()->allocate();
																																																																																																																																																	        out_act6.allocator()->allocate();

																																																																																																																																																		        out_fc2.allocator()->allocate();
																																																																																																																																																			        out_softmax.allocator()->allocate();
																																																																																																																																																				        /* -----------------------End: [ Add tensors to memory manager ] */

																																																																																																																																																				        /* [Allocate tensors] */

																																																																																																																																																				        src.allocator()->allocate();
																																																																																																																																																					        weights0.allocator()->allocate();
																																																																																																																																																						        weights1.allocator()->allocate();
																																																																																																																																																							        weights2.allocator()->allocate();
																																																																																																																																																								        weights3.allocator()->allocate();
																																																																																																																																																									        weights4.allocator()->allocate();
																																																																																																																																																										        weights5.allocator()->allocate();
																																																																																																																																																											        weights6.allocator()->allocate();
																																																																																																																																																												        weights7.allocator()->allocate();

																																																																																																																																																													        biases0.allocator()->allocate();
																																																																																																																																																														        biases1.allocator()->allocate();
																																																																																																																																																															        biases2.allocator()->allocate();
																																																																																																																																																																        biases3.allocator()->allocate();
																																																																																																																																																																	        biases4.allocator()->allocate();
																																																																																																																																																																		        biases5.allocator()->allocate();
																																																																																																																																																																			        biases6.allocator()->allocate();
																																																																																																																																																																				        biases7.allocator()->allocate();
																																																																																																																																																																					        /* -----------------------End: [Allocate tensors] */
																																																																																																																																																																					        return true;

																																																																																																																																																																						    }

		        void do_run() override
				    {
					            /*ofstream out("./lab3/alexnet/avg_time.csv", ios::out | ios::app);*/
					            unsigned int times=100;
						            for(unsigned int i=0; i<times; i++)
								            { 
										                /*auto tbegin0=std::chrono::high_resolution_clock::now()*/;
												            conv0.run();
													                /*auto tend0=std::chrono::high_resolution_clock::now()*/;
															            /*doublelayer_time0 = std::chrono::duration_cast<std::chrono::duration<double>>(tend0 - tbegin0).count();*/
															            act0.run();
																                norm0.run();
																		            pool0.run();


																			                /*auto tbegin1=std::chrono::high_resolution_clock::now()*/;
																					            conv1.run();
																						                /*auto tend1=std::chrono::high_resolution_clock::now()*/;
																								            /*doublelayer_time1 = std::chrono::duration_cast<std::chrono::duration<double>>(tend1 - tbegin1).count();*/
																								            act1.run();
																									                norm1.run();
																											            pool1.run();

																												                
																												                /*auto tbegin2=std::chrono::high_resolution_clock::now()*/;
																														            conv2.run();
																															                /*auto tend2=std::chrono::high_resolution_clock::now()*/;
																																	            /*doublelayer_time2 = std::chrono::duration_cast<std::chrono::duration<double>>(tend2 - tbegin2).count();*/
																																	            act2.run();

																																		              
																																		                /*auto tbegin3=std::chrono::high_resolution_clock::now()*/;
																																				            conv3.run();
																																					                /*auto tend3=std::chrono::high_resolution_clock::now()*/;
																																							            /*doublelayer_time3 = std::chrono::duration_cast<std::chrono::duration<double>>(tend3 - tbegin3).count();*/
																																							            act3.run();
																																								               
																																								                /*auto tbegin4=std::chrono::high_resolution_clock::now()*/;
																																										            conv4.run();
																																											                /*auto tend4=std::chrono::high_resolution_clock::now()*/;
																																													            /*doublelayer_time4 = std::chrono::duration_cast<std::chrono::duration<double>>(tend4 - tbegin4).count();*/
																																													            act4.run();
																																														                pool2.run();

																																																            /*auto tbegin5=std::chrono::high_resolution_clock::now()*/;
																																																	                fc0.run();
																																																			            /*auto tend5=std::chrono::high_resolution_clock::now()*/;
																																																				                /*doublelayer_time5 = std::chrono::duration_cast<std::chrono::duration<double>>(tend5 - tbegin5).count();*/
																																																				                act5.run();

																																																						            /*auto tbegin6=std::chrono::high_resolution_clock::now()*/;
																																																							                fc1.run();
																																																									            /*auto tend6=std::chrono::high_resolution_clock::now()*/;
																																																										                /*doublelayer_time6=std::chrono::duration_cast<std::chrono::duration<double>>(tend6 - tbegin6).count();*/
																																																										                act6.run();

																																																												            /*auto tbegin7=std::chrono::high_resolution_clock::now()*/;
																																																													                fc2.run();
																																																															            /*auto tend7=std::chrono::high_resolution_clock::now()*/;
																																																																                /*doublelayer_time7=std::chrono::duration_cast<std::chrono::duration<double>>(tend7 - tbegin7).count();*/
																																																																                softmax.run();
																																																																		            
																																																																		            /*
																																																																			     *             if(i>0){
																																																																			     *                             conv_layer_time0+=layer_time0*1000;
																																																																			     *                                             store_conv_kernel_time(conv0, conv_kernel_time0);
																																																																			     *                                                             conv_layer_time1+=layer_time1*1000;
																																																																			     *                                                                             store_conv_kernel_time(conv1, conv_kernel_time1);
																																																																			     *                                                                                             conv_layer_time2+=layer_time2*1000;
																																																																			     *                                                                                                             store_conv_kernel_time(conv2, conv_kernel_time2);
																																																																			     *                                                                                                                             conv_layer_time3+=layer_time3*1000;
																																																																			     *                                                                                                                                             store_conv_kernel_time(conv3, conv_kernel_time3);
																																																																			     *                                                                                                                                                             conv_layer_time4+=layer_time4*1000;
																																																																			     *                                                                                                                                                                             store_conv_kernel_time(conv4, conv_kernel_time4);
																																																																			     *                                                                                                                                                                                             fc_layer_time0+=layer_time5*1000;
																																																																			     *                                                                                                                                                                                                             store_fc_kernel_time(fc0, fc_kernel_time0);
																																																																			     *                                                                                                                                                                                                                             fc_layer_time1+=layer_time6*1000;
																																																																			     *                                                                                                                                                                                                                                             store_fc_kernel_time(fc1, fc_kernel_time1);
																																																																			     *                                                                                                                                                                                                                                                             fc_layer_time2+=layer_time7*1000;
																																																																			     *                                                                                                                                                                                                                                                                             store_fc_kernel_time(fc2, fc_kernel_time2);
																																																																			     *                                                                                                                                                                                                                                                                                         }
																																																																			     *                                                                                                                                                                                                                                                                                                     */
																																																																		        }
							            /*
								     *         conv_layer_time0=conv_layer_time0/(count-1);avg_kernel_time(conv_kernel_time0, 11, (count-1));
								     *                 conv_layer_time1=conv_layer_time1/(count-1);avg_kernel_time(conv_kernel_time1, 11, (count-1));
								     *                         conv_layer_time2=conv_layer_time2/(count-1);avg_kernel_time(conv_kernel_time2, 11, (count-1));
								     *                                 conv_layer_time3=conv_layer_time3/(count-1);avg_kernel_time(conv_kernel_time3, 11, (count-1));
								     *                                         conv_layer_time4=conv_layer_time4/(count-1);avg_kernel_time(conv_kernel_time4, 11, (count-1));
								     *                                                 fc_layer_time0=fc_layer_time0/(count-1);avg_kernel_time(fc_kernel_time0, 10, (count-1));
								     *                                                         fc_layer_time1=fc_layer_time1/(count-1);avg_kernel_time(fc_kernel_time1, 10, (count-1));
								     *                                                                 fc_layer_time2=fc_layer_time2/(count-1);avg_kernel_time(fc_kernel_time2, 10, (count-1));
								     *                                                                         
								     *                                                                                 out<<"convolution layer"<<",";
								     *                                                                                         for(unsigned int j=0; j<11; j++)
								     *                                                                                                 {
								     *                                                                                                             out<<conv_kernel_name[j]<<",";
								     *                                                                                                                     }
								     *                                                                                                                             out<<std::endl;
								     *                                                                                                                                     out<<conv_layer_time0<<",";
								     *                                                                                                                                             for(unsigned int j=0; j<11; j++)
								     *                                                                                                                                                     {
								     *                                                                                                                                                                 out<<conv_kernel_time0[j]<<",";
								     *                                                                                                                                                                         }
								     *                                                                                                                                                                                 out<<std::endl;
								     *                                                                                                                                                                                         out<<conv_layer_time1<<",";
								     *                                                                                                                                                                                                 for(unsigned int j=0; j<11; j++)
								     *                                                                                                                                                                                                         {
								     *                                                                                                                                                                                                                     out<<conv_kernel_time1[j]<<",";
								     *                                                                                                                                                                                                                             }
								     *                                                                                                                                                                                                                                     out<<std::endl;
								     *                                                                                                                                                                                                                                             out<<conv_layer_time2<<",";
								     *                                                                                                                                                                                                                                                     for(unsigned int j=0; j<11; j++)
								     *                                                                                                                                                                                                                                                             {
								     *                                                                                                                                                                                                                                                                         out<<conv_kernel_time2[j]<<",";
								     *                                                                                                                                                                                                                                                                                 }
								     *                                                                                                                                                                                                                                                                                         out<<std::endl;
								     *                                                                                                                                                                                                                                                                                                 out<<conv_layer_time3<<",";
								     *                                                                                                                                                                                                                                                                                                         for(unsigned int j=0; j<11; j++)
								     *                                                                                                                                                                                                                                                                                                                 {
								     *                                                                                                                                                                                                                                                                                                                             out<<conv_kernel_time3[j]<<",";
								     *                                                                                                                                                                                                                                                                                                                                     }
								     *                                                                                                                                                                                                                                                                                                                                             out<<std::endl;
								     *                                                                                                                                                                                                                                                                                                                                                     out<<conv_layer_time4<<",";
								     *                                                                                                                                                                                                                                                                                                                                                             for(unsigned int j=0; j<11; j++)
								     *                                                                                                                                                                                                                                                                                                                                                                     {
								     *                                                                                                                                                                                                                                                                                                                                                                                 out<<conv_kernel_time4[j]<<",";
								     *                                                                                                                                                                                                                                                                                                                                                                                         }
								     *                                                                                                                                                                                                                                                                                                                                                                                                 out<<std::endl;
								     *                                                                                                                                                                                                                                                                                                                                                                                                         out<<"fc layer"<<",";
								     *                                                                                                                                                                                                                                                                                                                                                                                                                 for(unsigned int j=0; j<10; j++)
								     *                                                                                                                                                                                                                                                                                                                                                                                                                         {
								     *                                                                                                                                                                                                                                                                                                                                                                                                                                     out<<fc_kernel_name[j]<<",";
								     *                                                                                                                                                                                                                                                                                                                                                                                                                                             }
								     *                                                                                                                                                                                                                                                                                                                                                                                                                                                     out<<std::endl;
								     *                                                                                                                                                                                                                                                                                                                                                                                                                                                             out<<fc_layer_time0<<",";
								     *                                                                                                                                                                                                                                                                                                                                                                                                                                                                     for(unsigned int j=0; j<10; j++)
								     *                                                                                                                                                                                                                                                                                                                                                                                                                                                                             {
								     *                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         out<<fc_kernel_time0[j]<<",";
								     *                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 }
								     *                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         out<<std::endl;
								     *                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 out<<fc_layer_time1<<",";
								     *                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         for(unsigned int j=0; j<10; j++)
								     *                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 {
								     *                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             out<<fc_kernel_time1[j]<<",";
								     *                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     }
								     *                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             out<<std::endl;
								     *                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     out<<fc_layer_time2<<",";
								     *                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             for(unsigned int j=0; j<10; j++)
								     *                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     {
								     *                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 out<<fc_kernel_time2[j]<<",";
								     *                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         }
								     *                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 out<<std::endl;
								     *                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         out.close();
							    *                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 */
								        }

	private:
			    Tensor src{};

			        Tensor weights0{},weights1{},weights2{},weights3{},weights4{},weights5{},weights6{},weights7{};
				    Tensor biases0{},biases1{},biases2{},biases3{},biases4{},biases5{},biases6{},biases7{};

				        Tensor out_conv0{};
					    Tensor out_conv1{};
					        Tensor out_conv2{};
						    Tensor out_conv3{};
						        Tensor out_conv4{};

							    Tensor out_act0{};
							        Tensor out_act1{};
								    Tensor out_act2{};
								        Tensor out_act3{};
									    Tensor out_act4{};
									        Tensor out_act5{};
										    Tensor out_act6{};

										        Tensor out_norm0{};
											    Tensor out_norm1{};

											        Tensor out_pool0{};
												    Tensor out_pool1{};
												        Tensor out_pool2{};

													    Tensor out_fc0{};
													        Tensor out_fc1{};
														    Tensor out_fc2{};

														        Tensor out_softmax{};


															    NEGEMMConvolutionLayer          conv0{};
															        NEGEMMConvolutionLayer          conv1{};
																    NEGEMMConvolutionLayer          conv2{};
																        NEGEMMConvolutionLayer          conv3{};
																	    NEGEMMConvolutionLayer          conv4{};

																	        NEFullyConnectedLayer       fc0{};
																		    NEFullyConnectedLayer       fc1{};
																		        NEFullyConnectedLayer       fc2{};

																			    NESoftmaxLayer              softmax{};

																			        NENormalizationLayer        norm0{}; 
																				    NENormalizationLayer        norm1{};


																				        NEPoolingLayer              pool0{};
																					    NEPoolingLayer              pool1{};
																					        NEPoolingLayer              pool2{};

																						    NEActivationLayer           act0{};
																						        NEActivationLayer           act1{};
																							    NEActivationLayer           act2{};
																							        NEActivationLayer           act3{};
																								    NEActivationLayer           act4{};
																								        NEActivationLayer           act5{};
																									    NEActivationLayer           act6{};

																									        /*
																										 *     double conv_layer_time0=0.f;
																										 *         double conv_layer_time1=0.f;
																										 *             double conv_layer_time2=0.f;
																										 *                 double conv_layer_time3=0.f;
																										 *                     double conv_layer_time4=0.f;
																										 *                         double fc_layer_time0=0.f;
																										 *                             double fc_layer_time1=0.f;
																										 *                                 double fc_layer_time2=0.f;
																										 *
																										 *                                     double conv_kernel_time0[11]={0.f};
																										 *                                         double conv_kernel_time1[11]={0.f};
																										 *                                             double conv_kernel_time2[11]={0.f};
																										 *                                                 double conv_kernel_time3[11]={0.f};
																										 *                                                     double conv_kernel_time4[11]={0.f};
																										 *                                                         double fc_kernel_time0[10]={0.f};
																										 *                                                             double fc_kernel_time1[10]={0.f};
																										 *                                                                 double fc_kernel_time2[10]={0.f};
																										 *                                                                     string conv_kernel_name[11]={
																										 *                                                                             "Weights Reshape",
																										 *                                                                                     "Im2Col",
																										 *                                                                                             "GEMM-Assembly-Prepare",
																										 *                                                                                                     "GEMM-Assembly-Run",
																										 *                                                                                                             "GEMM-transpose1xw",
																										 *                                                                                                                     "GEMM-interleave",
																										 *                                                                                                                             "GEMM-matrix-multiply",
																										 *                                                                                                                                     "GEMM-matrix-addition",
																										 *                                                                                                                                             "Arithmetic Addition",
																										 *                                                                                                                                                     "Col2Im",
																										 *                                                                                                                                                             "Activation"
																										 *                                                                                                                                                                 };
																										 *                                                                                                                                                                     string fc_kernel_name[10]={
																										 *                                                                                                                                                                             "Weights Reshape",
																										 *                                                                                                                                                                                     "Convert Weights",
																										 *                                                                                                                                                                                             "Flatten",
																										 *                                                                                                                                                                                                     "GEMM-Assembly-Prepare",
																										 *                                                                                                                                                                                                             "GEMM-Assembly-Run",
																										 *                                                                                                                                                                                                                     "GEMM-transpose1xw",
																										 *                                                                                                                                                                                                                             "GEMM-interleave",
																										 *                                                                                                                                                                                                                                     "GEMM-matrix-multiply",
																										 *                                                                                                                                                                                                                                             "GEMM-matrix-addition",
																										 *                                                                                                                                                                                                                                                     "Accumulate Bias",
																										 *                                                                                                                                                                                                                                                         };
																										 *                                                                                                                                                                                                                                                             
																										 *                                                                                                                                                                                                                                                                 void store_conv_kernel_time(NEGEMMConvolutionLayer &l, double* kernel_time){
																										 *                                                                                                                                                                                                                                                                         for(unsigned int j=0; j<11; j++)
																										 *                                                                                                                                                                                                                                                                                 {
																										 *                                                                                                                                                                                                                                                                                             kernel_time[j]+=(l.get_kernels_time()[j]*1000);
																										 *                                                                                                                                                                                                                                                                                                     }
																										 *                                                                                                                                                                                                                                                                                                         };
																										 *                                                                                                                                                                                                                                                                                                             void store_fc_kernel_time(NEFullyConnectedLayer &f, double* kernel_time){
																										 *                                                                                                                                                                                                                                                                                                                     for(unsigned int j=0; j<10; j++)
																										 *                                                                                                                                                                                                                                                                                                                             {
																										 *                                                                                                                                                                                                                                                                                                                                         kernel_time[j]+=(f.get_kernels_time()[j]*1000);
																										 *                                                                                                                                                                                                                                                                                                                                                 }
																										 *                                                                                                                                                                                                                                                                                                                                                     }
																										 *                                                                                                                                                                                                                                                                                                                                                         void avg_kernel_time(double* kernel_time, unsigned int length, unsigned int count)
																										 *                                                                                                                                                                                                                                                                                                                                                             {
																										 *                                                                                                                                                                                                                                                                                                                                                                     for(unsigned int j=0; j<length; j++)
																										 *                                                                                                                                                                                                                                                                                                                                                                             {
																										 *                                                                                                                                                                                                                                                                                                                                                                                         kernel_time[j]=kernel_time[j]/count;
																										 *                                                                                                                                                                                                                                                                                                                                                                                                 }
																										 *                                                                                                                                                                                                                                                                                                                                                                                                     }
																										 *                                                                                                                                                                                                                                                                                                                                                                                                         */

};

int main(int argc, char **argv)
{
	    int cpus = 8;
	        
	        for(int i=0;i<4;i++){
			        ofstream out("./lab2/alexnet/alexnet_avg_time.csv", ios::out | ios::app);
				        out<<"core= "<<(i+1)<<"s"<<std::endl;
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
																		                    printf("this process %d of running processor: %d\n", getpid(), j); 
																				                    num++;
																						                }    
																	            }
															        out.close();
																        CPPScheduler::get().set_num_threads(num);

																	        utils::run_example<NEONALEXExample>(argc, argv);
																		    }
		    
		    for(int i=4;i<8;i++){
			            /*ofstream out("./lab3/alexnet/avg_time.csv", ios::out | ios::app);*/
			            ofstream out("./lab2/alexnet/alexnet_avg_time.csv", ios::out | ios::app);
				            out<<"core= "<<i-3<<"b"<<std::endl;
					            std::cout <<"core="<<i<< std::endl;
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
																				                printf("this process %d of running processor: %d\n", getpid(), j); 
																						                num++;
																								            }    
																			        }
																            out.close();
																	            CPPScheduler::get().set_num_threads(num);
																		            utils::run_example<NEONALEXExample>(argc, argv);
																			        }
}


