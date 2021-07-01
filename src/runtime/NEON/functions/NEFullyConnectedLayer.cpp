#include "arm_compute/runtime/NEON/functions/NEFullyConnectedLayer.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Size2D.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/core/utils/quantization/AsymmHelpers.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <fstream>
using namespace std;

using namespace arm_compute;
using namespace arm_compute::misc::shape_calculator;

namespace
{
	Status validate_mm(const ITensorInfo &input, const ITensorInfo &weights, const ITensorInfo &output)
	{
		    if(is_data_type_quantized_asymmetric(input.data_type()))
			        {
					        
					        const QuantizationInfo input_quantization_info(input.quantization_info().uniform().scale, -input.quantization_info().uniform().offset);
						        const QuantizationInfo weights_quantization_info(weights.quantization_info().uniform().scale, -weights.quantization_info().uniform().offset);

							        ARM_COMPUTE_RETURN_ON_ERROR(NEGEMMLowpMatrixMultiplyCore::validate(&input.clone()->set_quantization_info(input_quantization_info),
											                                                                           &weights.clone()->set_quantization_info(weights_quantization_info),
																				                                                                              nullptr,
																													                                                                                 &output));
								    }
		        else
				    {
					            ARM_COMPUTE_RETURN_ON_ERROR(NEGEMM::validate(&input, &weights, nullptr, &output, 1.f, 0.0f, GEMMInfo(false, false, true /* Reshape weights only for the first run */)));
						        }

			    return Status{};
	}
} 

void NEFullyConnectedLayerReshapeWeights::configure(const ITensor *input, ITensor *output)
{
	    auto k = arm_compute::support::cpp14::make_unique<NETransposeKernel>();
	        k->configure(input, output);
		    _kernel = std::move(k);
}

Status NEFullyConnectedLayerReshapeWeights::validate(const ITensorInfo *input, const ITensorInfo *output)
{
	    return NETransposeKernel::validate(input, output);
}

NEFullyConnectedLayer::NEFullyConnectedLayer(std::shared_ptr<IMemoryManager> memory_manager)
	    : _memory_group(std::move(memory_manager)), _flatten_kernel(), _convert_weights(), _reshape_weights_function(), _mm_gemm(), _mm_gemmlowp(), _gemmlowp_output_stage(), _accumulate_biases_kernel(),
	          _flatten_output(), _gemmlowp_output(), _converted_weights_output(), _reshape_weights_output(), _original_weights(nullptr), _are_weights_converted(true), _are_weights_reshaped(false),
		        _is_fc_after_conv(false), _accumulate_biases(false), _is_quantized(false), _is_prepared(false),
			      _weights_reshape_time(0.f), _convert_weights_time(0.f), _flatten_time(0.f), _gemm_assembly_prepare_time(0.f), 
			            _gemm_assembly_run_time(0.f), _transpose1xw_time(0.f),
				          _interleave_time(0.f), _gemm_matrix_multiply_time(0.f), _matrix_addition_time(0.f), _accumulate_biases_time(0.f),
					        _time(), _layer_time(0), _kernel_time(), count(100), now(0)
{
}

void NEFullyConnectedLayer::configure_mm(const ITensor *input, const ITensor *weights, ITensor *output)
{
	    if(_is_quantized)
		        {
				        
				        const QuantizationInfo input_quantization_info   = input->info()->quantization_info();
					        const QuantizationInfo weights_quantization_info = weights->info()->quantization_info();

						        input->info()->set_quantization_info(QuantizationInfo(input_quantization_info.uniform().scale, -input_quantization_info.uniform().offset));
							        weights->info()->set_quantization_info(QuantizationInfo(weights_quantization_info.uniform().scale, -weights_quantization_info.uniform().offset));

								        _mm_gemmlowp.configure(input, weights, nullptr, output);

									        input->info()->set_quantization_info(input_quantization_info);
										        weights->info()->set_quantization_info(weights_quantization_info);
											    }
	        else
			    {
				            _mm_gemm.configure(input, weights, nullptr, output, 1.f, 0.0f, GEMMInfo(false, false, true /* Reshape weights only for the first run */));
					        }
}

void NEFullyConnectedLayer::configure_conv_fc(const ITensor *input, const ITensor *weights, ITensor *output)
{
	    ARM_COMPUTE_ERROR_ON((weights->info()->dimension(1) != (input->info()->dimension(0) * input->info()->dimension(1) * input->info()->dimension(2))));

	       
	        TensorShape shape_flatten = compute_flatten_shape(input->info());
		    _flatten_output.allocator()->init(input->info()->clone()->set_is_resizable(true).reset_padding().set_tensor_shape(shape_flatten));

		        _memory_group.manage(&_flatten_output);
			    _flatten_kernel.configure(input, &_flatten_output);

			        configure_mm(&_flatten_output, weights, output);

				    _flatten_output.allocator()->allocate();
}

void NEFullyConnectedLayer::configure_fc_fc(const ITensor *input, const ITensor *weights, ITensor *output)
{
	    ARM_COMPUTE_ERROR_ON(input->info()->dimension(0) != weights->info()->dimension(1));

	        configure_mm(input, weights, output);
}

void NEFullyConnectedLayer::configure(const ITensor *input, const ITensor *weights, const ITensor *biases, ITensor *output,
		                                      FullyConnectedLayerInfo fc_info)
{
	    ARM_COMPUTE_ERROR_ON_NULLPTR(input, weights, output);

	        ARM_COMPUTE_ERROR_THROW_ON(NEFullyConnectedLayer::validate(input->info(),
					                                                               weights->info(),
												                                                                      biases != nullptr ? biases->info() : nullptr,
																				                                                                     output->info(),
																												                                                                    fc_info));

		    _are_weights_converted = true;
		        _are_weights_reshaped  = fc_info.transpose_weights ? fc_info.are_weights_reshaped : true;
			    _is_fc_after_conv      = true;
			        _accumulate_biases     = false;
				    _is_quantized          = is_data_type_quantized_asymmetric(input->info()->data_type());
				        _original_weights      = weights;

					    if(_is_quantized)
						        {
								        _gemmlowp_output.allocator()->init(output->info()->clone()->set_is_resizable(true).reset_padding().set_data_type(DataType::S32));
									    }

					        if(biases != nullptr && !_is_quantized)
							    {
								            _accumulate_biases = true;

									            _accumulate_biases_kernel.configure(output, biases);
										        }

						    
						    const ITensor *weights_to_use = weights;

						        const bool is_batched_fc_layer = output->info()->dimension(1) > 1;
							    if(is_batched_fc_layer)
								        {
										        _is_fc_after_conv = (TensorShape::num_max_dimensions >= 4) && (std::equal(input->info()->tensor_shape().cbegin() + 3,
														                                                                                  input->info()->tensor_shape().cend(),
																								                                                                                    output->info()->tensor_shape().cbegin() + 1));
											    }
							        else
									    {
										            _is_fc_after_conv = input->info()->num_dimensions() > 1;
											        }

								    if(!_are_weights_reshaped)
									        {
											        _reshape_weights_function.configure(weights, &_reshape_weights_output);
												        weights_to_use = &_reshape_weights_output;
													    }

								        if(_is_fc_after_conv && (input->info()->data_layout() != fc_info.weights_trained_layout))
										    {
											            _convert_weights.configure(weights_to_use,
														                                       &_converted_weights_output,
																		                                          input->info()->tensor_shape(),
																							                                     fc_info.weights_trained_layout);

												            weights_to_use         = &_converted_weights_output;
													            _are_weights_converted = false;
														        }

									    ITensor *tmp_output = (_is_quantized) ? &_gemmlowp_output : output;
									        if(_is_fc_after_conv)
											    {
												            configure_conv_fc(input, weights_to_use, tmp_output);
													        }
										    else
											        {
													        configure_fc_fc(input, weights_to_use, tmp_output);
														    }

										        if(_is_quantized)
												    {
													            const UniformQuantizationInfo iq_info = input->info()->quantization_info().uniform();
														            const UniformQuantizationInfo wq_info = weights->info()->quantization_info().uniform();
															            const UniformQuantizationInfo oq_info = output->info()->quantization_info().uniform();

																            float multiplier = (iq_info.scale * wq_info.scale) / oq_info.scale;
																	            int   output_multiplier;
																		            int   output_shift;
																			            quantization::calculate_quantized_multiplier_less_than_one(multiplier, &output_multiplier, &output_shift);
																				            _gemmlowp_output_stage.configure(&_gemmlowp_output, biases, output, output_multiplier, output_shift, oq_info.offset);
																					            _gemmlowp_output.allocator()->allocate();
																						        }

											    _are_weights_reshaped = _are_weights_reshaped || fc_info.retain_internal_weights;
}

Status NEFullyConnectedLayer::validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output,
		                                       FullyConnectedLayerInfo fc_info)
{
	    ARM_COMPUTE_UNUSED(fc_info.retain_internal_weights);
	        ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, weights, output);
		    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QASYMM8, DataType::F16, DataType::F32);
		        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, weights, output);
			    ARM_COMPUTE_RETURN_ERROR_ON(weights->num_dimensions() > 2);

			        bool weights_reshaped = fc_info.transpose_weights ? fc_info.are_weights_reshaped : true;
				    bool is_fc_after_conv = true;
				        bool is_quantized     = is_data_type_quantized_asymmetric(input->data_type());

					    const ITensorInfo &flatten_input     = TensorInfo(input->clone()->set_is_resizable(true).reset_padding().set_tensor_shape(compute_flatten_shape(input)));
					        const ITensorInfo &reshaped_weights  = TensorInfo(weights->clone()->set_is_resizable(true).reset_padding().set_tensor_shape(compute_transposed_shape(*weights)));
						    const ITensorInfo &converted_weights = weights_reshaped ? TensorInfo(weights->clone()->set_is_resizable(true).reset_padding()) : TensorInfo(*reshaped_weights.clone());
						        const ITensorInfo &gemmlowp_output   = TensorInfo(output->clone()->set_is_resizable(true).reset_padding().set_data_type(DataType::S32));

							    if(biases != nullptr && !is_quantized)
								        {
										        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, biases);
											        ARM_COMPUTE_RETURN_ON_ERROR(NEGEMMMatrixAccumulateBiasesKernel::validate(output, biases));
												    }

							        /*
								 *      With the Fully Connected layer we can have 4 different cases:
								 *           1) Convolution layer -> Fully Connected layer without batches
								 *                2) Fully Connected layer -> Fully Connected layer without batches
								 *                    3) Convolution layer -> Fully Connected layer with batches
								 *                        4) Fully Connected layer -> Fully Connected layer with batches
								 *                                */
							        const ITensorInfo *input_to_use   = input;
								    const ITensorInfo *weights_to_use = weights;
								        const ITensorInfo *tmp_output     = (is_quantized) ? &gemmlowp_output : output;

									    const bool is_batched_fc_layer = output->dimension(1) > 1;

									        if(is_batched_fc_layer)
											    {
												            is_fc_after_conv = (TensorShape::num_max_dimensions >= 4) && (std::equal(input->tensor_shape().cbegin() + 3,
																                                                                                     input->tensor_shape().cend(),
																										                                                                                      output->tensor_shape().cbegin() + 1));
													        }
										    else
											        {
													        is_fc_after_conv = input->num_dimensions() > 1;
														    }

										        if(!weights_reshaped)
												    {
													            ARM_COMPUTE_RETURN_ON_ERROR(NEFullyConnectedLayerReshapeWeights::validate(weights, &reshaped_weights));
														            weights_to_use = &reshaped_weights;
															        }

											    if(is_fc_after_conv && (input->data_layout() != fc_info.weights_trained_layout))
												        {
														        ARM_COMPUTE_RETURN_ON_ERROR(NEConvertFullyConnectedWeights::validate(weights_to_use,
																		                                                                             &converted_weights,
																											                                                                                  input->tensor_shape(),
																																					                                                                               fc_info.weights_trained_layout));
															        weights_to_use = &converted_weights;
																    }

											        if(is_fc_after_conv)
													    {
														            ARM_COMPUTE_RETURN_ERROR_ON((weights_to_use->dimension(1) != (input->dimension(0) * input->dimension(1) * input->dimension(2))));

															            ARM_COMPUTE_RETURN_ON_ERROR(NEFlattenLayerKernel::validate(input, &flatten_input));
																            input_to_use = &flatten_input;
																	        }
												    else
													        {
															        ARM_COMPUTE_RETURN_ERROR_ON(input->dimension(0) != weights_to_use->dimension(1));
																    }
												        ARM_COMPUTE_RETURN_ON_ERROR(validate_mm(*input_to_use, *weights_to_use, *tmp_output));

													    if(is_quantized)
														        {
																        const UniformQuantizationInfo iq_info    = input->quantization_info().uniform();
																	        const UniformQuantizationInfo wq_info    = weights->quantization_info().uniform();
																		        const UniformQuantizationInfo oq_info    = output->quantization_info().uniform();
																			        const float                   multiplier = iq_info.scale * wq_info.scale / oq_info.scale;

																				        ARM_COMPUTE_UNUSED(multiplier);
																					        ARM_COMPUTE_RETURN_ERROR_ON(multiplier > 1.0f);
																						        ARM_COMPUTE_RETURN_ON_ERROR(NEGEMMLowpQuantizeDownInt32ToUint8ScaleByFixedPoint::validate(&gemmlowp_output, biases, output));
																							    }

													        return Status{};
}

void NEFullyConnectedLayer::run()
{
	    
	    ofstream out("./lab2/alexnet/alexnet_avg_time.csv", ios::out | ios::app);
	        /*ofstream out("./lab1/1/kernel_exec_time.csv", ios::out | ios::app);*/
	        /*ofstream out("./lab3/kernel_exec_time.csv", ios::out | ios::app);*/
	        for(unsigned int i=0; i<10; i++)
			    {
				            _time[i]=0;
					        }
		    auto b=std::chrono::high_resolution_clock::now();
		        prepare();

			    MemoryGroupResourceScope scope_mg(_memory_group);

			        
			        if(_is_fc_after_conv)
					    {
						            auto begin9=std::chrono::high_resolution_clock::now();
							            NEScheduler::get().schedule(&_flatten_kernel, Window::DimY);
								            auto end9=std::chrono::high_resolution_clock::now();
									            _flatten_time=std::chrono::duration_cast<std::chrono::duration<double>>(end9- begin9).count();
										            _time[2]=_flatten_time;
											        }

				    if(_is_quantized)
					        {
							        _mm_gemmlowp.run();
								    }
				        else
						    {
							            
							            _mm_gemm.run();
								            _gemm_assembly_prepare_time=_mm_gemm.get_assembly_prepare_time();_time[3]=_gemm_assembly_prepare_time;
									            _gemm_assembly_run_time=_mm_gemm.get_assembly_run_time();_time[4]=_gemm_assembly_run_time;
										            _transpose1xw_time=_mm_gemm.get_transpose1xw_kernel_time();_time[5]=_transpose1xw_time;
											            _interleave_time=_mm_gemm.get_interleave_kernel_time();_time[6]=_interleave_time;
												            _gemm_matrix_multiply_time=_mm_gemm.get_matrix_multiply_kernel_time();_time[7]=_gemm_matrix_multiply_time;
													            _matrix_addition_time=_mm_gemm.get_matrix_addition_kernel_time();_time[8]=_matrix_addition_time;

														        }


					    if(_is_quantized)
						        {
								        _gemmlowp_output_stage.run();
									    }
					        else
							    {
								            if(_accumulate_biases)
										            {
												                auto begin10=std::chrono::high_resolution_clock::now();
														            NEScheduler::get().schedule(&_accumulate_biases_kernel, Window::DimY);
															                auto end10=std::chrono::high_resolution_clock::now();
																	            _accumulate_biases_time=std::chrono::duration_cast<std::chrono::duration<double>>(end10 - begin10).count();
																		                _time[9]=_accumulate_biases_time;
																				        }
									        }
						    auto e=std::chrono::high_resolution_clock::now();
						        double ttime=std::chrono::duration_cast<std::chrono::duration<double>>(e - b).count(); 
							    if(now>0)
								        {
										        _layer_time+=(ttime*1000);
											        for(unsigned int k=0; k<10; k++)
													        {
															            _kernel_time[k]+=_time[k]*1000;
																            }
												    }
							        if(now==(count-1))
									    {
										            _layer_time=_layer_time/(count-1);
											            for(unsigned int k=0; k<10; k++)
													            {
															                _kernel_time[k]=_kernel_time[k]/(count-1);
																	        }
												            /*
													     *         out<<_layer_time<<",";
													     *                 for(unsigned int j=0; j<10; j++)
													     *                         {
													     *                                     out<<_kernel_time[j]<<",";
													     *                                             }
													     *                                                     */
												            /*
													     *         out<<_kernel_time[2]<<","<<_kernel_time[4]<<","<<_kernel_time[9];
													     *                 */
												            out<<"fc_layer"<<","<<_layer_time;
													            out<<std::endl;
														            out.close();
															        }
								    now++;

}

void NEFullyConnectedLayer::prepare()
{
	    if(!_is_prepared)
		        {
				        ARM_COMPUTE_ERROR_ON(!_original_weights->is_used());
					        
					        auto release_unused = [](Tensor * w)
							        {
									            if(!w->is_used())
											                {
														                w->allocator()->free();
																            }
										            };

						        const ITensor *cur_weights = _original_weights;

							        if(!_are_weights_reshaped)
									        {
											            _reshape_weights_output.allocator()->allocate();
												                auto begin1=std::chrono::high_resolution_clock::now();
														            _reshape_weights_function.run();
															                auto end1=std::chrono::high_resolution_clock::now();
																	            _weights_reshape_time=std::chrono::duration_cast<std::chrono::duration<double>>(end1 - begin1).count();
																		                _time[0]=_weights_reshape_time;
																				            cur_weights->mark_as_unused();
																					                cur_weights           = &_reshape_weights_output;
																							            _are_weights_reshaped = true;
																								            }

								        if(!_are_weights_converted)
										        {
												            _converted_weights_output.allocator()->allocate();
													                auto begin2=std::chrono::high_resolution_clock::now();
															            _convert_weights.run();
																                auto end2=std::chrono::high_resolution_clock::now();
																		            _convert_weights_time=std::chrono::duration_cast<std::chrono::duration<double>>(end2 - begin2).count();
																			                _time[1]=_convert_weights_time;
																					            cur_weights->mark_as_unused();
																						                _are_weights_converted = true;
																								        }

									        release_unused(&_reshape_weights_output);

										      
										        /*
											 *         if(!_is_quantized)
											 *                 {
											 *                             _mm_gemm.prepare();
											 *                                     }
											 *                                             */
										        release_unused(&_reshape_weights_output);
											        release_unused(&_converted_weights_output);

												        _is_prepared = true;
													    }
}
double* NEFullyConnectedLayer::get_kernels_time()
{
	    return _time;
}
