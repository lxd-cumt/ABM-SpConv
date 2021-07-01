/*
 *  * Copyright (c) 2017-2019 ARM Limited.
 *   *
 *    * SPDX-License-Identifier: MIT
 *     *
 *      * Permission is hereby granted, free of charge, to any person obtaining a copy
 *       * of this software and associated documentation files (the "Software"), to
 *        * deal in the Software without restriction, including without limitation the
 *         * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 *          * sell copies of the Software, and to permit persons to whom the Software is
 *           * furnished to do so, subject to the following conditions:
 *            *
 *             * The above copyright notice and this permission notice shall be included in all
 *              * copies or substantial portions of the Software.
 *               *
 *                * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *                 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *                  * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *                   * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *                    * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *                     * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *                      * SOFTWARE.
 *                       */
#include "arm_compute/runtime/NEON/functions/NEGEMMConvolutionLayer.h"

#include "arm_compute/core/Size2D.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/core/utils/quantization/AsymmHelpers.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "support/ToolchainSupport.h"

#include <cmath>
#include <set>
#include <tuple>
#include <chrono>
#include <iostream>
#include <fstream>
using namespace std;


using namespace arm_compute;
using namespace arm_compute::misc::shape_calculator;

NEConvolutionLayerReshapeWeights::NEConvolutionLayerReshapeWeights()
	    : _weights_reshape_kernel()
{
}

void NEConvolutionLayerReshapeWeights::configure(const ITensor *weights, const ITensor *biases, ITensor *output)
{
	        ARM_COMPUTE_ERROR_ON_NULLPTR(weights, output);
		    ARM_COMPUTE_ERROR_THROW_ON(NEConvolutionLayerReshapeWeights::validate(weights->info(),
					                                                                              (biases != nullptr) ? biases->info() : nullptr,
														                                                                                output->info()));

		        const bool     append_biases = (biases != nullptr) && !is_data_type_quantized_asymmetric(weights->info()->data_type());
			    const ITensor *biases_to_use = (append_biases) ? biases : nullptr;

			        _weights_reshape_kernel.configure(weights, biases_to_use, output);

				    output->info()->set_quantization_info(weights->info()->quantization_info());
}

Status NEConvolutionLayerReshapeWeights::validate(const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output)
{
	    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(weights);
	        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(weights, 1, DataType::QASYMM8, DataType::F16, DataType::F32);
		    ARM_COMPUTE_RETURN_ERROR_ON(weights->num_dimensions() > 4);

		        if(biases != nullptr)
				    {
					            const int idx_kernels = get_data_layout_dimension_index(weights->data_layout(), DataLayoutDimension::BATCHES);
						            ARM_COMPUTE_RETURN_ERROR_ON(is_data_type_quantized_asymmetric(weights->data_type()));
							            ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(weights, biases);
								            ARM_COMPUTE_RETURN_ERROR_ON(biases->dimension(0) != weights->dimension(idx_kernels));
									            ARM_COMPUTE_RETURN_ERROR_ON(biases->num_dimensions() > 1);
										        }

			    if((output != nullptr) && (output->total_size() != 0))
				        {
						        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(weights, output);

							        NEWeightsReshapeKernel::validate(weights, biases, output);
								    }

			        return Status{};
}

void NEConvolutionLayerReshapeWeights::run()
{
	    NEScheduler::get().schedule(&_weights_reshape_kernel, 3);
}


NEGEMMConvolutionLayer::NEGEMMConvolutionLayer(const std::shared_ptr<IMemoryManager> &memory_manager)
	    : _memory_group(memory_manager), _reshape_weights(), _im2col_kernel(), _mm_gemm(memory_manager), _mm_gemmlowp(memory_manager), _col2im_kernel(), _activationlayer_function(), _add_bias_kernel(),
	          _reshape_layer(), _original_weights(nullptr), _im2col_output(), _weights_reshaped(), _gemm_output(), _tmp_output(), _data_layout(DataLayout::NCHW), _append_bias(false), _skip_im2col(false),
		        _skip_col2im(false), _is_quantized(false), _is_activationlayer_enabled(false), _is_prepared(false),
			      _weights_reshape_time(0.f),_im2col_time(0.f), 
			            _matrix_a_reshape_time(0.f), _matrix_a_reduction_time(0.f), _matrix_b_reshape_time(0.f), _matrix_b_reduction_time(0.f),
				          _gemmlowp_assembly_dispatch_time(0.f), _gemmlowp_matrix_multiply_time(0.f), _offset_contribution_time(0.f), _offset_contribution_output_stage_time(0.f),
					        _gemm_assembly_prepare_time(0.f),_gemm_assembly_run_time(0.f), _transpose1xw_time(0.f), _interleave_time(0.f), _gemm_matrix_multiply_time(0.f), _matrix_addition_time(0.f), 
						      _arithmetic_addition_time(0.f), _col2im_time(0.f), _reshape_layer_time(0.f), _activation_layer_time(0.f), _time(), _layer_time(0), _kernel_time(),  count(100), now(0)
{
}

void NEGEMMConvolutionLayer::configure_mm(const ITensor *input, const ITensor *weights, const ITensor *biases, ITensor *output, const ActivationLayerInfo &act_info, int gemm_3d_depth)
{
	    ARM_COMPUTE_ERROR_ON_NULLPTR(input, weights);
	        ARM_COMPUTE_ERROR_THROW_ON(validate_mm(input->info(), weights->info(), biases == nullptr ? nullptr : biases->info(), output == nullptr ? nullptr : output->info(), act_info, gemm_3d_depth,
					                                           _skip_im2col));

		    const GEMMInfo &gemm_info = GEMMInfo(false, false, true /* Reshape weights only for the first run */,
				                                             gemm_3d_depth, _skip_im2col /* Reinterpret the input as 3D if im2col is skipped */);

		        if(_is_quantized)
				    {
					                            const UniformQuantizationInfo iqinfo = input->info()->quantization_info().uniform();
								            const UniformQuantizationInfo wqinfo = weights->info()->quantization_info().uniform();

									            input->info()->set_quantization_info(QuantizationInfo(iqinfo.scale, -iqinfo.offset));
										            weights->info()->set_quantization_info(QuantizationInfo(wqinfo.scale, -wqinfo.offset));

											            const UniformQuantizationInfo oqinfo = (output->info()->total_size() == 0) ? iqinfo : output->info()->quantization_info().uniform();

												            float multiplier = iqinfo.scale * wqinfo.scale / oqinfo.scale;
													            int   output_multiplier;
														            int   output_shift;
															            quantization::calculate_quantized_multiplier_less_than_one(multiplier, &output_multiplier, &output_shift);

																                    int min_activation = 0;
																		            int max_activation = 255;

																			            const std::set<ActivationLayerInfo::ActivationFunction> supported_acts = { ActivationLayerInfo::ActivationFunction::RELU,
																					                                                                                       ActivationLayerInfo::ActivationFunction::BOUNDED_RELU,
																															                                                                                          ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU
																																											                                                                                   };
																				            if(_is_activationlayer_enabled && supported_acts.count(act_info.activation()) != 0)
																						            {
																								                const int a_const_int = quantize_qasymm8(act_info.a(), oqinfo);
																										            const int b_const_int = quantize_qasymm8(act_info.b(), oqinfo);

																											                min_activation = act_info.activation() != ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU ? oqinfo.offset : b_const_int;
																													            max_activation = act_info.activation() == ActivationLayerInfo::ActivationFunction::RELU ? 255 : a_const_int;

																														                _is_activationlayer_enabled = false;
																																        }

																					            GEMMLowpOutputStageInfo output_info;
																						            output_info.type                = GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT;
																							            output_info.gemmlowp_offset     = oqinfo.offset;
																								            output_info.gemmlowp_multiplier = output_multiplier;
																									            output_info.gemmlowp_shift      = output_shift;
																										            output_info.gemmlowp_min_bound  = min_activation;
																											            output_info.gemmlowp_max_bound  = max_activation;

																												            _mm_gemmlowp.configure(input, weights, biases, output, GEMMInfo(false, false, true, gemm_3d_depth, _skip_im2col, false, output_info));

																													                    input->info()->set_quantization_info(QuantizationInfo(iqinfo.scale, iqinfo.offset));
																															            weights->info()->set_quantization_info(QuantizationInfo(wqinfo.scale, wqinfo.offset));
																																        }
			    else
				        {
						                _mm_gemm.configure(input, weights, nullptr, output, 1.0f, 0.0f, gemm_info);
								    }
}

Status NEGEMMConvolutionLayer::validate_mm(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const ActivationLayerInfo &act_info,
		                                           int gemm_3d_depth, bool skip_im2col)
{
	    const bool is_quantized          = is_data_type_quantized_asymmetric(input->data_type());
	        const bool is_activation_enabled = act_info.enabled();

		    const GEMMInfo &gemm_info = GEMMInfo(false, false, true /* Reshape weights only for the first run */,
				                                             gemm_3d_depth, skip_im2col /* Reinterpret the input as 3D if im2col is skipped */);
		        if(is_quantized)
				    {
					                            const UniformQuantizationInfo iqinfo = input->quantization_info().uniform();
								            const UniformQuantizationInfo wqinfo = weights->quantization_info().uniform();

									            std::unique_ptr<ITensorInfo> input_qa   = input->clone();
										            std::unique_ptr<ITensorInfo> weights_qa = weights->clone();
											            input_qa->set_quantization_info(QuantizationInfo(iqinfo.scale, -iqinfo.offset));
												            weights_qa->set_quantization_info(QuantizationInfo(wqinfo.scale, -wqinfo.offset));

													            const UniformQuantizationInfo oqinfo = (output->total_size() == 0) ? iqinfo : output->quantization_info().uniform();

														            float multiplier = iqinfo.scale * wqinfo.scale / oqinfo.scale;
															            int   output_multiplier;
																            int   output_shift;
																	            ARM_COMPUTE_RETURN_ON_ERROR(quantization::calculate_quantized_multiplier_less_than_one(multiplier, &output_multiplier, &output_shift));

																		                    int min_activation = 0;
																				            int max_activation = 255;

																					            const std::set<ActivationLayerInfo::ActivationFunction> supported_acts = { ActivationLayerInfo::ActivationFunction::RELU,
																							                                                                                       ActivationLayerInfo::ActivationFunction::BOUNDED_RELU,
																																	                                                                                          ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU
																																													                                                                                   };
																						            if(is_activation_enabled && supported_acts.count(act_info.activation()) != 0)
																								            {
																										                const int a_const_int = quantize_qasymm8(act_info.a(), oqinfo);
																												            const int b_const_int = quantize_qasymm8(act_info.b(), oqinfo);

																													                min_activation = act_info.activation() != ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU ? oqinfo.offset : b_const_int;
																															            max_activation = act_info.activation() == ActivationLayerInfo::ActivationFunction::RELU ? 255 : a_const_int;
																																            }

																							            GEMMLowpOutputStageInfo output_info;
																								            output_info.type                = GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT;
																									            output_info.gemmlowp_offset     = oqinfo.offset;
																										            output_info.gemmlowp_multiplier = output_multiplier;
																											            output_info.gemmlowp_shift      = output_shift;
																												            output_info.gemmlowp_min_bound  = min_activation;
																													            output_info.gemmlowp_max_bound  = max_activation;

																														                    return NEGEMMLowpMatrixMultiplyCore::validate(input_qa.get(), weights_qa.get(), biases, output, GEMMInfo(false, false, true, gemm_3d_depth, skip_im2col, false, output_info));
																																        }
			    else
				        {
						                return NEGEMM::validate(input, weights, nullptr, output, 1.0f, 0.0f, gemm_info);
								    }
}

Status NEGEMMConvolutionLayer::validate_gemm3d(const ITensorInfo *input_info, const ActivationLayerInfo &act_info, int gemm_3d_depth, bool skip_im2col)
{
	    const DataType     data_type = input_info->data_type();
	        const unsigned int mult_y    = skip_im2col ? 1U : gemm_3d_depth;
		    const unsigned int mult_z    = skip_im2col ? gemm_3d_depth : 1U;

		            const TensorInfo dummy_input_info(TensorShape(4U, 4U * mult_y, 1U * mult_z), 1, data_type, input_info->quantization_info());
			        const TensorInfo dummy_weights_info(TensorShape(4U, 4U), 1, data_type);
				    const TensorInfo dummy_output_info(TensorShape(4U, 4U, gemm_3d_depth), 1, data_type, input_info->quantization_info());

				        return validate_mm(&dummy_input_info, &dummy_weights_info, nullptr, &dummy_output_info, act_info, gemm_3d_depth, skip_im2col);
}

void NEGEMMConvolutionLayer::configure(const ITensor *input, const ITensor *weights, const ITensor *biases, ITensor *output, const PadStrideInfo &conv_info, const WeightsInfo &weights_info,
		                                       const Size2D &dilation, const ActivationLayerInfo &act_info, unsigned int num_groups)
{
	    ARM_COMPUTE_ERROR_ON_NULLPTR(input, weights, output);
	        ARM_COMPUTE_UNUSED(num_groups);
		    ARM_COMPUTE_ERROR_THROW_ON(NEGEMMConvolutionLayer::validate(input->info(),
					                                                                    weights->info(),
													                                                                    biases != nullptr ? biases->info() : nullptr,
																					                                                                    output->info(),
																													                                                                    conv_info,
																																					                                                                    weights_info,
																																													                                                                    dilation,
																																																					                                                                    act_info,
																																																													                                                                    num_groups));

		        const DataType   data_type   = input->info()->data_type();
			    const DataLayout data_layout = input->info()->data_layout();
			        const int        idx_width   = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
				    const int        idx_height  = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);
				        const int        idx_kernels = get_data_layout_dimension_index(data_layout, DataLayoutDimension::BATCHES);

					    const unsigned int kernel_width  = weights->info()->dimension(idx_width);
					        const unsigned int kernel_height = weights->info()->dimension(idx_height);

						    _is_prepared                = weights_info.retain_internal_weights();
						        _original_weights           = weights;
							    _is_quantized               = is_data_type_quantized_asymmetric(input->info()->data_type());
							        _data_layout                = data_layout;
								    _skip_im2col                = (data_layout == DataLayout::NHWC && kernel_width == 1 && kernel_height == 1 && conv_info.stride().first == 1 && conv_info.stride().second == 1);
								        _append_bias                = (biases != nullptr) && (!_is_quantized);
									    _is_activationlayer_enabled = act_info.enabled();

									        const ITensor *gemm_input_to_use  = input;
										    ITensor       *gemm_output_to_use = output;

										            unsigned int conv_w = 0;
											        unsigned int conv_h = 0;
												    std::tie(conv_w, conv_h) = scaled_dimensions(input->info()->dimension(idx_width),
														                                                     input->info()->dimension(idx_height),
																				                                                      kernel_width,
																										                                                       kernel_height,
																																                                                        conv_info,
																																							                                                 dilation);

												            if(data_layout == DataLayout::NHWC)
														        {
																        _skip_col2im = bool(validate_gemm3d(input->info(), act_info, conv_h, true));
																	                if(!_skip_col2im)
																				        {
																						            _skip_im2col = false;
																							            }
																			    }
													        else
															    {
																            _skip_col2im = false;
																	        }

														    const ITensor *biases_to_use = (_append_bias && !_skip_im2col) ? biases : nullptr;

														            unsigned int stride_x = 0;
															        unsigned int stride_y = 0;
																    std::tie(stride_x, stride_y) = conv_info.stride();

																        unsigned int mat_weights_cols = weights->info()->dimension(idx_kernels);

																	            _reshape_weights.configure(weights, biases_to_use, &_weights_reshaped);

																		            if(!_skip_im2col)
																				        {
																						        _memory_group.manage(&_im2col_output);

																							                _im2col_kernel.configure(input, &_im2col_output, Size2D(kernel_width, kernel_height), conv_info, _append_bias, dilation);

																									                gemm_input_to_use = &_im2col_output;
																											    }
																			        else if(_append_bias)
																					    {
																						                    _add_bias_kernel.configure(output, biases, output, ConvertPolicy::SATURATE);
																								        }

																				        if(!_skip_col2im)
																						    {
																							            TensorShape shape_gemm;

																								                    shape_gemm = _im2col_output.info()->tensor_shape();
																										            shape_gemm.set(0, mat_weights_cols);
																											            shape_gemm.set(1, conv_w * conv_h);

																												                    TensorInfo info_gemm(shape_gemm, 1, data_type);
																														            info_gemm.set_quantization_info(output->info()->quantization_info()).set_data_layout(input->info()->data_layout());
																															            _gemm_output.allocator()->init(info_gemm);
																																            _memory_group.manage(&_gemm_output);

																																	                    gemm_output_to_use = &_gemm_output;
																																			        }

																					            const unsigned int gemm_3d_depth = _skip_col2im ? conv_h : 0;
																						        configure_mm(gemm_input_to_use, &_weights_reshaped, biases, gemm_output_to_use, act_info, gemm_3d_depth);

																							    if(!_skip_im2col)
																								        {
																										        _im2col_output.allocator()->allocate();
																											    }

																							        if(!_skip_col2im)
																									    {
																										            if(_data_layout == DataLayout::NCHW)
																												            {
																														                            _col2im_kernel.configure(gemm_output_to_use, output, Size2D(conv_w, conv_h));
																																	            }
																											            else
																													            {
																															                            _reshape_layer.configure(gemm_output_to_use, output);
																																		            }
																												        }

																								    if(_is_quantized && !_skip_col2im)
																									        {
																											        _tmp_output.allocator()->allocate();
																												    }

																								        if(!_skip_col2im || _is_quantized)
																										    {
																											            _gemm_output.allocator()->allocate();
																												        }

																									    ARM_COMPUTE_ERROR_ON_MSG((output->info()->dimension(idx_width) != conv_w) || (output->info()->dimension(idx_height) != conv_h),
																											                                 "Output shape does not match the expected one");

																									            if(_is_activationlayer_enabled)
																											        {
																													        _activationlayer_function.configure(output, nullptr, act_info);
																														    }

																										        ARM_COMPUTE_UNUSED(weights_info);
}

Status NEGEMMConvolutionLayer::validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const PadStrideInfo &conv_info,
		                                        const WeightsInfo &weights_info, const Size2D &dilation, const ActivationLayerInfo &act_info, unsigned int num_groups)
{
	    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, weights, output);
	        ARM_COMPUTE_RETURN_ERROR_ON_MSG(weights_info.are_reshaped(), "Weights already reshaped are not supported!");
		    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QASYMM8, DataType::F16, DataType::F32);
		        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, weights);
			    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_LAYOUT(input, weights);
			        ARM_COMPUTE_RETURN_ERROR_ON_MSG(num_groups > 1, "Grouping (num_groups != 1) is not supported on NEON");

				    const DataLayout data_layout = input->data_layout();
				        const DataType   data_type   = input->data_type();
					    const int        idx_width   = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
					        const int        idx_height  = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);
						    const int        idx_channel = get_data_layout_dimension_index(data_layout, DataLayoutDimension::CHANNEL);
						        const int        idx_kernels = get_data_layout_dimension_index(data_layout, DataLayoutDimension::BATCHES);

							    const unsigned int kernel_width  = weights->dimension(idx_width);
							        const unsigned int kernel_height = weights->dimension(idx_height);

								    TensorInfo         im2col_reshaped_info{};
								        TensorInfo         info_gemm{};
									    TensorInfo         tmp_info{};
									        TensorInfo         weights_reshaped_info{};
										    const ITensorInfo *gemm_input_to_use  = input;
										        const ITensorInfo *gemm_output_to_use = output;
											    const ITensorInfo *weights_to_use     = weights;

											        const bool is_quantized          = is_data_type_quantized_asymmetric(data_type);
												    const bool append_bias           = (biases != nullptr) && (!is_quantized);
												        bool       skip_im2col           = (data_layout == DataLayout::NHWC && kernel_width == 1 && kernel_height == 1 && conv_info.stride().first == 1 && conv_info.stride().second == 1);
													    bool       is_activation_enabled = act_info.enabled();

													            unsigned int conv_w = 0;
														        unsigned int conv_h = 0;

															    std::tie(conv_w, conv_h) = scaled_dimensions(input->dimension(idx_width),
																	                                                     input->dimension(idx_height),
																							                                                      kernel_width,
																													                                                       kernel_height,
																																			                                                        conv_info,
																																										                                                 dilation);

															            bool skip_col2im = false;
																        if(data_layout == DataLayout::NHWC)
																		    {
																			            skip_col2im = bool(validate_gemm3d(input, act_info, conv_h, true));
																				                    if(!skip_col2im)
																							            {
																									                skip_im2col = false;
																											        }
																						        }

																	    if(skip_col2im)
																		        {
																				                if(!bool(validate_gemm3d(input, act_info, conv_h, skip_im2col)))
																							        {
																									            skip_im2col = false;
																										                skip_col2im = false;
																												        }
																						    }

																	        const unsigned     bias_element  = (append_bias && !skip_im2col) ? 1 : 0;
																		    const ITensorInfo *biases_to_use = (append_bias && !skip_im2col) ? biases : nullptr;

																		        ARM_COMPUTE_RETURN_ERROR_ON(weights->dimension(idx_channel) != input->dimension(idx_channel));
																			    ARM_COMPUTE_RETURN_ERROR_ON(weights->num_dimensions() > 4);

																			            if(biases != nullptr)
																					        {
																							        if(is_quantized)
																									        {
																											            ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(biases, 1, DataType::S32);
																												            }
																								        else
																										        {
																												            ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, biases);
																													            }
																									        ARM_COMPUTE_RETURN_ERROR_ON(biases->dimension(0) != weights->dimension(idx_kernels));
																										        ARM_COMPUTE_RETURN_ERROR_ON(biases->num_dimensions() > 1);
																											    }

																				        if(act_info.enabled())
																						    {
																							            ARM_COMPUTE_ERROR_ON(act_info.b() > act_info.a());
																								        }

																					    unsigned int mat_weights_cols = weights->dimension(idx_kernels);
																					        unsigned int mat_weights_rows = weights->dimension(idx_width) * weights->dimension(idx_height) * weights->dimension(idx_channel) + bias_element;

																						        ARM_COMPUTE_RETURN_ON_ERROR(NEConvolutionLayerReshapeWeights::validate(weights, biases_to_use, nullptr));
																							    weights_reshaped_info = TensorInfo(compute_weights_reshaped_shape(*weights, (append_bias && !skip_im2col)), 1, data_type);
																							        weights_reshaped_info.set_quantization_info(weights->quantization_info());
																								    weights_to_use        = &weights_reshaped_info;

																								        if(!skip_im2col)
																										    {
																											                                    TensorShape shape_im2col = input->tensor_shape();
																															            shape_im2col.set(0, mat_weights_rows);
																																            shape_im2col.set(1, conv_w * conv_h);
																																	            shape_im2col.set(2, 1);

																																		            im2col_reshaped_info = TensorInfo(shape_im2col, 1, data_type);
																																			            im2col_reshaped_info.set_quantization_info(input->quantization_info());

																																				            ARM_COMPUTE_RETURN_ON_ERROR(NEIm2ColKernel::validate(input, &im2col_reshaped_info, Size2D(kernel_width, kernel_height), conv_info, append_bias, dilation));
																																					            gemm_input_to_use = &im2col_reshaped_info;
																																						        }
																									    else if(append_bias)
																										        {
																												                ARM_COMPUTE_RETURN_ON_ERROR(NEArithmeticAdditionKernel::validate(output, biases, output, ConvertPolicy::SATURATE));
																														    }

																									            if(!skip_col2im)
																											        {
																													        TensorShape shape_gemm = gemm_input_to_use->tensor_shape();
																														        shape_gemm.set(0, mat_weights_cols);
																															        shape_gemm.set(1, conv_w * conv_h);
																																        info_gemm = TensorInfo(shape_gemm, 1, data_type);
																																	    }
																										        else
																												    {
																													            info_gemm = TensorInfo(output->tensor_shape(), 1, data_type);
																														        }
																											    info_gemm.set_quantization_info(output->quantization_info()).set_data_layout(input->data_layout());
																											        gemm_output_to_use = &info_gemm;
																												    ARM_COMPUTE_RETURN_ON_ERROR(validate_mm(gemm_input_to_use, weights_to_use, biases, gemm_output_to_use, act_info, skip_col2im ? conv_h : 0, skip_im2col));

																												            if(!skip_col2im && (data_layout == DataLayout::NCHW))
																														        {
																																        ARM_COMPUTE_RETURN_ON_ERROR(NECol2ImKernel::validate(gemm_output_to_use, output, Size2D(conv_w, conv_h)));
																																	    }

																													            if(is_activation_enabled)
																															        {
																																	        ARM_COMPUTE_RETURN_ON_ERROR(NEActivationLayer::validate(output, nullptr, act_info));
																																		    }

																														        return Status{};
}

void NEGEMMConvolutionLayer::run()
{
	    
	    ofstream out("./lab2/alexnet/alexnet_avg_time.csv", ios::out | ios::app);
	        
	        /*ofstream out("./lab1/1/kernel_exec_time.csv", ios::out | ios::app);*/
	        /*ofstream out("./lab3/kernel_exec_time.csv", ios::out | ios::app);*/
	        for (unsigned int i=0; i<20; i++)
			    {
				            _time[i]=0.f;
					        }
		    auto b=std::chrono::high_resolution_clock::now();
		        prepare();
			    
			    MemoryGroupResourceScope scope_mg(_memory_group);

			        if(!_skip_im2col)
					    {
						                    unsigned int y_dim = get_data_layout_dimension_index(_data_layout, DataLayoutDimension::HEIGHT);
								                    auto begin2=std::chrono::high_resolution_clock::now();
										            NEScheduler::get().schedule(&_im2col_kernel, y_dim);
											            auto end2=std::chrono::high_resolution_clock::now();
												            _im2col_time=std::chrono::duration_cast<std::chrono::duration<double>>(end2 - begin2).count();
													            _time[1]=_im2col_time;
														        }

				        if(_is_quantized)
						    {
							                    _mm_gemmlowp.run(); 
									                    _matrix_a_reshape_time=_mm_gemmlowp.get_matrix_a_reshape_kernel_time();_time[12]=_matrix_a_reshape_time;
											                    _matrix_a_reduction_time=_mm_gemmlowp.get_matrix_a_reduction_kernel_time();_time[13]=_matrix_a_reduction_time;
													                    _matrix_b_reshape_time=_mm_gemmlowp.get_matrix_b_reshape_kernel_time();_time[14]=_matrix_b_reshape_time;
															                    _matrix_b_reduction_time=_mm_gemmlowp.get_matrix_b_reduction_kernel_time();_time[15]=_matrix_b_reduction_time;
																	                    _gemmlowp_assembly_dispatch_time=_mm_gemmlowp.get_assembly_dispatch_kernel_time();_time[16]=_gemmlowp_assembly_dispatch_time;
																			                    _gemmlowp_matrix_multiply_time=_mm_gemmlowp.get_matrix_multiply_kernel_time();_time[17]=_gemmlowp_matrix_multiply_time;
																					                    _offset_contribution_time=_mm_gemmlowp.get_offset_contribution_kernel_time();_time[18]=_offset_contribution_time;
																							                    _offset_contribution_output_stage_time=_mm_gemmlowp.get_offset_contribution_output_stage_kernel_time();_time[19]=_offset_contribution_output_stage_time;
																									        }
					    else
						        {
								                _mm_gemm.run();
										                _gemm_assembly_prepare_time=_mm_gemm.get_assembly_prepare_time();
												                _gemm_assembly_run_time=_mm_gemm.get_assembly_run_time();
														                _transpose1xw_time=_mm_gemm.get_transpose1xw_kernel_time();
																                _interleave_time=_mm_gemm.get_interleave_kernel_time();
																		                _gemm_matrix_multiply_time=_mm_gemm.get_matrix_multiply_kernel_time();
																				                _matrix_addition_time=_mm_gemm.get_matrix_addition_kernel_time();
																						                _time[2]=_gemm_assembly_prepare_time;_time[3]=_gemm_assembly_run_time;
																								                _time[4]=_transpose1xw_time;_time[5]=_interleave_time;
																										                _time[6]=_gemm_matrix_multiply_time;_time[7]=_matrix_addition_time;
																												    }

					        if(_skip_im2col && _append_bias)
							    {
								            auto begin9=std::chrono::high_resolution_clock::now();
									            NEScheduler::get().schedule(&_add_bias_kernel, Window::DimY);
										            auto end9=std::chrono::high_resolution_clock::now();
											            _arithmetic_addition_time=std::chrono::duration_cast<std::chrono::duration<double>>(end9 - begin9).count(); 
												            _time[8]=_arithmetic_addition_time;
													        }

						        if(!_skip_col2im)
								    {
									            if(_data_layout == DataLayout::NCHW)
											            {
													                auto begin10=std::chrono::high_resolution_clock::now();
															            NEScheduler::get().schedule(&_col2im_kernel, Window::DimY);
																                auto end10=std::chrono::high_resolution_clock::now();
																		            _col2im_time=std::chrono::duration_cast<std::chrono::duration<double>>(end10 - begin10).count(); 
																			                _time[9]=_col2im_time;
																					        }
										            else
												            {
														                auto begin11=std::chrono::high_resolution_clock::now();
																            _reshape_layer.run();
																	                auto end11=std::chrono::high_resolution_clock::now();
																			            _reshape_layer_time=std::chrono::duration_cast<std::chrono::duration<double>>(end11 - begin11).count();
																				                _time[11]= _reshape_layer_time;

																						        }
											        }

							    if(_is_activationlayer_enabled)
								        {
										        auto begin12=std::chrono::high_resolution_clock::now();
											        _activationlayer_function.run();
												        auto end12=std::chrono::high_resolution_clock::now();
													        _activation_layer_time=std::chrono::duration_cast<std::chrono::duration<double>>(end12 - begin12).count(); 
														        _time[10]=_activation_layer_time;
															    }
							        auto e=std::chrono::high_resolution_clock::now();
								    double ttime=std::chrono::duration_cast<std::chrono::duration<double>>(e - b).count(); 
								        if(now>0)
										    {
											            _layer_time+=(ttime*1000);
												            for(unsigned int k=0; k<11; k++)
														            {
																                _kernel_time[k]+=_time[k]*1000;
																		        }
													        }
									    if(now==(count-1))
										        {
												        _layer_time=_layer_time/(count-1);
													        for(unsigned int k=0; k<11; k++)
															        {
																	            _kernel_time[k]=_kernel_time[k]/(count-1);
																		            }
														        
														        /*
															 *         out<<_layer_time<<",";
															 *                 for(unsigned int j=0; j<11; j++)
															 *                         {
															 *                                     out<<_kernel_time[j]<<",";
															 *                                             }
															 *                                                     */
														        
														        /*
															 *         out<<_kernel_time[1]<<","<<_kernel_time[3]<<","<<_kernel_time[9];
															 *                 */
														        out<<"conv_layer"<<","<<_layer_time;
															        
															        out<<std::endl;
																        out.close();
																	    }
									        now++;
										    

										    
}

void NEGEMMConvolutionLayer::prepare()
{
	    if(!_is_prepared)
		        {
				        ARM_COMPUTE_ERROR_ON(!_original_weights->is_used());

					                _weights_reshaped.allocator()->allocate();
							                auto begin1=std::chrono::high_resolution_clock::now();
									        _reshape_weights.run();
										        auto end1=std::chrono::high_resolution_clock::now();
											        _weights_reshape_time=std::chrono::duration_cast<std::chrono::duration<double>>(end1 - begin1).count(); 
												        _time[0]=_weights_reshape_time;
													        _original_weights->mark_as_unused();

														                _is_quantized ? _mm_gemmlowp.prepare() : _mm_gemm.prepare();
																        if(!_weights_reshaped.is_used())
																		        {
																				            _weights_reshaped.allocator()->free();
																					            }

																	        _is_prepared = true;
																		    }
}

double* NEGEMMConvolutionLayer::get_kernels_time()
{
	    return _time;
}

