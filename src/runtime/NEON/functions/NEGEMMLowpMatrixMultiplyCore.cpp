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
#include "arm_compute/runtime/NEON/functions/NEGEMMLowpMatrixMultiplyCore.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/NEON/kernels/NEGEMMInterleave4x4Kernel.h"
#include "arm_compute/core/NEON/kernels/NEGEMMLowpMatrixMultiplyKernel.h"
#include "arm_compute/core/NEON/kernels/NEGEMMTranspose1xWKernel.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "support/ToolchainSupport.h"

#include<chrono>

using namespace arm_compute;
using namespace arm_compute::misc::shape_calculator;

NEGEMMLowpMatrixMultiplyCore::NEGEMMLowpMatrixMultiplyCore(std::shared_ptr<IMemoryManager> memory_manager)
	    : _memory_group(memory_manager), _asm_glue(memory_manager), _mm_kernel(nullptr), _mtx_a_reshape_kernel(nullptr), _mtx_b_reshape_kernel(nullptr), _mtx_a_reduction_kernel(), _mtx_b_reduction_kernel(),
	          _offset_contribution_kernel(), _offset_contribution_output_stage_kernel(), _vector_sum_col(), _vector_sum_row(), _tmp_a(), _tmp_b(), _mm_result_s32(), _original_b(nullptr), _a_offset(0), _b_offset(0),
		        _run_vector_matrix_multiplication(false), _assembly_path(false), _fused_assembly_path(false), _reshape_b_only_on_first_run(false), _is_prepared(false), _fuse_output_stage(false),
			      t_matrix_b_reshape_kernel(0.f), t_matrix_b_reduction_kernel(0.f), t_matrix_a_reshape_kernel(0.f), t_matrix_a_reduction_kernel(0.f), t_assembly_dispatch_kernel(0.f),
			            t_matrix_multiply_kernel(0.f), t_offset_contribution_kernel(0.f), t_offset_contribution_output_stage_kernel(0.f)
{
}

void NEGEMMLowpMatrixMultiplyCore::configure(const ITensor *a, const ITensor *b, const ITensor *c, ITensor *output, const GEMMInfo &gemm_info)
{
	    ARM_COMPUTE_ERROR_ON_NULLPTR(a, b, output);
	        ARM_COMPUTE_UNUSED(c);
		    ARM_COMPUTE_ERROR_THROW_ON(NEGEMMLowpMatrixMultiplyCore::validate(a->info(), b->info(), c != nullptr ? c->info() : nullptr, output->info(), gemm_info));

		        const ITensor *matrix_a = a;
			    const ITensor *matrix_b = b;

			            _mtx_a_reshape_kernel = nullptr;
				        _mtx_b_reshape_kernel = nullptr;

					        _a_offset                         = a->info()->quantization_info().uniform().offset;
						    _b_offset                         = b->info()->quantization_info().uniform().offset;
						        _run_vector_matrix_multiplication = a->info()->dimension(1) < 2;
							    _reshape_b_only_on_first_run      = gemm_info.reshape_b_only_on_first_run();
							        _is_prepared                      = false;
								    _fused_assembly_path              = false;
								        _original_b                       = b;

									        if(gemm_info.gemmlowp_output_stage().type != GEMMLowpOutputStageType::NONE)
											    {
												            _fuse_output_stage = true;
													            _memory_group.manage(&_mm_result_s32);
														            TensorInfo info_mm_result_s32(output->info()->tensor_shape(), 1, DataType::S32);
															            _mm_result_s32.allocator()->init(info_mm_result_s32);
																        }

#ifdef __aarch64__
										    switch(a->info()->data_type())
											        {
													        case DataType::QASYMM8:
															        case DataType::U8:
															        case DataType::S8:
															        {
																	            if(a->info()->data_type() == DataType::QASYMM8 && gemm_info.gemmlowp_output_stage().type == GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT)
																			                {
																						                _asm_glue.configure(a, b, c, output, 1.f, 0.f, gemm_info);
																								                _fused_assembly_path = _asm_glue.is_configured();
																										            }
																		                else
																					            {
																							                    _asm_glue.configure(a, b, nullptr, _fuse_output_stage ? &_mm_result_s32 : output, 1.f, 0.f, gemm_info);
																									                }
																				            _assembly_path = _asm_glue.is_configured();
																					                break;
																							        }
																        default:
																        {
																		            ARM_COMPUTE_ERROR("Datatype not supported");
																			                break;
																					        }
																	    }
#endif /* __aarch64__ */
										        if(!(_assembly_path || _run_vector_matrix_multiplication))
												    {
													            matrix_a = &_tmp_a;
														            matrix_b = &_tmp_b;

															                    TensorInfo a_info(compute_interleaved_shape(*a->info()), 1, a->info()->data_type(), a->info()->quantization_info());
																	                    TensorInfo b_info(compute_transpose1xW_shape(*b->info()), 1, b->info()->data_type(), b->info()->quantization_info());
																			            _tmp_a.allocator()->init(a_info);
																				            _tmp_b.allocator()->init(b_info);
																					            _memory_group.manage(&_tmp_a);
																						            if(!_reshape_b_only_on_first_run)
																								            {
																										                _memory_group.manage(&_tmp_b);
																												        }

																							                    {
																										                auto k = arm_compute::support::cpp14::make_unique<NEGEMMInterleave4x4Kernel>();
																												            k->configure(a, &_tmp_a);
																													                _mtx_a_reshape_kernel = std::move(k);
																															        }

																									                    {
																												                auto k = arm_compute::support::cpp14::make_unique<NEGEMMTranspose1xWKernel>();
																														            k->configure(b, &_tmp_b);
																															                _mtx_b_reshape_kernel = std::move(k);
																																	        }
																											        }

											    if(!_fused_assembly_path)
												        {
														                if(_a_offset != 0)
																	        {
																			            TensorInfo info_vector_sum_col(compute_reductionA_shape(*b->info()), 1, DataType::S32);

																				                _vector_sum_col.allocator()->init(info_vector_sum_col);
																						            if(!_reshape_b_only_on_first_run)
																								                {
																											                _memory_group.manage(&_vector_sum_col);
																													            }

																							                            _mtx_b_reduction_kernel.configure(b, &_vector_sum_col, a->info()->dimension(0), false);
																										            }

																                if(_b_offset != 0)
																			        {
																					            TensorInfo info_vector_sum_row(compute_reductionB_shape(*a->info()), 1, DataType::S32);

																						                _vector_sum_row.allocator()->init(info_vector_sum_row);
																								            _memory_group.manage(&_vector_sum_row);

																									                            _mtx_a_reduction_kernel.configure(a, &_vector_sum_row, a->info()->dimension(0), false);
																												            }

																		        if(_fuse_output_stage)
																				        {
																						                        if(!_assembly_path)
																										            {
																												                    auto k = arm_compute::support::cpp14::make_unique<NEGEMMLowpMatrixMultiplyKernel>();
																														                    k->configure(matrix_a, matrix_b, &_mm_result_s32);
																																                    _mm_kernel = std::move(k);
																																		                }

																									            _offset_contribution_output_stage_kernel.configure(&_mm_result_s32, _a_offset == 0 ? nullptr : &_vector_sum_col, _b_offset == 0 ? nullptr : &_vector_sum_row, c, output, a->info()->dimension(0),
																												                                                                   _a_offset, _b_offset, gemm_info.gemmlowp_output_stage());
																										            }
																			        else
																					        {
																							                        if(!_assembly_path)
																											            {
																													                    auto k = arm_compute::support::cpp14::make_unique<NEGEMMLowpMatrixMultiplyKernel>();
																															                    k->configure(matrix_a, matrix_b, output);
																																	                    _mm_kernel = std::move(k);
																																			                }
																										                        _offset_contribution_kernel.configure(output, _a_offset == 0 ? nullptr : &_vector_sum_col, _b_offset == 0 ? nullptr : &_vector_sum_row, a->info()->dimension(0), _a_offset, _b_offset);
																													        }
																				    }

											            if(!_assembly_path && !_run_vector_matrix_multiplication)
													        {
															        _tmp_a.allocator()->allocate();
																        if(!_reshape_b_only_on_first_run)
																		        {
																				            _tmp_b.allocator()->allocate();
																					            }
																	    }

												        if(!_fused_assembly_path)
														    {
															            if(_a_offset != 0 && !_reshape_b_only_on_first_run)
																	            {
																			                _vector_sum_col.allocator()->allocate();
																					        }

																            if(_b_offset != 0)
																		            {
																				                _vector_sum_row.allocator()->allocate();
																						        }
																	        }

													    if(_fuse_output_stage)
														        {
																        _mm_result_s32.allocator()->allocate();
																	    }
}

Status NEGEMMLowpMatrixMultiplyCore::validate(const ITensorInfo *a, const ITensorInfo *b, const ITensorInfo *c, const ITensorInfo *output, const GEMMInfo &gemm_info)
{
	    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(a, 1, DataType::QASYMM8);
	        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::S32, DataType::QASYMM8);
		    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(a, b);
		        ARM_COMPUTE_RETURN_ERROR_ON_MSG(c != nullptr && gemm_info.gemmlowp_output_stage().type == GEMMLowpOutputStageType::NONE, "Bias addition not supported in NEGEMMLowpMatrixMultiplyCore for output S32");
			    ARM_COMPUTE_RETURN_ERROR_ON_MSG((a)->dimension(0) != (b)->dimension(1),
					                                        "The product AB is defined only if the number of columns in A is equal to the number of rows in B");
			        ARM_COMPUTE_RETURN_ERROR_ON_MSG(gemm_info.is_a_reshaped(), "Matrix A already reshaped is not supported");
				    ARM_COMPUTE_RETURN_ERROR_ON_MSG(gemm_info.is_b_reshaped(), "Matrix B already reshaped is not supported");

				        const ITensorInfo *matrix_a_info = a;
					    const ITensorInfo *matrix_b_info = b;

					        TensorInfo tmp_a_info{};
						    TensorInfo tmp_b_info{};
						        TensorInfo mm_result_s32_info{};

							    int32_t a_offset = a->quantization_info().uniform().offset;
							        int32_t b_offset = b->quantization_info().uniform().offset;

								    bool fuse_output_stage = gemm_info.gemmlowp_output_stage().type != GEMMLowpOutputStageType::NONE;
								        if(fuse_output_stage)
										    {
											            auto_init_if_empty(mm_result_s32_info, a->clone()->set_tensor_shape(output->tensor_shape()).set_data_type(DataType::S32));
												        }

									        bool run_optimised             = false;
										    bool run_optimised_requantized = false;
										        if(is_data_type_quantized_asymmetric(a->data_type()))
												    {
													            run_optimised             = bool(NEGEMMAssemblyDispatch::validate(a, b, c, output, 1.f, 0.f, gemm_info));
														            run_optimised_requantized = run_optimised;
															        }
											    else
												        {
														        run_optimised = bool(NEGEMMAssemblyDispatch::validate(a, b, nullptr, fuse_output_stage ? &mm_result_s32_info : output, 1.f, 0.f, gemm_info));
															    }

											        if(run_optimised)
													    {
														            ARM_COMPUTE_RETURN_ERROR_ON(b->dimension(0) != output->dimension(0));
															            if(gemm_info.depth_output_gemm3d() != 0)
																	            {
																			                if(gemm_info.reinterpret_input_as_3d())
																						            {
																								                    ARM_COMPUTE_RETURN_ERROR_ON(a->dimension(1) != output->dimension(1));
																										                    ARM_COMPUTE_RETURN_ERROR_ON(a->dimension(2) != output->dimension(2));
																												                }
																					            else
																							                {
																										                ARM_COMPUTE_RETURN_ERROR_ON(a->dimension(1) != output->dimension(1) * output->dimension(2));
																												            }
																						            }
																            else
																		            {
																				                ARM_COMPUTE_RETURN_ERROR_ON(a->dimension(1) != output->dimension(1));
																						        }
																	        }
												    else
													        {
															        ARM_COMPUTE_RETURN_ERROR_ON_MSG(gemm_info.reinterpret_input_as_3d(), "NEGEMM cannot reinterpret the input tensor as 3D");
																        ARM_COMPUTE_RETURN_ERROR_ON_MSG(gemm_info.depth_output_gemm3d() != 0, "NEGEMM cannot reinterpret the output tensor as 3D");

																	        const bool run_vector_matrix_multiplication = a->dimension(1) < 2;
																		        if(!run_vector_matrix_multiplication)
																				        {
																						            matrix_a_info = &tmp_a_info;
																							                matrix_b_info = &tmp_b_info;

																									                        TensorShape shape_tmp_a = a->tensor_shape();
																												            shape_tmp_a.set(0, a->dimension(0) * 4);
																													                shape_tmp_a.set(1, std::ceil(a->dimension(1) / 4.f));

																															                        TensorShape shape_tmp_b = b->tensor_shape();
																																		            shape_tmp_b.set(0, b->dimension(1) * 16);
																																			                shape_tmp_b.set(1, std::ceil(b->dimension(0) / 16.f));

																																					                        auto_init_if_empty(tmp_a_info, a->clone()->set_tensor_shape(shape_tmp_a));
																																								            auto_init_if_empty(tmp_b_info, b->clone()->set_tensor_shape(shape_tmp_b));

																																									                ARM_COMPUTE_RETURN_ON_ERROR(NEGEMMInterleave4x4Kernel::validate(a, &tmp_a_info));
																																											            ARM_COMPUTE_RETURN_ON_ERROR(NEGEMMTranspose1xWKernel::validate(b, &tmp_b_info));
																																												            }
																			    }

												        if(!run_optimised_requantized)
														    {
															            TensorInfo info_vector_sum_col{};
																            TensorInfo info_vector_sum_row{};

																	                    if(a_offset != 0)
																				            {
																						                info_vector_sum_col = TensorInfo(compute_reductionA_shape(*b), 1, DataType::S32);

																								                        ARM_COMPUTE_RETURN_ON_ERROR(NEGEMMLowpMatrixBReductionKernel::validate(b, &info_vector_sum_col, a->dimension(0), false));
																											        }

																			                    if(b_offset != 0)
																						            {
																								                info_vector_sum_row = TensorInfo(compute_reductionB_shape(*a), 1, DataType::S32);

																										                        ARM_COMPUTE_RETURN_ON_ERROR(NEGEMMLowpMatrixAReductionKernel::validate(a, &info_vector_sum_row, a->dimension(0), false));
																													        }

																					            if(fuse_output_stage)
																							            {
																									                if(!run_optimised)
																												            {
																														                    ARM_COMPUTE_RETURN_ON_ERROR(NEGEMMLowpMatrixMultiplyKernel::validate(matrix_a_info, matrix_b_info, &mm_result_s32_info));
																																                }

																											                        ARM_COMPUTE_RETURN_ON_ERROR(NEGEMMLowpOffsetContributionOutputStageKernel::validate(&mm_result_s32_info,
																																	                                                                                                a_offset == 0 ? nullptr : &info_vector_sum_col,
																																													                                                                                                b_offset == 0 ? nullptr : &info_vector_sum_row,
																																																									                                                                                                c, output, a_offset, b_offset,
																																																																					                                                                                                gemm_info.gemmlowp_output_stage()));
																														        }
																						            else
																								            {
																										                if(!run_optimised)
																													            {
																															                    ARM_COMPUTE_RETURN_ON_ERROR(NEGEMMLowpMatrixMultiplyKernel::validate(matrix_a_info, matrix_b_info, output));
																																	                }
																												                        ARM_COMPUTE_RETURN_ON_ERROR(NEGEMMLowpOffsetContributionKernel::validate(output,
																																		                                                                                     a_offset == 0 ? nullptr : &info_vector_sum_col,
																																												                                                                                          b_offset == 0 ? nullptr : &info_vector_sum_row,
																																																							                                                                                       a_offset, b_offset));
																															        }
																							        }
													    return Status{};
}

void NEGEMMLowpMatrixMultiplyCore::run()
{
	    t_matrix_b_reshape_kernel=0;
	        t_matrix_b_reduction_kernel=0;
		    t_matrix_a_reshape_kernel=0;
		        t_matrix_a_reduction_kernel=0;
			    t_assembly_dispatch_kernel=0;
			        t_matrix_multiply_kernel=0;
				    t_offset_contribution_kernel=0;
				        t_offset_contribution_output_stage_kernel=0;



					    prepare();

					        MemoryGroupResourceScope scope_mg(_memory_group);

						        if(_mtx_a_reshape_kernel)
								    {
									            auto begin15=std::chrono::high_resolution_clock::now();
										            NEScheduler::get().schedule(_mtx_a_reshape_kernel.get(), Window::DimY);
											            auto end15=std::chrono::high_resolution_clock::now();
												            t_matrix_a_reshape_kernel=std::chrono::duration_cast<std::chrono::duration<double>>(end15 - begin15).count(); 
													        }
							    if(_mtx_b_reshape_kernel && !_reshape_b_only_on_first_run)
								        {
										        auto begin16=std::chrono::high_resolution_clock::now();
											        NEScheduler::get().schedule(_mtx_b_reshape_kernel.get(), Window::DimY);
												        auto end16=std::chrono::high_resolution_clock::now();
													        t_matrix_b_reshape_kernel=std::chrono::duration_cast<std::chrono::duration<double>>(end16 - begin16).count(); 
														    }

							            if(_asm_glue.is_configured())
									        {
											        auto begin17 = std::chrono::high_resolution_clock::now();
												        _asm_glue.run();
													        auto end17=std::chrono::high_resolution_clock::now();
														        t_assembly_dispatch_kernel = std::chrono::duration_cast<std::chrono::duration<double>>(end17 - begin17).count();
															    }
								        else
										    {
											            auto begin18 = std::chrono::high_resolution_clock::now();
												            NEScheduler::get().schedule(_mm_kernel.get(), Window::DimY);
													            auto end18=std::chrono::high_resolution_clock::now();
														            t_matrix_multiply_kernel = std::chrono::duration_cast<std::chrono::duration<double>>(end18 - begin18).count();
															        }

									    if(!_fused_assembly_path)
										        {
												                if(_b_offset != 0)
															        {
																	            auto begin19=std::chrono::high_resolution_clock::now();
																		                NEScheduler::get().schedule(&_mtx_a_reduction_kernel, Window::DimX);
																				            auto end19=std::chrono::high_resolution_clock::now();
																					                t_matrix_a_reduction_kernel=std::chrono::duration_cast<std::chrono::duration<double>>(end19 - begin19).count(); 
																							        }

														                if(_a_offset != 0 && !_reshape_b_only_on_first_run)
																	        {
																			            auto begin20=std::chrono::high_resolution_clock::now();
																				                NEScheduler::get().schedule(&_mtx_b_reduction_kernel, Window::DimX);
																						            auto end20=std::chrono::high_resolution_clock::now();
																							                t_matrix_b_reduction_kernel=std::chrono::duration_cast<std::chrono::duration<double>>(end20 - begin20).count(); 
																									        }

																        if(_fuse_output_stage)
																		        {
																				            auto begin21=std::chrono::high_resolution_clock::now();
																					                            NEScheduler::get().schedule(&_offset_contribution_output_stage_kernel, Window::DimY);
																								                auto end21=std::chrono::high_resolution_clock::now();
																										            t_offset_contribution_kernel=std::chrono::duration_cast<std::chrono::duration<double>>(end21 - begin21).count(); 
																											            }
																	        else
																			        {
																					            auto begin22=std::chrono::high_resolution_clock::now();
																						                            NEScheduler::get().schedule(&_offset_contribution_kernel, Window::DimY);
																									                auto end22=std::chrono::high_resolution_clock::now();
																											            t_offset_contribution_output_stage_kernel=std::chrono::duration_cast<std::chrono::duration<double>>(end22 - begin22).count(); 
																												            }
																		    }
}

void NEGEMMLowpMatrixMultiplyCore::prepare()
{
	    t_matrix_b_reshape_kernel=0;
	        t_matrix_b_reduction_kernel=0;
		    if(!_is_prepared)
			        {
					                if(_asm_glue.is_configured() && _reshape_b_only_on_first_run)
								        {
										            ARM_COMPUTE_ERROR_ON(!_original_b->is_used());

											                _asm_glue.prepare();
													            _original_b->mark_as_unused();
														            }
							                else if(_mtx_b_reshape_kernel && _reshape_b_only_on_first_run)
										        {
												            ARM_COMPUTE_ERROR_ON(!_original_b->is_used());

													                            _tmp_b.allocator()->allocate();
																                auto begin13=std::chrono::high_resolution_clock::now();
																		            NEScheduler::get().schedule(_mtx_b_reshape_kernel.get(), Window::DimY);
																			                auto end13=std::chrono::high_resolution_clock::now();
																					            t_matrix_b_reshape_kernel=std::chrono::duration_cast<std::chrono::duration<double>>(end13 - begin13).count(); 
																						                _original_b->mark_as_unused();
																								        }

									                if(_a_offset != 0 && _reshape_b_only_on_first_run)
												        {
														            _vector_sum_col.allocator()->allocate();
															                auto begin14=std::chrono::high_resolution_clock::now();
																	            NEScheduler::get().schedule(&_mtx_b_reduction_kernel, Window::DimX);
																		                auto end14=std::chrono::high_resolution_clock::now();
																				            t_matrix_b_reduction_kernel=std::chrono::duration_cast<std::chrono::duration<double>>(end14 - begin14).count(); 
																					            }

											        _is_prepared = true;
												    }
}

double NEGEMMLowpMatrixMultiplyCore::get_matrix_a_reshape_kernel_time()
{
	    return t_matrix_a_reshape_kernel;
}
double NEGEMMLowpMatrixMultiplyCore::get_matrix_a_reduction_kernel_time()
{
	    return t_matrix_a_reduction_kernel;
}
double NEGEMMLowpMatrixMultiplyCore::get_matrix_b_reshape_kernel_time()
{
	    return t_matrix_b_reshape_kernel;
}
double NEGEMMLowpMatrixMultiplyCore::get_matrix_b_reduction_kernel_time()
{
	    return t_matrix_b_reduction_kernel;
}
double NEGEMMLowpMatrixMultiplyCore::get_assembly_dispatch_kernel_time()
{
	    return t_assembly_dispatch_kernel;
}
double NEGEMMLowpMatrixMultiplyCore::get_matrix_multiply_kernel_time()
{
	    return t_matrix_multiply_kernel;
}
double NEGEMMLowpMatrixMultiplyCore::get_offset_contribution_kernel_time()
{
	    return t_offset_contribution_kernel;
}
double NEGEMMLowpMatrixMultiplyCore::get_offset_contribution_output_stage_kernel_time()
{
	    return t_offset_contribution_output_stage_kernel;
}
