/*
 *  * Copyright (c) 2018 ARM Limited.
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
#include "arm_compute/runtime/NEON/INESimpleFunctionNoBorder.h"

#include "arm_compute/runtime/NEON/NEScheduler.h"

#include <string>
#include <fstream>
#include <chrono>
using namespace std;

namespace arm_compute
{
	INESimpleFunctionNoBorder::INESimpleFunctionNoBorder() 
		    : _kernel(),
		        _layer_time(0.f), count(100), now(0)
	{
	}

	void INESimpleFunctionNoBorder::run()
	{
		    ofstream out("./lab2/alexnet/alexnet_avg_time.csv", ios::out | ios::app);
		        auto b=std::chrono::high_resolution_clock::now();

			    NEScheduler::get().schedule(_kernel.get(), Window::DimY);

			        auto e=std::chrono::high_resolution_clock::now();
				    double ttime=std::chrono::duration_cast<std::chrono::duration<double>>(e - b).count(); 

				        const char* name=_kernel.get()->name();
					    std::string kernel_name=std::string(name);
					        if(kernel_name=="NEActivationLayerKernel")
							    {
								            if(now>0)
										            {
												                _layer_time+=(ttime*1000);
														        }
									            if(now==(count-1))
											            {
													                _layer_time=_layer_time/(count-1);
															            out<<"act_layer"<<","<<_layer_time;
																                out<<std::endl;
																		            out.close();
																			            }
										            now++;
											        }
	}
} 
