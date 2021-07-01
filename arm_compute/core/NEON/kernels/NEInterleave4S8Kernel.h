/*
 *  * Copyright (c) 2016-2018 ARM Limited.
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
#ifndef __ARM_COMPUTE_NEINTERLEAVE4S8KERNEL_H__
#define __ARM_COMPUTE_NEINTERLEAVE4S8KERNEL_H__

#include "arm_compute/core/NEON/INESimpleKernel.h"

namespace arm_compute
{
	class ITensor;


	class NEInterleave4S8Kernel : public INESimpleKernel
	{
		public:
			    const char *name() const override
				        {
						        return "NEInterleave4S8Kernel";
							    }
			        /* Constructor */
			        NEInterleave4S8Kernel();
				    /** Initialise the kernel's input and output.
				     *      *
				     *           * @param[in]  input  Input tensor. Data types supported: U8/S8/QASYMM8/U16/S16/F16/U32/S32/F32
				     *                * @param[out] output Output tensor which stores the interleaved matrix. Data type supported: same as @p input.
				     *                     */
				    void configure(const ITensor *input, ITensor *output);
				        /** Static function to check if given info will lead to a valid configuration of @ref NEGEMMInterleave4x4Kernel
					 *      *
					 *           * @param[in] input  Input tensor info. Data types supported: U8/S8/QASYMM8/U16/S16/F16/U32/S32/F32
					 *                * @param[in] output Output tensor info which stores the interleaved matrix. Data type supported: same as @p input.
					 *                     *
					 *                          * @return a status
					 *                               */
				        static Status validate(const ITensorInfo *input, const ITensorInfo *output);

					    void run(const Window &window, const ThreadInfo &info) override;

		private:
					        unsigned int _x_last;
						    unsigned int _y_last;
	};
} 
#endif /*__ARM_COMPUTE_NEGEMMINTERLEAVE4x4KERNEL_H__*/
