#ifndef __ARM_COMPUTE_NEPOOLINGLAYER_H__
#define __ARM_COMPUTE_NEPOOLINGLAYER_H__

#include "arm_compute/runtime/IFunction.h"

#include "arm_compute/core/NEON/kernels/NEFillBorderKernel.h"
#include "arm_compute/core/NEON/kernels/NEPoolingLayerKernel.h"
#include "arm_compute/core/Types.h"

namespace arm_compute
{
	class ITensor;

	class NEPoolingLayer : public IFunction
	{
		public:
			    /** Constructor */
			    NEPoolingLayer();
			        /** Set the input and output tensors.
				 *      *
				 *           * @note F16 is supported for pool sizes 2 and 3 only
				 *                *
				 *                     * @param[in, out] input     Source tensor. (Written to only when padding != 0) Data types supported: QASYMM8/F16/F32.
				 *                          * @param[out]     output    Destination tensor. Data types supported: Same as @p input.
				 *                               * @param[in]      pool_info Contains pooling operation information described in @ref PoolingLayerInfo.
				 *                                    */
			        void configure(ITensor *input, ITensor *output, const PoolingLayerInfo &pool_info);
				    /** Static function to check if given info will lead to a valid configuration of @ref NEPoolingLayer
				     *      *
				     *           * @note F16 is supported for pool sizes 2 and 3 only
				     *                *
				     *                     * @param[in] input     Source tensor. (Written to only when padding != 0) Data types supported: QASYMM8/F16/F32.
				     *                          * @param[in] output    Destination tensor. Data types supported: Same as @p input.
				     *                               * @param[in] pool_info Contains pooling operation information described in @ref PoolingLayerInfo.
				     *                                    *
				     *                                         * @return a status
				     *                                              */
				    static Status validate(const ITensorInfo *input, const ITensorInfo *output, const PoolingLayerInfo &pool_info);

				        void run() override;

		private:
					    NEPoolingLayerKernel _pooling_layer_kernel;
					        NEFillBorderKernel   _border_handler;
						    bool                 _is_global_pooling_layer;
						        DataLayout           _data_layout;

							    double _layer_time=0.f;
							        unsigned int count=100;
								    unsigned int now=0;
	};
}
#endif 
