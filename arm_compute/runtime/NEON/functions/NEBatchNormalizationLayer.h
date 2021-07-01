#ifndef __ARM_COMPUTE_NEBATCHNORMALIZATIONLAYER_H__
#define __ARM_COMPUTE_NEBATCHNORMALIZATIONLAYER_H__

#include "arm_compute/core/NEON/kernels/NEBatchNormalizationLayerKernel.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/NEON/functions/NEActivationLayer.h"

namespace arm_compute
{
	class ITensor;


	class NEBatchNormalizationLayer : public IFunction
	{
		public:
			    /** Default constructor */
			    NEBatchNormalizationLayer();
			        /** Set the input and output tensors.
				 *      *
				 *           * @note If the output tensor is a nullptr or is equal to the input, the batch normalization function will be performed in-place
				 *                *
				 *                     * @param[in, out] input    Source tensor. In case of @p output tensor = nullptr, this tensor will store the result.
				 *                          *                          3 lower dimensions represent a single input with dimensions [width, height, FM].
				 *                               *                          The rest are optional and used for representing batches. Data types supported: F16/F32.
				 *                                    * @param[out]     output   Destination tensor. Output will have the same number of dimensions as input. Data type supported: same as @p input
				 *                                         * @param[in]      mean     Mean values tensor. 1 dimension with size equal to the feature maps [FM]. Data types supported: Same as @p input
				 *                                              * @param[in]      var      Variance values tensor. 1 dimension with size equal to the feature maps [FM]. Data types supported: Same as @p input
				 *                                                   * @param[in]      beta     (Optional) Beta values tensor info. 1 dimension with size equal to the feature maps [FM]. If not provided, default value for beta is 0. Data types supported: Same as @p input
				 *                                                        * @param[in]      gamma    (Optional) Gamma values tensor info. 1 dimension with size equal to the feature maps [FM]. If not provided, default value for gamma is 1. Data types supported: Same as @p input
				 *                                                             * @param[in]      epsilon  (Optional) Small value to avoid division with zero. Default value is 0.001f.
				 *                                                                  * @param[in]      act_info (Optional) Activation layer information in case of a fused activation. Only RELU, BOUNDED_RELU and LU_BOUNDED_RELU supported.
				 *                                                                       */
			        void configure(ITensor *input, ITensor *output, const ITensor *mean, const ITensor *var, const ITensor *beta = nullptr, const ITensor *gamma = nullptr, float epsilon = 0.001f,
						                   ActivationLayerInfo act_info = ActivationLayerInfo());
				    /** Static function to check if given info will lead to a valid configuration of @ref NEBatchNormalizationLayer
				     *      *
				     *           * @param[in] input    Source tensor info. In case of @p output tensor = nullptr, this tensor will store the result.
				     *                *                     3 lower dimensions represent a single input with dimensions [width, height, FM].
				     *                     *                     The rest are optional and used for representing batches. Data types supported: F16/F32.
				     *                          * @param[in] output   Destination tensor info. Output will have the same number of dimensions as input. Data type supported: same as @p input
				     *                               * @param[in] mean     Mean values tensor info. 1 dimension with size equal to the feature maps [FM]. Data types supported: Same as @p input
				     *                                    * @param[in] var      Variance values tensor info. 1 dimension with size equal to the feature maps [FM]. Data types supported: Same as @p input
				     *                                         * @param[in] beta     (Optional) Beta values tensor info. 1 dimension with size equal to the feature maps [FM]. If not provided, default value for beta is 0. Data types supported: Same as @p input
				     *                                              * @param[in] gamma    (Optional) Gamma values tensor info. 1 dimension with size equal to the feature maps [FM]. If not provided, default value for gamma is 1. Data types supported: Same as @p input
				     *                                                   * @param[in] epsilon  (Optional) Small value to avoid division with zero. Default value is 0.001f.
				     *                                                        * @param[in] act_info (Optional) Activation layer information in case of a fused activation. Only RELU, BOUNDED_RELU and LU_BOUNDED_RELU supported.
				     *                                                             *
				     *                                                                  * @return a status
				     *                                                                       */
				    static Status validate(const ITensorInfo *input, const ITensorInfo *output,
						                               const ITensorInfo *mean, const ITensorInfo *var,
									                                  const ITensorInfo *beta = nullptr, const ITensorInfo *gamma = nullptr,
													                             float epsilon = 0.001f, ActivationLayerInfo act_info = ActivationLayerInfo());

				        void run() override;

		private:
					    NEBatchNormalizationLayerKernel _norm_kernel; /**< Batch normalization layer kernel */
					        double _layer_time=0.f;
						    unsigned int count=100;
						        unsigned int now=0;
	};
}
#endif /* __ARM_COMPUTE_NEBATCHNORMALIZATIONLAYER_H__ */
