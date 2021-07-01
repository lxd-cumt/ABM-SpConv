#ifndef __ARM_COMPUTE_NEACTLAYERKERNEL_H__
#define __ARM_COMPUTE_NEACTLAYERKERNEL_H__

#include "arm_compute/core/NEON/INEKernel.h"

#include "arm_compute/core/Size2D.h"

namespace arm_compute
{
	class ITensor;

	class NEActLayerKernel : public INEKernel
	{
		public:
			    const char *name() const override
				        {
						        return "NEActLayerKernel";
							    }
			        /** Default constructor */
			        NEActLayerKernel();
				    /** Prevent instances of this class from being copied (As this class contains pointers) */
				    NEActLayerKernel(const NEActLayerKernel &) = delete;
				        /** Prevent instances of this class from being copied (As this class contains pointers) */
				        NEActLayerKernel &operator=(const NEActLayerKernel &) = delete;
					    /** Allow instances of this class to be moved */
					    NEActLayerKernel(NEActLayerKernel &&) = default;
					        /** Allow instances of this class to be moved */
					        NEActLayerKernel &operator=(NEActLayerKernel &&) = default;
						    /** Default destructor */
						    ~NEActLayerKernel() = default;

						        
						        void configure(const ITensor *input, ITensor *output);
							    
							    static Status validate(const ITensorInfo *input, const ITensorInfo *output);

							        void run(const Window &window, const ThreadInfo &info) override;

		private:
								    const ITensor    *_input;
								        ITensor          *_output;
	};
}
#endif 

