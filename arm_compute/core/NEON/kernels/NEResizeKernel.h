#ifndef __ARM_COMPUTE_NERESIZEKERNEL_H__
#define __ARM_COMPUTE_NERESIZEKERNEL_H__

#include "arm_compute/core/NEON/INEKernel.h"

#include "arm_compute/core/Size2D.h"

namespace arm_compute
{
	class ITensor;


	class NEResizeKernel : public INEKernel
	{
		public:
			    const char *name() const override
				        {
						        return "NEResizeKernel";
							    }
			        /** Default constructor */
			        NEResizeKernel();
				    /** Prevent instances of this class from being copied (As this class contains pointers) */
				    NEResizeKernel(const NEResizeKernel &) = delete;
				        /** Prevent instances of this class from being copied (As this class contains pointers) */
				        NEResizeKernel &operator=(const NEResizeKernel &) = delete;
					    /** Allow instances of this class to be moved */
					    NEResizeKernel(NEResizeKernel &&) = default;
					        /** Allow instances of this class to be moved */
					        NEResizeKernel &operator=(NEResizeKernel &&) = default;
						    /** Default destructor */
						    ~NEResizeKernel() = default;

						        void configure(const ITensor *input, ITensor *output, TensorShape output_shape);


							    static Status validate(const ITensorInfo *input, const ITensorInfo *output);

							        void run(const Window &window, const ThreadInfo &info) override;

		private:
								    const ITensor    *_input;
								        ITensor          *_output;
									    TensorShape _output_shape;
	};
} 
#endif 

