#ifndef __ARM_COMPUTE_INESIMPLEFUNCTIONNOBORDER_H__
#define __ARM_COMPUTE_INESIMPLEFUNCTIONNOBORDER_H__

#include "arm_compute/core/NEON/INEKernel.h"
#include "arm_compute/runtime/IFunction.h"

#include <memory>

namespace arm_compute
{
	/** Basic interface for functions which have a single NEON kernel and no border */
	class INESimpleFunctionNoBorder : public IFunction
	{
		public:
			    /** Constructor */
			    INESimpleFunctionNoBorder();

			        void run() override final;

		protected:
				    std::unique_ptr<INEKernel> _kernel; /**< Kernel to run */

				        double _layer_time=0.f;
					    unsigned int count=100;
					        unsigned int now=0;
	};
} 
#endif /*__ARM_COMPUTE_INESIMPLEFUNCTIONNOBORDER_H__ */
