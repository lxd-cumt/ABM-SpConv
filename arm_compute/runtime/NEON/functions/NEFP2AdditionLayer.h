#ifndef _ARM_COMPUTE_NEFP2ADDITIONLAYER_H_

#define _ARM_COMPUTE_NEFP2ADDITIONLAYER_H_



#include "arm_compute/runtime/IFunction.h"

#include "arm_compute/core/NEON/kernels/NEFP2AdditionKernel.h"

#include "arm_compute/core/Types.h"

#include "arm_compute/runtime/MemoryGroup.h"

#include "arm_compute/runtime/Tensor.h"

#include <memory>

using namespace std;

namespace arm_compute
{
	class ITensor;

	class NEFP2AdditionLayer : public IFunction

	{

		public:

			    /** Constructor */

			    NEFP2AdditionLayer();

			        /** Prevent instances of this class from being copied (As this class contains pointers) */

			        NEFP2AdditionLayer(const NEFP2AdditionLayer &) = delete;

				    /** Default move constructor */

				    NEFP2AdditionLayer(NEFP2AdditionLayer &&) = default;

				        /** Prevent instances of this class from being copied (As this class contains pointers) */

				        NEFP2AdditionLayer &operator=(const NEFP2AdditionLayer &) = delete;

					    /** Default move assignment operator */

					    NEFP2AdditionLayer &operator=(NEFP2AdditionLayer &&) = default;



					        void configure(const ITensor *input_a, const ITensor *input_b, ITensor *output,  int *fp);



						    static Status validate(const ITensorInfo *input_a, const ITensorInfo *input_b, const ITensorInfo *output);



						        void run() override;



		private:

							    NEFP2AdditionKernel _add_kernel;

	};

}
#endif
