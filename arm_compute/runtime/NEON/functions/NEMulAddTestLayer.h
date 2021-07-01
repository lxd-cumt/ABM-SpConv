#ifndef _ARM_COMPUTE_NEMULADDTESTLAYER_H_

#define _ARM_COMPUTE_NEMULADDTESTLAYER_H_



#include "arm_compute/runtime/IFunction.h"

#include "arm_compute/core/NEON/kernels/NEMulAddTestKernel.h"

#include "arm_compute/core/Types.h"

#include "arm_compute/runtime/MemoryGroup.h"

#include "arm_compute/runtime/Tensor.h"

#include <memory>

using namespace std;

namespace arm_compute
{
	class ITensor;

	class NEMulAddTestLayer : public IFunction

	{

		public:

			    /** Constructor */

			    NEMulAddTestLayer();

			        /** Prevent instances of this class from being copied (As this class contains pointers) */

			        NEMulAddTestLayer(const NEMulAddTestLayer &) = delete;

				    /** Default move constructor */

				    NEMulAddTestLayer(NEMulAddTestLayer &&) = default;

				        /** Prevent instances of this class from being copied (As this class contains pointers) */

				        NEMulAddTestLayer &operator=(const NEMulAddTestLayer &) = delete;

					    /** Default move assignment operator */

					    NEMulAddTestLayer &operator=(NEMulAddTestLayer &&) = default;



					        void configure(const ITensor *input_a, const ITensor *input_b,  ITensor *output);



						    static Status validate(const ITensorInfo *input_a,  const ITensorInfo *input_b,  const ITensorInfo *output);



						        void run() override;



		private:

							    NEMulAddTestKernel _test_kernel;

	};

}
#endif
