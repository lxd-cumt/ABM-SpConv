#ifndef _ARM_COMPUTE_NERESIZELAYER_H_

#define _ARM_COMPUTE_NERESIZELAYER_H_



#include "arm_compute/runtime/IFunction.h"

#include "arm_compute/core/NEON/kernels/NEResizeKernel.h"

#include "arm_compute/core/Types.h"

#include "arm_compute/runtime/MemoryGroup.h"

#include "arm_compute/runtime/Tensor.h"

#include <memory>

using namespace std;

namespace arm_compute
{
	class ITensor;

	class NEResizeLayer : public IFunction

	{

		public:

			    /** Constructor */

			    NEResizeLayer();

			        /** Prevent instances of this class from being copied (As this class contains pointers) */

			        NEResizeLayer(const NEResizeLayer &) = delete;

				    /** Default move constructor */

				    NEResizeLayer(NEResizeLayer &&) = default;

				        /** Prevent instances of this class from being copied (As this class contains pointers) */

				        NEResizeLayer &operator=(const NEResizeLayer &) = delete;

					    /** Default move assignment operator */

					    NEResizeLayer &operator=(NEResizeLayer &&) = default;



					        void configure(const ITensor *input, ITensor *output, TensorShape output_shape);



						    static Status validate(const ITensorInfo *input, const ITensorInfo *output);





						        void run() override;



		private:

							    NEResizeKernel _resize_kernel;

	};

}
#endif
