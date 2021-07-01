#ifndef _ARM_COMPUTE_NEACTLAYER_H_

#define _ARM_COMPUTE_NEACTLAYER_H_



#include "arm_compute/runtime/IFunction.h"

#include "arm_compute/core/NEON/kernels/NEActLayerKernel.h"

#include "arm_compute/core/Types.h"

#include "arm_compute/runtime/MemoryGroup.h"

#include "arm_compute/runtime/Tensor.h"

#include <memory>

using namespace std;

namespace arm_compute
{
	class ITensor;

	class NEActLayer : public IFunction

	{

		public:


			    NEActLayer();


			        NEActLayer(const NEActLayer &) = delete;


				    NEActLayer(NEActLayer &&) = default;


				        NEActLayer &operator=(const NEActLayer &) = delete;


					    NEActLayer &operator=(NEActLayer &&) = default;



					        void configure(const ITensor *input, ITensor *output);



						    static Status validate(const ITensorInfo *input, const ITensorInfo *output);




						        void run() override;



		private:

							    NEActLayerKernel _act_kernel;

	};

}
#endif
