#ifndef __ARM_COMPUTE_NEIM2COLKERNEL4S8_H__
#define __ARM_COMPUTE_NEIM2COLKERNEL4S8_H__

#include "arm_compute/core/NEON/INEKernel.h"

namespace arm_compute
{
class ITensor;
class Size2D;

class NEIm2ColKernel4S8 : public INEKernel
{
public:
    const char *name() const override
    {
        return "NEIm2ColKernel4S8";
    }
    /** Default constructor */
    NEIm2ColKernel4S8();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEIm2ColKernel4S8(const NEIm2ColKernel4S8 &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEIm2ColKernel4S8 &operator=(const NEIm2ColKernel4S8 &) = delete;
    /** Allow instances of this class to be moved */
    NEIm2ColKernel4S8(NEIm2ColKernel4S8 &&) = default;
    /** Allow instances of this class to be moved */
    NEIm2ColKernel4S8 &operator=(NEIm2ColKernel4S8 &&) = default;
    /** Default destructor */
    ~NEIm2ColKernel4S8() = default;

    /** Set the input and output of the kernel.
     *
     * @param[in]  input       The input tensor to convert. 3 lower dimensions represent a single input [width, height, IFM],
     *                         while every optional dimension from 4 and above represent a batch of inputs. Data types supported: QASYMM8/F16/F32
     *                         Note: QASYMM8 works only for has_bias = false
     * @param[out] input_transpose      The output tensor. Data types supported: Same as @p input
     * @param[in]  kernel_dims The kernel dimensions (width and height).
     * @param[in]  conv_info   Contains padding and stride information described in @ref PadStrideInfo.
     * @param[in]  has_bias    In case biases are provided expands the matrix with 1.
     * @param[in]  dilation    (Optional) Dilation, in elements, across x and y. Defaults to (1, 1).
     * @param[in]  num_groups  (Optional) Number of groups when performing a grouped convolution. num_groups != 1 is not supported
     */
    void configure(const ITensor *input, ITensor *output, const Size2D &kernel_dims, const PadStrideInfo &conv_info,
                   bool has_bias, const Size2D &dilation = Size2D(1U, 1U), unsigned int num_groups = 1);
    /** Static function to check if given info will lead to a valid configuration of @ref NEIm2ColKernel
     *
     * @param[in] input       The input tensor to convert. 3 lower dimensions represent a single input [width, height, IFM],
     *                        while every optional dimension from 4 and above represent a batch of inputs. Data types supported: QASYMM8/F16/F32
     *                        Note: QASYMM8 works only for has_bias = false
     * @param[in] output      The output tensor. Data types supported: Same as @p input
     * @param[in] kernel_dims The kernel dimensions (width and height).
     * @param[in] conv_info   Contains padding and stride information described in @ref PadStrideInfo.
     * @param[in] has_bias    In case biases are provided expands the matrix with 1.
     * @param[in] dilation    (Optional) Dilation, in elements, across x and y. Defaults to (1, 1).
     * @param[in] num_groups  (Optional) Number of groups when performing a grouped convolution. num_groups != 1 is not supported
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, const Size2D &kernel_dims, const PadStrideInfo &conv_info,
                           bool has_bias, const Size2D &dilation = Size2D(1U, 1U), unsigned int num_groups = 1);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;

private:
    const ITensor    *_input;
    ITensor          *_output;
    std::pair<unsigned int, unsigned int> _convolved_dims;
    PadStrideInfo _conv_info;
    unsigned int  _kernel_width;
    unsigned int  _kernel_height;
    bool          _has_bias;
    Size2D        _dilation;
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_NEIM2COLKERNEL_H__ */

