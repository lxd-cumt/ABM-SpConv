#include "utils/GraphUtils.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/graph/Logger.h"
#include "arm_compute/runtime/SubTensor.h"
#include "utils/ImageLoader.h"
#include "utils/Utils.h"

#include <iomanip>
#include <limits>

using namespace arm_compute::graph_utils;

namespace
{
std::pair<arm_compute::TensorShape, arm_compute::PermutationVector> compute_permutation_parameters(const arm_compute::TensorShape &shape,
                                                                                                   arm_compute::DataLayout data_layout)
{
    arm_compute::TensorShape       permuted_shape = shape;
    arm_compute::PermutationVector perm;
    if(shape.num_dimensions() > 2)
    {
        perm = (data_layout == arm_compute::DataLayout::NHWC) ? arm_compute::PermutationVector(2U, 0U, 1U) : arm_compute::PermutationVector(1U, 2U, 0U);

        arm_compute::PermutationVector perm_shape = (data_layout == arm_compute::DataLayout::NCHW) ? arm_compute::PermutationVector(2U, 0U, 1U) : arm_compute::PermutationVector(1U, 2U, 0U);
        arm_compute::permute(permuted_shape, perm_shape);
    }

    return std::make_pair(permuted_shape, perm);
}
} 

TFPreproccessor::TFPreproccessor(float min_range, float max_range)
    : _min_range(min_range), _max_range(max_range)
{
}
void TFPreproccessor::preprocess(ITensor &tensor)
{
    Window window;
    window.use_tensor_dimensions(tensor.info()->tensor_shape());

    const float range = _max_range - _min_range;

    execute_window_loop(window, [&](const Coordinates & id)
    {
        const float value                                     = *reinterpret_cast<float *>(tensor.ptr_to_element(id));
        float       res                                       = value / 255.f;            
        res                                                   = res * range + _min_range; 
        *reinterpret_cast<float *>(tensor.ptr_to_element(id)) = res;
    });
}

CaffePreproccessor::CaffePreproccessor(std::array<float, 3> mean, bool bgr, float scale)
    : _mean(mean), _bgr(bgr), _scale(scale)
{
    if(_bgr)
    {
        std::swap(_mean[0], _mean[2]);
    }
}

void CaffePreproccessor::preprocess(ITensor &tensor)
{
    Window window;
    window.use_tensor_dimensions(tensor.info()->tensor_shape());

    const int channel_idx = get_data_layout_dimension_index(tensor.info()->data_layout(), DataLayoutDimension::CHANNEL);

    float a[3]={0.485,0.456,0.406};
    float b[3]={0.229,0.224,0.225};
    execute_window_loop(window, [&](const Coordinates & id)
    {
        /*const float value                                     = *reinterpret_cast<float *>(tensor.ptr_to_element(id)) - _mean[id[channel_idx]];*/
        const float value                                     = (((((*reinterpret_cast<float *>(tensor.ptr_to_element(id))) - _mean[id[channel_idx]]+_mean[id[channel_idx]])/255)-a[id[channel_idx]])/b[channel_idx])*32;
        *reinterpret_cast<float *>(tensor.ptr_to_element(id)) = value * _scale;
    });
}

PPMWriter::PPMWriter(std::string name, unsigned int maximum)
    : _name(std::move(name)), _iterator(0), _maximum(maximum)
{
}

bool PPMWriter::access_tensor(ITensor &tensor)
{
    std::stringstream ss;
    ss << _name << _iterator << ".ppm";

    arm_compute::utils::save_to_ppm(tensor, ss.str());

    _iterator++;
    if(_maximum == 0)
    {
        return true;
    }
    return _iterator < _maximum;
}

DummyAccessor::DummyAccessor(unsigned int maximum)
    : _iterator(0), _maximum(maximum)
{
}

bool DummyAccessor::access_tensor(ITensor &tensor)
{
    ARM_COMPUTE_UNUSED(tensor);
    bool ret = _maximum == 0 || _iterator < _maximum;
    if(_iterator == _maximum)
    {
        _iterator = 0;
    }
    else
    {
        _iterator++;
    }
    return ret;
}

NumPyAccessor::NumPyAccessor(std::string npy_path, TensorShape shape, DataType data_type, DataLayout data_layout, std::ostream &output_stream)
    : _npy_tensor(), _filename(std::move(npy_path)), _output_stream(output_stream)
{
    NumPyBinLoader loader(_filename, data_layout);

    TensorInfo info(shape, 1, data_type);
    info.set_data_layout(data_layout);

    _npy_tensor.allocator()->init(info);
    _npy_tensor.allocator()->allocate();

    loader.access_tensor(_npy_tensor);
}

template <typename T>
void NumPyAccessor::access_numpy_tensor(ITensor &tensor, T tolerance)
{
    const int num_elements          = tensor.info()->tensor_shape().total_size();
    int       num_mismatches        = utils::compare_tensor<T>(tensor, _npy_tensor, tolerance);
    float     percentage_mismatches = static_cast<float>(num_mismatches) / num_elements;

    _output_stream << "Results: " << 100.f - (percentage_mismatches * 100) << " % matches with the provided output[" << _filename << "]." << std::endl;
    _output_stream << "         " << num_elements - num_mismatches << " out of " << num_elements << " matches with the provided output[" << _filename << "]." << std::endl
                   << std::endl;
}

bool NumPyAccessor::access_tensor(ITensor &tensor)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(&tensor, 1, DataType::F32, DataType::QASYMM8);
    ARM_COMPUTE_ERROR_ON(_npy_tensor.info()->dimension(0) != tensor.info()->dimension(0));

    switch(tensor.info()->data_type())
    {
        case DataType::QASYMM8:
            access_numpy_tensor<qasymm8_t>(tensor, 0);
            break;
        case DataType::F32:
            access_numpy_tensor<float>(tensor, 0.0001f);
            break;
        default:
            ARM_COMPUTE_ERROR("NOT SUPPORTED!");
    }

    return false;
}

SaveNumPyAccessor::SaveNumPyAccessor(std::string npy_name, const bool is_fortran)
    : _npy_name(std::move(npy_name)), _is_fortran(is_fortran)
{
}

bool SaveNumPyAccessor::access_tensor(ITensor &tensor)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(&tensor, 1, DataType::F32);

    utils::save_to_npy(tensor, _npy_name, _is_fortran);

    return false;
}

ImageAccessor::ImageAccessor(std::string filename, bool bgr, std::unique_ptr<IPreprocessor> preprocessor)
    : _already_loaded(false), _filename(std::move(filename)), _bgr(bgr), _preprocessor(std::move(preprocessor))
{
}

bool ImageAccessor::access_tensor(ITensor &tensor)
{
    if(!_already_loaded)
    {
        auto image_loader = utils::ImageLoaderFactory::create(_filename);
        ARM_COMPUTE_EXIT_ON_MSG(image_loader == nullptr, "Unsupported image type");

        image_loader->open(_filename);

        TensorShape                    permuted_shape = tensor.info()->tensor_shape();
        arm_compute::PermutationVector perm;
        if(tensor.info()->data_layout() != DataLayout::NCHW)
        {
            std::tie(permuted_shape, perm) = compute_permutation_parameters(tensor.info()->tensor_shape(), tensor.info()->data_layout());
        }
        ARM_COMPUTE_EXIT_ON_MSG(image_loader->width() != permuted_shape.x() || image_loader->height() != permuted_shape.y(),
                                "Failed to load image file: dimensions [%d,%d] not correct, expected [%d,%d].",
                                image_loader->width(), image_loader->height(), permuted_shape.x(), permuted_shape.y());

        image_loader->fill_planar_tensor(tensor, _bgr);

        if(_preprocessor)
        {
            _preprocessor->preprocess(tensor);
        }
    }

    _already_loaded = !_already_loaded;
    return _already_loaded;
}

ValidationInputAccessor::ValidationInputAccessor(const std::string             &image_list,
                                                 std::string                    images_path,
                                                 std::unique_ptr<IPreprocessor> preprocessor,
                                                 bool                           bgr,
                                                 unsigned int                   start,
                                                 unsigned int                   end,
                                                 std::ostream                  &output_stream)
    : _path(std::move(images_path)), _images(), _preprocessor(std::move(preprocessor)), _bgr(bgr), _offset(0), _output_stream(output_stream)
{
    ARM_COMPUTE_EXIT_ON_MSG(start > end, "Invalid validation range!");

    std::ifstream ifs;
    try
    {
        ifs.exceptions(std::ifstream::badbit);
        ifs.open(image_list, std::ios::in | std::ios::binary);

        unsigned int counter = 0;
        for(std::string line; !std::getline(ifs, line).fail() && counter <= end; ++counter)
        {
            if(counter >= start)
            {
                std::stringstream linestream(line);
                std::string       image_name;

                linestream >> image_name;
                _images.emplace_back(std::move(image_name));
            }
        }
    }
    catch(const std::ifstream::failure &e)
    {
        ARM_COMPUTE_ERROR("Accessing %s: %s", image_list.c_str(), e.what());
    }
}

bool ValidationInputAccessor::access_tensor(arm_compute::ITensor &tensor)
{
    bool ret = _offset < _images.size();
    if(ret)
    {
        utils::JPEGLoader jpeg;

        std::string image_name = _path + _images[_offset++];
        jpeg.open(image_name);
        _output_stream << "[" << _offset << "/" << _images.size() << "] Validating " << image_name << std::endl;

        TensorShape                    permuted_shape = tensor.info()->tensor_shape();
        arm_compute::PermutationVector perm;
        if(tensor.info()->data_layout() != DataLayout::NCHW)
        {
            std::tie(permuted_shape, perm) = compute_permutation_parameters(tensor.info()->tensor_shape(),
                                                                            tensor.info()->data_layout());
        }
        ARM_COMPUTE_EXIT_ON_MSG(jpeg.width() != permuted_shape.x() || jpeg.height() != permuted_shape.y(),
                                "Failed to load image file: dimensions [%d,%d] not correct, expected [%d,%d].",
                                jpeg.width(), jpeg.height(), permuted_shape.x(), permuted_shape.y());

        jpeg.fill_planar_tensor(tensor, _bgr);


        if(_preprocessor)
        {
            _preprocessor->preprocess(tensor);
        }
    }

    return ret;
}

ValidationOutputAccessor::ValidationOutputAccessor(const std::string &image_list,
                                                   std::ostream      &output_stream,
                                                   unsigned int       start,
                                                   unsigned int       end)
    : _results(), _output_stream(output_stream), _offset(0), _positive_samples_top1(0), _positive_samples_top5(0)
{
    ARM_COMPUTE_EXIT_ON_MSG(start > end, "Invalid validation range!");

    std::ifstream ifs;
    try
    {
        ifs.exceptions(std::ifstream::badbit);
        ifs.open(image_list, std::ios::in | std::ios::binary);

        unsigned int counter = 0;
        for(std::string line; !std::getline(ifs, line).fail() && counter <= end; ++counter)
        {
            if(counter >= start)
            {
                std::stringstream linestream(line);
                std::string       image_name;
                int               result;

                linestream >> image_name >> result;
                _results.emplace_back(result);
            }
        }
    }
    catch(const std::ifstream::failure &e)
    {
        ARM_COMPUTE_ERROR("Accessing %s: %s", image_list.c_str(), e.what());
    }
}

void ValidationOutputAccessor::reset()
{
    _offset                = 0;
    _positive_samples_top1 = 0;
    _positive_samples_top5 = 0;
}

bool ValidationOutputAccessor::access_tensor(arm_compute::ITensor &tensor)
{
    bool ret = _offset < _results.size();
    if(ret)
    {
        std::vector<size_t> tensor_results;
        switch(tensor.info()->data_type())
        {
            case DataType::QASYMM8:
                tensor_results = access_predictions_tensor<uint8_t>(tensor);
                break;
            case DataType::F32:
                tensor_results = access_predictions_tensor<float>(tensor);
                break;
            default:
                ARM_COMPUTE_ERROR("NOT SUPPORTED!");
        }

        size_t correct_label = _results[_offset++];

        aggregate_sample(tensor_results, _positive_samples_top1, 1, correct_label);
        aggregate_sample(tensor_results, _positive_samples_top5, 5, correct_label);
    }

    if(_offset >= _results.size())
    {
        report_top_n(1, _results.size(), _positive_samples_top1);
        report_top_n(5, _results.size(), _positive_samples_top5);
    }

    return ret;
}

template <typename T>
std::vector<size_t> ValidationOutputAccessor::access_predictions_tensor(arm_compute::ITensor &tensor)
{
    std::vector<size_t> index;

    const auto   output_net  = reinterpret_cast<T *>(tensor.buffer() + tensor.info()->offset_first_element_in_bytes());
    const size_t num_classes = 1000;

    index.resize(num_classes);

    std::iota(std::begin(index), std::end(index), static_cast<size_t>(0));
    std::sort(std::begin(index), std::end(index),
              [&](size_t a, size_t b)
    {
        return output_net[a] > output_net[b];
    });

    return index;
}

void ValidationOutputAccessor::aggregate_sample(const std::vector<size_t> &res, size_t &positive_samples, size_t top_n, size_t correct_label)
{
    auto is_valid_label = [correct_label](size_t label)
    {
        return label == correct_label;
    };

    if(std::any_of(std::begin(res), std::begin(res) + top_n, is_valid_label))
    {
        ++positive_samples;
    }
}

void ValidationOutputAccessor::report_top_n(size_t top_n, size_t total_samples, size_t positive_samples)
{
    size_t negative_samples = total_samples - positive_samples;
    float  accuracy         = positive_samples / static_cast<float>(total_samples);

    _output_stream << "----------Top " << top_n << " accuracy ----------" << std::endl
                   << std::endl;
    _output_stream << "Positive samples : " << positive_samples << std::endl;
    _output_stream << "Negative samples : " << negative_samples << std::endl;
    _output_stream << "Accuracy : " << accuracy << std::endl;
}

DetectionOutputAccessor::DetectionOutputAccessor(const std::string &labels_path, std::vector<TensorShape> &imgs_tensor_shapes, std::ostream &output_stream)
    : _labels(), _tensor_shapes(std::move(imgs_tensor_shapes)), _output_stream(output_stream)
{
    _labels.clear();

    std::ifstream ifs;

    try
    {
        ifs.exceptions(std::ifstream::badbit);
        ifs.open(labels_path, std::ios::in | std::ios::binary);

        for(std::string line; !std::getline(ifs, line).fail();)
        {
            _labels.emplace_back(line);
        }
    }
    catch(const std::ifstream::failure &e)
    {
        ARM_COMPUTE_ERROR("Accessing %s: %s", labels_path.c_str(), e.what());
    }
}

template <typename T>
void DetectionOutputAccessor::access_predictions_tensor(ITensor &tensor)
{
    const size_t num_detection = tensor.info()->valid_region().shape.y();
    const auto   output_prt    = reinterpret_cast<T *>(tensor.buffer() + tensor.info()->offset_first_element_in_bytes());

    if(num_detection > 0)
    {
        _output_stream << "---------------------- Detections ----------------------" << std::endl
                       << std::endl;

        _output_stream << std::left << std::setprecision(4) << std::setw(8) << "Image | " << std::setw(8) << "Label | " << std::setw(12) << "Confidence | "
                       << "[ xmin, ymin, xmax, ymax ]" << std::endl;

        for(size_t i = 0; i < num_detection; ++i)
        {
            auto im = static_cast<const int>(output_prt[i * 7]);
            _output_stream << std::setw(8) << im << std::setw(8)
                           << _labels[output_prt[i * 7 + 1]] << std::setw(12) << output_prt[i * 7 + 2]
                           << " [" << (output_prt[i * 7 + 3] * _tensor_shapes[im].x())
                           << ", " << (output_prt[i * 7 + 4] * _tensor_shapes[im].y())
                           << ", " << (output_prt[i * 7 + 5] * _tensor_shapes[im].x())
                           << ", " << (output_prt[i * 7 + 6] * _tensor_shapes[im].y())
                           << "]" << std::endl;
        }
    }
    else
    {
        _output_stream << "No detection found." << std::endl;
    }
}

bool DetectionOutputAccessor::access_tensor(ITensor &tensor)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(&tensor, 1, DataType::F32);

    switch(tensor.info()->data_type())
    {
        case DataType::F32:
            access_predictions_tensor<float>(tensor);
            break;
        default:
            ARM_COMPUTE_ERROR("NOT SUPPORTED!");
    }

    return false;
}

TopNPredictionsAccessor::TopNPredictionsAccessor(const std::string &labels_path, size_t top_n, std::ostream &output_stream)
    : _labels(), _output_stream(output_stream), _top_n(top_n)
{
    _labels.clear();

    std::ifstream ifs;

    try
    {
        ifs.exceptions(std::ifstream::badbit);
        ifs.open(labels_path, std::ios::in | std::ios::binary);

        for(std::string line; !std::getline(ifs, line).fail();)
        {
            _labels.emplace_back(line);
        }
    }
    catch(const std::ifstream::failure &e)
    {
        ARM_COMPUTE_ERROR("Accessing %s: %s", labels_path.c_str(), e.what());
    }
}

template <typename T>
void TopNPredictionsAccessor::access_predictions_tensor(ITensor &tensor)
{
    std::vector<T>      classes_prob;
    std::vector<size_t> index;

    const auto   output_net  = reinterpret_cast<T *>(tensor.buffer() + tensor.info()->offset_first_element_in_bytes());
    const size_t num_classes = tensor.info()->dimension(0);

    classes_prob.resize(num_classes);
    index.resize(num_classes);

    std::copy(output_net, output_net + num_classes, classes_prob.begin());

    std::iota(std::begin(index), std::end(index), static_cast<size_t>(0));
    std::sort(std::begin(index), std::end(index),
              [&](size_t a, size_t b)
    {
        return classes_prob[a] > classes_prob[b];
    });

    _output_stream << "---------- Top " << _top_n << " predictions ----------" << std::endl
                   << std::endl;
    for(size_t i = 0; i < _top_n; ++i)
    {
        _output_stream << std::fixed << std::setprecision(4)
                       << +classes_prob[index.at(i)]
                       << " - [id = " << index.at(i) << "]"
                       << ", " << _labels[index.at(i)] << std::endl;
    }
}

bool TopNPredictionsAccessor::access_tensor(ITensor &tensor)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(&tensor, 1, DataType::F32, DataType::QASYMM8);
    ARM_COMPUTE_ERROR_ON(_labels.size() != tensor.info()->dimension(0));

    switch(tensor.info()->data_type())
    {
        case DataType::QASYMM8:
            access_predictions_tensor<uint8_t>(tensor);
            break;
        case DataType::F32:
            access_predictions_tensor<float>(tensor);
            break;
        default:
            ARM_COMPUTE_ERROR("NOT SUPPORTED!");
    }

    return false;
}

RandomAccessor::RandomAccessor(PixelValue lower, PixelValue upper, std::random_device::result_type seed)
    : _lower(lower), _upper(upper), _seed(seed)
{
}

template <typename T, typename D>
void RandomAccessor::fill(ITensor &tensor, D &&distribution)
{
    std::mt19937 gen(_seed);

    if(tensor.info()->padding().empty() && (dynamic_cast<SubTensor *>(&tensor) == nullptr))
    {
        for(size_t offset = 0; offset < tensor.info()->total_size(); offset += tensor.info()->element_size())
        {
            const auto value                                 = static_cast<T>(distribution(gen));
            *reinterpret_cast<T *>(tensor.buffer() + offset) = value;
        }
    }
    else
    {
        Window window;
        window.use_tensor_dimensions(tensor.info()->tensor_shape());

        execute_window_loop(window, [&](const Coordinates & id)
        {
            const auto value                                  = static_cast<T>(distribution(gen));
            *reinterpret_cast<T *>(tensor.ptr_to_element(id)) = value;
        });
    }
}

bool RandomAccessor::access_tensor(ITensor &tensor)
{
    switch(tensor.info()->data_type())
    {
        case DataType::QASYMM8:
        case DataType::U8:
        {
            std::uniform_int_distribution<uint8_t> distribution_u8(_lower.get<uint8_t>(), _upper.get<uint8_t>());
            fill<uint8_t>(tensor, distribution_u8);
            break;
        }
        case DataType::S8:
        {
            std::uniform_int_distribution<int8_t> distribution_s8(_lower.get<int8_t>(), _upper.get<int8_t>());
            fill<int8_t>(tensor, distribution_s8);
            break;
        }
        case DataType::U16:
        {
            std::uniform_int_distribution<uint16_t> distribution_u16(_lower.get<uint16_t>(), _upper.get<uint16_t>());
            fill<uint16_t>(tensor, distribution_u16);
            break;
        }
        case DataType::S16:
        {
            std::uniform_int_distribution<int16_t> distribution_s16(_lower.get<int16_t>(), _upper.get<int16_t>());
            fill<int16_t>(tensor, distribution_s16);
            break;
        }
        case DataType::U32:
        {
            std::uniform_int_distribution<uint32_t> distribution_u32(_lower.get<uint32_t>(), _upper.get<uint32_t>());
            fill<uint32_t>(tensor, distribution_u32);
            break;
        }
        case DataType::S32:
        {
            std::uniform_int_distribution<int32_t> distribution_s32(_lower.get<int32_t>(), _upper.get<int32_t>());
            fill<int32_t>(tensor, distribution_s32);
            break;
        }
        case DataType::U64:
        {
            std::uniform_int_distribution<uint64_t> distribution_u64(_lower.get<uint64_t>(), _upper.get<uint64_t>());
            fill<uint64_t>(tensor, distribution_u64);
            break;
        }
        case DataType::S64:
        {
            std::uniform_int_distribution<int64_t> distribution_s64(_lower.get<int64_t>(), _upper.get<int64_t>());
            fill<int64_t>(tensor, distribution_s64);
            break;
        }
        case DataType::F16:
        {
            std::uniform_real_distribution<float> distribution_f16(_lower.get<half>(), _upper.get<half>());
            fill<half>(tensor, distribution_f16);
            break;
        }
        case DataType::F32:
        {
            std::uniform_real_distribution<float> distribution_f32(_lower.get<float>(), _upper.get<float>());
            fill<float>(tensor, distribution_f32);
            break;
        }
        case DataType::F64:
        {
            std::uniform_real_distribution<double> distribution_f64(_lower.get<double>(), _upper.get<double>());
            fill<double>(tensor, distribution_f64);
            break;
        }
        default:
            ARM_COMPUTE_ERROR("NOT SUPPORTED!");
    }
    return true;
}

NumPyBinLoader::NumPyBinLoader(std::string filename, DataLayout file_layout)
    : _already_loaded(false), _filename(std::move(filename)), _file_layout(file_layout)
{
}

bool NumPyBinLoader::access_tensor(ITensor &tensor)
{
    if(!_already_loaded)
    {
        utils::NPYLoader loader;
        loader.open(_filename, _file_layout);
        loader.fill_tensor(tensor);
    }

    _already_loaded = !_already_loaded;
    return _already_loaded;
}

