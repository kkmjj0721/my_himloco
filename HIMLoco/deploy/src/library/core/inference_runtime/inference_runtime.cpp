#include "inference_runtime.hpp"

#include <cctype>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>

#ifdef USE_TORCH
#include <ATen/Parallel.h>
#endif

namespace InferenceRuntime
{
namespace
{

std::string ToLower(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

std::string FileExtension(const std::string& path)
{
    const size_t slash = path.find_last_of("/\\");
    const size_t dot = path.find_last_of('.');
    if (dot == std::string::npos || (slash != std::string::npos && dot < slash))
    {
        return "";
    }
    return ToLower(path.substr(dot));
}

bool FileExists(const std::string& path)
{
    std::ifstream file(path.c_str(), std::ios::binary);
    return file.good();
}

const char* ModelTypeName(ModelFactory::ModelType type)
{
    switch (type)
    {
    case ModelFactory::ModelType::TORCH:
        return "TorchScript";
    case ModelFactory::ModelType::ONNX:
        return "ONNX";
    case ModelFactory::ModelType::AUTO:
    default:
        return "AUTO";
    }
}

bool BackendCompiled(ModelFactory::ModelType type)
{
    switch (type)
    {
    case ModelFactory::ModelType::TORCH:
#ifdef USE_TORCH
        return true;
#else
        return false;
#endif
    case ModelFactory::ModelType::ONNX:
#ifdef USE_ONNX
        return true;
#else
        return false;
#endif
    case ModelFactory::ModelType::AUTO:
    default:
        return false;
    }
}

std::string BackendUnavailableMessage(ModelFactory::ModelType type, const std::string& model_path)
{
    return std::string(ModelTypeName(type)) + " backend is not compiled; cannot load model: " +
           model_path + ". Rebuild deploy with " +
           (type == ModelFactory::ModelType::TORCH ? "USE_TORCH and LibTorch" : "USE_ONNX and ONNX Runtime") +
           " support.";
}

int64_t ElementCount(const std::vector<int64_t>& shape)
{
    if (shape.empty())
    {
        return 0;
    }

    int64_t count = 1;
    for (int64_t dim : shape)
    {
        if (dim <= 0)
        {
            return -1;
        }
        count *= dim;
    }
    return count;
}

} // namespace

TorchModel::TorchModel()
{
#ifdef USE_TORCH
    torch::set_num_threads(1);
#endif
}

TorchModel::~TorchModel()
{
}

bool TorchModel::load(const std::string& model_path)
{
#ifdef USE_TORCH
    try
    {
        torch::set_num_threads(1);
        model_ = torch::jit::load(model_path);
        model_.eval();
        model_path_ = model_path;
        loaded_ = true;
        std::cout << LOGGER::INFO << "Loaded TorchScript policy model: " << model_path << std::endl;
        return true;
    }
    catch (const std::exception& e)
    {
        loaded_ = false;
        std::cout << LOGGER::ERROR << "Failed to load TorchScript policy model "
                  << model_path << ": " << e.what() << std::endl;
        return false;
    }
#else
    loaded_ = false;
    std::cout << LOGGER::ERROR << BackendUnavailableMessage(ModelFactory::ModelType::TORCH, model_path)
              << std::endl;
    return false;
#endif
}

std::vector<float> TorchModel::forward(const std::vector<std::vector<float>>& inputs)
{
    if (!loaded_)
    {
        throw std::runtime_error("TorchScript policy model is not loaded");
    }
    if (inputs.empty())
    {
        throw std::runtime_error("TorchScript policy inference received no inputs");
    }

#ifdef USE_TORCH
    try
    {
        const std::vector<float>& input = inputs.front();
        const std::vector<int64_t> shape = {1, static_cast<int64_t>(input.size())};
        torch::NoGradGuard no_grad;
        torch::set_num_threads(1);

        const torch::Tensor input_tensor = vector_to_torch(input, shape);
        const torch::Tensor output = model_.forward({input_tensor}).toTensor();
        return torch_to_vector(output);
    }
    catch (const std::exception& e)
    {
        std::cout << LOGGER::ERROR << "TorchScript policy inference failed for "
                  << model_path_ << ": " << e.what() << std::endl;
        throw;
    }
#else
    throw std::runtime_error("TorchScript backend is not compiled; cannot run policy inference");
#endif
}

#ifdef USE_TORCH
torch::Tensor TorchModel::vector_to_torch(const std::vector<float>& data,
                                          const std::vector<int64_t>& shape)
{
    return torch::tensor(data, torch::kFloat32).reshape(shape);
}

std::vector<float> TorchModel::torch_to_vector(const torch::Tensor& tensor)
{
    torch::Tensor cpu_tensor = tensor;
    if (cpu_tensor.device().type() != torch::kCPU)
    {
        cpu_tensor = cpu_tensor.to(torch::kCPU);
    }
    cpu_tensor = cpu_tensor.contiguous().to(torch::kFloat32);

    const float* data = cpu_tensor.data_ptr<float>();
    return std::vector<float>(data, data + cpu_tensor.numel());
}
#endif

ONNXModel::ONNXModel()
#ifdef USE_ONNX
    : memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
#endif
{
#ifdef USE_ONNX
    env_.reset(new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "sim2real_onnx_policy"));
#endif
}

ONNXModel::~ONNXModel()
{
#ifdef USE_ONNX
    session_.reset();
    env_.reset();
#endif
}

bool ONNXModel::load(const std::string& model_path)
{
#ifdef USE_ONNX
    try
    {
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        session_.reset(new Ort::Session(*env_, model_path.c_str(), session_options));
        setup_input_output_info();

        if (input_node_names_.empty() || output_node_names_.empty())
        {
            loaded_ = false;
            std::cout << LOGGER::ERROR << "ONNX policy model must expose at least one input and one output: "
                      << model_path << std::endl;
            return false;
        }

        model_path_ = model_path;
        loaded_ = true;
        std::cout << LOGGER::INFO << "Loaded ONNX policy model: " << model_path << std::endl;
        return true;
    }
    catch (const std::exception& e)
    {
        loaded_ = false;
        std::cout << LOGGER::ERROR << "Failed to load ONNX policy model "
                  << model_path << ": " << e.what() << std::endl;
        return false;
    }
#else
    loaded_ = false;
    std::cout << LOGGER::ERROR << BackendUnavailableMessage(ModelFactory::ModelType::ONNX, model_path)
              << std::endl;
    return false;
#endif
}

std::vector<float> ONNXModel::forward(const std::vector<std::vector<float>>& inputs)
{
    if (!loaded_)
    {
        throw std::runtime_error("ONNX policy model is not loaded");
    }
    if (inputs.empty())
    {
        throw std::runtime_error("ONNX policy inference received no inputs");
    }

#ifdef USE_ONNX
    try
    {
        const std::vector<float>& input = inputs.front();
        if (input.empty())
        {
            throw std::runtime_error("ONNX policy inference received an empty input tensor");
        }

        std::vector<int64_t> input_shape = input_shapes_.empty()
            ? std::vector<int64_t>{1, static_cast<int64_t>(input.size())}
            : input_shapes_.front();
        const int64_t expected_input_count = ElementCount(input_shape);
        if (expected_input_count != static_cast<int64_t>(input.size()))
        {
            throw std::runtime_error("ONNX policy input size mismatch: model expects " +
                                     std::to_string(expected_input_count) +
                                     " values, got " + std::to_string(input.size()));
        }

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info_,
            const_cast<float*>(input.data()),
            input.size(),
            input_shape.data(),
            input_shape.size());

        const char* input_names[] = {input_node_names_.front().c_str()};
        const char* output_names[] = {output_node_names_.front().c_str()};

        std::vector<Ort::Value> outputs = session_->Run(
            Ort::RunOptions{nullptr},
            input_names,
            &input_tensor,
            1,
            output_names,
            1);

        return extract_output_data(outputs);
    }
    catch (const std::exception& e)
    {
        std::cout << LOGGER::ERROR << "ONNX policy inference failed for "
                  << model_path_ << ": " << e.what() << std::endl;
        throw;
    }
#else
    throw std::runtime_error("ONNX backend is not compiled; cannot run policy inference");
#endif
}

#ifdef USE_ONNX
void ONNXModel::setup_input_output_info()
{
    Ort::AllocatorWithDefaultOptions allocator;

    const size_t input_count = session_->GetInputCount();
    input_node_names_.reserve(input_count);
    input_shapes_.reserve(input_count);
    for (size_t i = 0; i < input_count; ++i)
    {
        Ort::AllocatedStringPtr input_name = session_->GetInputNameAllocated(i, allocator);
        input_node_names_.push_back(input_name.get());

        Ort::TypeInfo type_info = session_->GetInputTypeInfo(i);
        std::vector<int64_t> shape = type_info.GetTensorTypeAndShapeInfo().GetShape();
        for (int64_t& dim : shape)
        {
            if (dim < 0)
            {
                dim = 1;
            }
        }
        input_shapes_.push_back(shape);
    }

    const size_t output_count = session_->GetOutputCount();
    output_node_names_.reserve(output_count);
    output_shapes_.reserve(output_count);
    for (size_t i = 0; i < output_count; ++i)
    {
        Ort::AllocatedStringPtr output_name = session_->GetOutputNameAllocated(i, allocator);
        output_node_names_.push_back(output_name.get());

        Ort::TypeInfo type_info = session_->GetOutputTypeInfo(i);
        std::vector<int64_t> shape = type_info.GetTensorTypeAndShapeInfo().GetShape();
        for (int64_t& dim : shape)
        {
            if (dim < 0)
            {
                dim = 1;
            }
        }
        output_shapes_.push_back(shape);
    }
}

std::vector<float> ONNXModel::extract_output_data(const std::vector<Ort::Value>& outputs)
{
    if (outputs.empty())
    {
        throw std::runtime_error("ONNX policy model returned no outputs");
    }

    const Ort::Value& output = outputs.front();
    const float* output_data = output.GetTensorData<float>();
    const std::vector<int64_t> output_shape = output.GetTensorTypeAndShapeInfo().GetShape();

    int64_t output_count = 1;
    for (int64_t dim : output_shape)
    {
        if (dim <= 0)
        {
            throw std::runtime_error("ONNX policy output has unresolved dynamic dimension");
        }
        output_count *= dim;
    }
    if (output_count <= 0)
    {
        throw std::runtime_error("ONNX policy output is empty");
    }

    return std::vector<float>(output_data, output_data + output_count);
}
#endif

std::unique_ptr<Model> ModelFactory::create_model(ModelType type)
{
    switch (type)
    {
    case ModelType::TORCH:
        return std::unique_ptr<Model>(new TorchModel());
    case ModelType::ONNX:
        return std::unique_ptr<Model>(new ONNXModel());
    case ModelType::AUTO:
    default:
        return nullptr;
    }
}

ModelFactory::ModelType ModelFactory::detect_model_type(const std::string& model_path)
{
    const std::string extension = FileExtension(model_path);
    if (extension == ".pt" || extension == ".pth")
    {
        return ModelType::TORCH;
    }
    if (extension == ".onnx")
    {
        return ModelType::ONNX;
    }
    throw std::runtime_error("Unknown policy model extension '" + extension +
                             "' for " + model_path +
                             ". Supported extensions: .pt, .pth, .onnx");
}

std::unique_ptr<Model> ModelFactory::load_model(const std::string& model_path,
                                                ModelType type,
                                                std::string* error)
{
    if (error)
    {
        error->clear();
    }

    try
    {
        if (type == ModelType::AUTO)
        {
            type = detect_model_type(model_path);
        }

        if (!FileExists(model_path))
        {
            const std::string message = "policy model file not found: " + model_path;
            if (error)
            {
                *error = message;
            }
            std::cout << LOGGER::ERROR << message << std::endl;
            return nullptr;
        }

        if (!BackendCompiled(type))
        {
            const std::string message = BackendUnavailableMessage(type, model_path);
            if (error)
            {
                *error = message;
            }
            std::cout << LOGGER::ERROR << message << std::endl;
            return nullptr;
        }

        std::unique_ptr<Model> model = create_model(type);
        if (!model)
        {
            const std::string message = "unable to create policy model backend for: " + model_path;
            if (error)
            {
                *error = message;
            }
            std::cout << LOGGER::ERROR << message << std::endl;
            return nullptr;
        }

        if (!model->load(model_path))
        {
            const std::string message = "failed to load " + std::string(ModelTypeName(type)) +
                                        " policy model: " + model_path;
            if (error)
            {
                *error = message;
            }
            return nullptr;
        }

        return model;
    }
    catch (const std::exception& e)
    {
        if (error)
        {
            *error = e.what();
        }
        std::cout << LOGGER::ERROR << "Failed to load policy model "
                  << model_path << ": " << e.what() << std::endl;
        return nullptr;
    }
}

} // namespace InferenceRuntime
