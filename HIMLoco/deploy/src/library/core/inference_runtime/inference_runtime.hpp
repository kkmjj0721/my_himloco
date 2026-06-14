#ifndef INFERENCE_RUNTIME_HPP
#define INFERENCE_RUNTIME_HPP

#include <vector>
#include <string>
#include <memory>
#include <algorithm>
#include "logger.hpp"

#ifdef USE_TORCH
#include <torch/script.h>
#endif

#ifdef USE_ONNX
#include <onnxruntime_cxx_api.h>
#endif

namespace InferenceRuntime
{

/**
 * @brief 模型接口基类
 */
class Model
{
public:
    virtual ~Model() = default;

    /**
     * @brief 加载模型文件
     * @param model_path 模型文件路径
     * @return Returns true if loading succeeds, false if it fails
     */
    virtual bool load(const std::string& model_path) = 0;

    /**
     * @brief 检测模型是否已成功加载
     * @return Returns true if loaded, false otherwise
     */
    virtual bool is_loaded() const = 0;

    /**
     * @brief 前向推理
     * @param inputs Vector of input data vectors (usually single element)
     * @return Inference result vector
     */
    virtual std::vector<float> forward(const std::vector<std::vector<float>>& inputs) = 0;

    /**
     * @brief 获取模型类型
     * @return Model type ("torch" or "onnx")
     */
    virtual std::string get_model_type() const = 0;
};

/**
 * @brief Torch模型实现类
 *
 * Model inference implementation based on PyTorch TorchScript
 */
class TorchModel : public Model
{
private:
    bool loaded_ = false;               // 模型是否已加载
    std::string model_path_;            // 模型文件路径

#ifdef USE_TORCH
    torch::jit::script::Module model_;  // 模型对象
#endif

public:
    TorchModel();
    ~TorchModel();

    bool load(const std::string& model_path) override;          // 重写父类的load方法      
    bool is_loaded() const override { return loaded_; }         // 重写父类的is_loaded方法
    std::vector<float> forward(const std::vector<std::vector<float>>& inputs) override;         // 重写父类的forward方法
    std::string get_model_type() const override { return "torch"; }                             // 内联重写：写死返回 "torch"

private:
#ifdef USE_TORCH
    /**
     * @brief 转换为Torch张量
     * @param data Input data vector
     * @param shape Tensor shape
     * @return Torch tensor
     */
    torch::Tensor vector_to_torch(const std::vector<float>& data, const std::vector<int64_t>& shape);

    /**
     * @brief 转换Torch张量为数据向量
     * @param tensor Input tensor
     * @return Data vector
     */
    std::vector<float> torch_to_vector(const torch::Tensor& tensor);
#endif
};

/**
 * @brief ONNX 模型实现类
 *
 * Model inference implementation based on ONNX Runtime
 */
class ONNXModel : public Model
{
private:
    bool loaded_ = false;               // 模型是否已加载
    std::string model_path_;            // 模型文件路径

#ifdef USE_ONNX
    std::unique_ptr<Ort::Session> session_;                 // ONNX 推理会话对象，管理整个推理过程
    std::unique_ptr<Ort::Env> env_;                         // ONNX 运行环境对象，管理全局状态（如多线程等）
    Ort::MemoryInfo memory_info_;                           // ONNX 内存信息对象，用于管理 CPU/GPU 内存分配方式
    std::vector<std::string> input_node_names_;             // 缓存输入节点的名称，ONNX 推理时需要传入
    std::vector<std::string> output_node_names_;            // 缓存输出节点的名称，告诉 ONNX 需要返回哪些节点的结果
    std::vector<std::vector<int64_t>> input_shapes_;        // 模型输入张量形状
    std::vector<std::vector<int64_t>> output_shapes_;       // 模型输出张量形状
#endif

public:
    ONNXModel();
    ~ONNXModel();

    bool load(const std::string& model_path) override;          // 重写父类的load方法
    bool is_loaded() const override { return loaded_; }         // 重写父类的is_loaded方法
    std::vector<float> forward(const std::vector<std::vector<float>>& inputs) override;         // 重写父类的forward方法
    std::string get_model_type() const override { return "onnx"; }                              // 内联重写：写死返回 "onnx"

private:
#ifdef USE_ONNX
    /**
     * @brief 设置输入，输出节点信息
     */
    void setup_input_output_info();

    /**
     * @brief 从 onnx 输出信息中提取数据
     * @param outputs ONNX 推理输出信息
     * @return Extracted data vector
     */
    std::vector<float> extract_output_data(const std::vector<Ort::Value>& outputs);
#endif
};

/**
 * @brief 模型工厂类
 *
 * 负责创建和加载不同类型的模型
 */
class ModelFactory
{
public:
    /**
     * @brief 模型类型枚举
     */
    enum class ModelType
    {
        TORCH,  
        ONNX,   
        AUTO    
    };

    /**
     * @brief 创建指定类型模型
     * @param type 模型类型
     * @return Model smart pointer
     */
    static std::unique_ptr<Model> create_model(ModelType type = ModelType::AUTO);

    /**
     * @brief 根据模型文件路径检查模型类型
     * @param model_path 模型路径
     * @return Detected model type
     */
    static ModelType detect_model_type(const std::string& model_path);

    /**
     * @brief 加载模型文件并返回模型对象
     * @param model_path 模型文件路径
     * @param type Model type (default: auto-detect)
     * @return Successfully loaded model smart pointer, returns nullptr on failure
     */
    static std::unique_ptr<Model> load_model(const std::string& model_path,
                                             ModelType type = ModelType::AUTO,
                                             std::string* error = nullptr);
};

} // namespace InferenceRuntime

#endif // INFERENCE_RUNTIME_HPP
