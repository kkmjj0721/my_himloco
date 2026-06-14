#ifndef OBSERVATION_BUFFER_HPP
#define OBSERVATION_BUFFER_HPP

#include <vector>
#include <string>

/**
 * @brief 用于存储历史观测数据的缓冲区
 */
class ObservationBuffer
{
public:
    /**
     * @brief 带参构造函数
     * @param num_envs 环境的总数量
     * @param obs_dims 每个观测组件的维度列表
     * @param history_length 需要保留的历史观测步数长度
     * @param priority 输出优先级模式（"time" 或 "term"）
     */
    ObservationBuffer(int num_envs, const std::vector<int>& obs_dims, int history_length, const std::string& priority);
    
    /**
     * @brief 默认构造函数
     */
    ObservationBuffer();

    /**
     * @brief 使用新的观测数据重置指定的环境
     * @param reset_idxs 需要被重置的环境索引列表
     * @param new_obs 用于重新填满缓冲区的新观测数据
     */
    void reset(std::vector<int> reset_idxs, const std::vector<float>& new_obs);
    
    /**
     * @brief 向缓冲区中插入新的观测数据
     * @param new_obs 需要插入的新观测数据
     */
    void insert(const std::vector<float>& new_obs);
    
    /**
     * @brief 根据指定的索引获取观测向量
     * @param obs_ids 指定要提取哪些观测组件的索引列表
     * @return 拼接好的最终一维观测向量
     */
    std::vector<float> get_obs_vec(std::vector<int> obs_ids);

private:
    int num_envs;                                           // 环境总数量
    std::vector<int> obs_dims;                              // 各个观测组件的维度列表
    std::string priority;                                   // 输出优先级模式
    int num_obs = 0;                                        // 单步观测数据的总维度
    int history_length = 0;                                 // 历史步数长度
    int num_obs_total = 0;                                  // 输出的观测数据总大小
    std::vector<std::vector<std::vector<float>>> obs_buf;   // 观测缓冲区，维度顺序：[环境索引][时间步][观测值]
};


#endif // OBSERVATION_BUFFER_HPP
