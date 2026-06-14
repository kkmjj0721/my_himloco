/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef RL_SDK_HPP
#define RL_SDK_HPP

#include <iostream>
#include <string>
#include <exception>
#include <unistd.h>
#include <algorithm>
#include <tbb/concurrent_queue.h>
#include <vector>
#include <memory>
#include <fstream>
#include <mutex>

#include <yaml-cpp/yaml.h>
#include "fsm.hpp"
#include "observation_buffer.hpp"
#include "vector_math.hpp"
#include "inference_runtime.hpp"
#include "logger.hpp"


// 机器人控制指令模板类（下发给电机的指令）
template <typename T>
struct RobotCommand
{
    struct MotorCommand
    {
        std::vector<int> mode; // 控制模式（例如：位置控制、力矩控制、阻抗控制）
        std::vector<T> q;      // 目标关节角度 (Position)
        std::vector<T> dq;     // 目标关节速度 (Velocity)
        std::vector<T> tau;    // 目标前馈力矩 (Torque)
        std::vector<T> kp;     // 位置比例增益 (Stiffness / 刚度)
        std::vector<T> kd;     // 速度微分增益 (Damping / 阻尼)
        
        // 根据机器人的关节数量初始化数组大小
        void resize(size_t num_joints)
        {
            mode.resize(num_joints, 0);
            q.resize(num_joints, 0.0f);
            dq.resize(num_joints, 0.0f);
            tau.resize(num_joints, 0.0f);
            kp.resize(num_joints, 0.0f);
            kd.resize(num_joints, 0.0f);
        }
    } motor_command;
};


// 机器人状态模板类（从传感器/电机读取的数据）
template <typename T>
struct RobotState
{
    struct IMU
    {
        std::vector<T> quaternion = {1.0f, 0.0f, 0.0f, 0.0f};       // 躯干姿态四元数 (w, x, y, z)
        std::vector<T> gyroscope = {0.0f, 0.0f, 0.0f};              // 陀螺仪：三轴角速度
        std::vector<T> accelerometer = {0.0f, 0.0f, 0.0f};          // 加速度计：三轴线加速度
    } imu;

    // 电机反馈状态
    struct MotorState
    {
        std::vector<T> q;       // 当前关节实际角度
        std::vector<T> dq;      // 当前关节实际速度
        std::vector<T> ddq;     // 当前关节实际加速度
        std::vector<T> tau_est; // 估计的关节实际输出力矩
        std::vector<T> cur;     // 电机电流

        void resize(size_t num_joints)
        {
            q.resize(num_joints, 0.0f);
            dq.resize(num_joints, 0.0f);
            ddq.resize(num_joints, 0.0f);
            tau_est.resize(num_joints, 0.0f);
            cur.resize(num_joints, 0.0f);
        }
    } motor_state;
};


// 用户输入与控制映射
namespace Input
{
    // ================== 键盘按键枚举 ==================
    // 推荐的键位映射说明：
    // Num0: 站起 | Num9: 趴下 | N: 切换导航模式 (自动驾驶)
    // R: 重置仿真 | Enter: 暂停/开始仿真
    // M: 开启电机 | K: 关闭电机 | P: 电机进入被动(阻尼)模式防摔倒
    // Num1: 基础移动模式 | Num2-Num8: 各种特殊技能(如跳跃、作揖等)
    // WSADQE: 控制线速度和角速度 | Space: 速度归零急刹
    enum class Keyboard
    {
        None = 0,               // 默认状态：没有按键被按下
        // 26个英文字母按键
        A, B, C, D, E, F, G, H, I, J, K, L, M,
        N, O, P, Q, R, S, T, U, V, W, X, Y, Z,
        // 键盘上方的数字键 0-9
        Num0, Num1, Num2, Num3, Num4, Num5, Num6, Num7, Num8, Num9,
        // 特殊控制键：空格、回车、ESC退出键
        Space, Enter, Escape,
        // 方向键：上下左右
        Up, Down, Left, Right
    };

    // ================== 手柄按键枚举 ==================
    // 推荐的键位映射说明：
    // A: 站起 | B: 趴下 | X: 切换导航模式 | Y: 暂无功能
    // RB + Y: 重置仿真 | RB + X: 暂停/恢复仿真
    // LB + A: 电机使能 | LB + B: 关闭电机 | LB + X: 电机进入被动阻尼模式
    // RB + 十字键上: 基础移动模式 | RB/LB + 十字键其他方向: 触发各种特殊技能
    // 左摇杆Y轴(LY): 控制前进后退 | 左摇杆X轴(LX): 控制左右平移 | 右摇杆X轴(RX): 控制原地转向(Yaw)
    
    // 使用强类型枚举定义游戏手柄的按键状态
    enum class Gamepad
    {
        None = 0,               // 默认状态：没有按键被按下
        // 基础单键：A/B/X/Y，左肩键(LB), 右肩键(RB), 左摇杆按下(LStick), 右摇杆按下(RStick), 十字键(DPad)上下左右
        A, B, X, Y, LB, RB, LStick, RStick, DPadUp, DPadDown, DPadLeft, DPadRight,
        // 组合键系列一：按住 LB (左肩键) 的同时按下其他键
        LB_A, LB_B, LB_X, LB_Y, LB_LStick, LB_RStick, LB_DPadUp, LB_DPadDown, LB_DPadLeft, LB_DPadRight,
        // 组合键系列二：按住 RB (右肩键) 的同时按下其他键
        RB_A, RB_B, RB_X, RB_Y, RB_LStick, RB_RStick, RB_DPadUp, RB_DPadDown, RB_DPadLeft, RB_DPadRight,
        // 特殊组合键：同时按下左肩键和右肩键
        LB_RB
    };
}


// 运行时控制状态结构体
struct Control
{
    // 记录当前和上一次的按键，用于边缘检测(比如按键刚刚按下的瞬间)
    Input::Keyboard current_keyboard = Input::Keyboard::None, last_keyboard = Input::Keyboard::None;
    Input::Gamepad current_gamepad = Input::Gamepad::None, last_gamepad = Input::Gamepad::None;

    // 机器人期望速度目标
    float x = 0.0f;
    float y = 0.0f;
    float yaw = 0.0f;

    // 更新键盘状态的方法
    void SetKeyboard(Input::Keyboard keyboad)
    {
        if (current_keyboard != keyboad)
        {
            last_keyboard = current_keyboard;
            current_keyboard = keyboad;
        }
    }

    // 更新手柄状态的方法
    void SetGamepad(Input::Gamepad gamepad)
    {
        if (current_gamepad != gamepad)
        {
            last_gamepad = current_gamepad;
            current_gamepad = gamepad;
        }
    }

    // 清除瞬时输入（防止按键卡死）
    void ClearInput()
    {
        current_keyboard = last_keyboard;
        current_gamepad = Input::Gamepad::None;
    }
};


// 包装 yaml-cpp 的结构体，提供更安全的配置读取接口
struct YamlParams
{
    // 存储读取到的 config.yaml 节点
    YAML::Node config_node;

    // 根据键名 (key) 获取对应的配置值
    // 警告 (WARNING)：当读取的数据类型是容器类（如 std::vector 时），
    // 必须先把返回值存入一个局部变量中，然后再使用迭代器或引用：
    template<typename T>
    T Get(const std::string& key, const T& default_value = T()) const
    {
        if (config_node[key])
        {
            return config_node[key].as<T>();
        }
        return default_value;
    }

    // 检查配置文件中是否包含某个特定的键名
    bool Has(const std::string& key) const
    {   
        // 调用 yaml-cpp 底层的 IsDefined() 方法判断该节点是否被定义
        return config_node[key].IsDefined();
    }
};


// 强化学习网络需要的观测值（Observation）载体
template <typename T>
struct Observations
{
    std::vector<T> commands;            // 3
    std::vector<T> ang_vel;             // 3
    std::vector<T> gravity_vec;         // 3
    std::vector<T> dof_pos;             // 12
    std::vector<T> dof_vel;             // 12
    std::vector<T> actions;             // 12

    std::vector<T> base_quat;           // 基座四元数
};


// 核心控制大类
class RL
{
public:
    RL() {};
    ~RL() {};

    YamlParams params;                  // 全局配置参数
    Observations<float> obs;            // 当前的观测状态
    std::vector<int> obs_dims;          // 记录不同观测特征的维度大小

    RobotState<float> robot_state;              // 机器人的最新硬件状态
    RobotCommand<float> robot_command;          // 准备下发给机器人的硬件指令

    // ================= 并发队列 (核心多线程通信机制) =================
    // 并发队列 (非常关键的架构设计)：
    // 由于强化学习神经网络的推理（Inference）往往比较耗时，通常放在独立的线程运行。
    // 而底层电机通信（如以太网/串口）要求极高的实时性，在主循环运行。
    // 使用 tbb::concurrent_queue 可以安全地在两个线程间传递计算好的目标位置和速度，避免加锁导致的阻塞延迟。
    tbb::concurrent_queue<std::vector<float>> output_dof_pos_queue;
    tbb::concurrent_queue<std::vector<float>> output_dof_vel_queue;
    tbb::concurrent_queue<std::vector<float>> output_dof_tau_queue;

    FSM fsm;                                        // 有限状态机管理器
    RobotState<float> start_state;                  // 状态机切换时记录的起始状态（用于插值缓动） 
    RobotState<float> now_state;                    // 当前状态
    bool rl_init_done = false;                      // 标记初始化是否完成

    // ================= 初始化相关函数声明 =================
    void InitObservations();                                        // 初始化观测值（清零或赋予默认值）
    void InitOutputs();                                             // 初始化网络输出相关的数组
    void InitControl();                                             // 初始化遥控器/键盘的控制输入状态
    void InitRL(std::string robot_config_path);                     // 核心初始化入口：传入机器人配置路径，统筹调用以上初始化并加载神经网络
    void InitJointNum(size_t num_joints);                           // 根据机器人的自由度（DOF）数量，动态分配各类数组的内存大小

    // ================= 核心虚函数（必须由继承它的真实机器人/仿真器子类来实现） =================
    // 触发神经网络前向推理（推理）
    virtual std::vector<float> Forward() = 0;
    // 计算神经网络输入特征
    std::vector<float> ComputeObservation();
    // 从具体的硬件SDK或仿真引擎获取状态
    virtual void GetState(RobotState<float> *state) = 0;
    // 将计算好的指令下发给具体的硬件SDK或仿真引擎
    virtual void SetCommand(const RobotCommand<float> *command) = 0;

    // 通用状态控制器和输出计算（实现在 rl_sdk.cpp 中）
    void StateController(const RobotState<float> *state, RobotCommand<float> *command);                 // 根据键盘输入和状态机更新机器人当前的期望运动方向
    void ComputeOutput(const std::vector<float> &actions, std::vector<float> &output_dof_pos, std::vector<float> &output_dof_vel, std::vector<float> &output_dof_tau);      // 将神经网络输出的原始 Action（-1~1之间）转化为物理世界真实的期望位置、速度和力矩

    // ================= 配置解析 =================
    void ReadYaml(const std::string& file_path, const std::string& file_name);

    // ================= CSV 日志记录模块 =================
    // Sim-to-Real (仿真到现实) 调试中最重要的一环，用于记录真实电机反馈的数据以便进行分析
    std::string csv_filename;
    void CSVInit(std::string robot_name);
    void CSVLogger(const std::vector<float> &torque, const std::vector<float> &tau_est, const std::vector<float> &joint_pos, const std::vector<float> &joint_pos_target, const std::vector<float> &joint_vel);

    // ================= 外部输入控制 =================
    Control control;                            // 实例化控制结构体，存放当前的 XYZ 速度期望
    void KeyboardInterface();                   // 在终端中非阻塞地监听并解析用户的输入
\
    // 历史观测缓冲区 (用于支持含有延迟或序列需求的 RNN/History 策略)
    ObservationBuffer history_obs_buf;
    std::vector<float> history_obs;

    // ================= 其他杂项与状态追踪 =================
    int motiontime = 0;                                     // 整个系统运行的控制周期计数器（如 500Hz 下，每运行一次加 1）
    std::string robot_name, config_name;                    // 机器人型号和配置名
    bool simulation_running = true;                         // 针对 MuJoCo 等仿真环境的标志位，用于控制仿真的暂停与继续
    std::string ang_vel_axis = "body";                      // 角速度坐标系定义："world"（世界坐标系）或 "body"（机体坐标系），用于抹平不同仿真器之间的坐标系差异
    unsigned long long episode_length_buf = 0;              // 当前环境 episode 运行的总长度缓冲                             
    int InverseJointMapping(int idx) const;                 // 逆向关节映射函数：将 SDK 中电机的索引号，反向映射为训练神经网络时定义的关节顺序号

    // ================= 安全保护机制 =================
    void TorqueProtect(const std::vector<float> &origin_output_dof_tau);                                                // 力矩超限保护：计算出的期望力矩太大时触发断电或趴下
    void AttitudeProtect(const std::vector<float> &quaternion, float pitch_threshold, float roll_threshold);            // 姿态保护：检测到机体横滚/俯仰角过大（即快要摔倒）时强制趴下，保护电机和结构件

    // ================= 强化学习模型推理 =================
    std::unique_ptr<InferenceRuntime::Model> model;             // 神经网络推理基类的智能指针。通过它可以多态地加载 LibTorch (PyTorch) 或 ONNX 模型引擎
    
    // ================= 临时输出缓冲区 =================
    // 暂存 ComputeOutput() 函数计算出来的结果，随后会被推送到前面的并发队列中去
    std::vector<float> output_dof_tau;          // 期望的前馈力矩数组
    std::vector<float> output_dof_pos;          // 期望的关节位置数组
    std::vector<float> output_dof_vel;          // 期望的关节速度数组

    // ================= 线程安全 =================
    std::mutex model_mutex;         // 互斥锁。如果在多线程中同时有系统要加载模型或者重新读取参数，用它来锁住保证不发生数据竞争崩溃
};


// 继承自基础 FSMState 的 RL 专属状态类
// 在控制机器人时，它不能一开始直接跑 RL 模型，通常需要经历：被动模式 -> 站起(插值过渡) -> 运行RL策略 -> 趴下(插值过渡) -> 被动模式 这样的状态机流转
class RLFSMState : public FSMState
{
public:
    // 构造函数，需要传入所属的 RL 类的引用，并初始化指针
    RLFSMState(RL& rl, const std::string& name)
        : FSMState(name), rl(rl), fsm_state(nullptr), fsm_command(nullptr) {}


    RL& rl;                                     // 引用外部的 RL 核心类
    const RobotState<float> *fsm_state;         // 指向当前机器人状态的只读指针
    RobotCommand<float> *fsm_command;           // 指向要下发的指令的指针

    // 线性插值函数：用于状态切换时（例如从趴下到站立），让关节平滑地从 start_pos 运动到 target_pos
    // 防止瞬间巨大的位置误差导致电机抽搐或机器人跳起
    bool Interpolate(
        float& percent,                                 // 当前插值进度 (0.0 到 1.0)
        const std::vector<float>& start_pos,            // 起点关节位置
        const std::vector<float>& target_pos,           // 终点关节位置
        float duration_seconds,                         // 完成动作期望耗时
        const std::string& description = "",            // 日志打印描述
        bool use_fixed_gains = true                     // 是否使用高刚度的固定的 PD 参数（通常站起时需要高刚度）
    );

    // 在 RL 策略运行状态中调用，直接把并发队列里的网络输出值灌入电机指令中
    void RLControl();
};

#endif // RL_SDK_HPP
