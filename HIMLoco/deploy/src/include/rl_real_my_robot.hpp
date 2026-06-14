#ifndef RL_REAL_MY_ROBOT_HPP
#define RL_REAL_MY_ROBOT_HPP

#include "rl_sdk.hpp"
#include "observation_buffer.hpp"
#include "inference_runtime.hpp"
#include "loop.hpp"
#include "fsm_my_robot.hpp"

#include <atomic>
#include <mutex>
#include <thread>
#include <vector>
#include <string>
#include <memory>

// ==========================================
// Linux 原生手柄驱动头文件 (用于接收 Xbox 手柄)
// ==========================================
#include <linux/joystick.h>
#include <fcntl.h>
#include <unistd.h>


// ==========================================
// 电机适配器接口（完整定义见 motor_adapter_interface.hpp）
// 之前只前向声明会让模板/智能指针实例化失败，所以这里改成完整包含。
// ==========================================
#include "motor_adapter_interface.hpp"


class RL_Real : public RL
{
public:
    // --- 构造函数与析构函数 ---
    RL_Real(MotorAdapterInterface *adapter = nullptr);                  
    RL_Real(std::shared_ptr<MotorAdapterInterface> adapter);
    ~RL_Real();

    // --- 状态机与控制运行线程的生命周期管理 ---
    void Start();           // 启动控制系统：异步开启键盘、手柄、控制和模型的所有后台线程
    void Stop();            // 停止控制系统：安全终止并回收所有后台线程，下发刹车或释放指令

    // --- 动态挂载、卸载 USB转485 电机适配器 ---
    void AttachMotorAdapter(MotorAdapterInterface *adapter);
    void AttachMotorAdapter(std::shared_ptr<MotorAdapterInterface> adapter);

    // --- 电机使能与断电安全保护 ---
    bool ArmMotorAdapter();             // 电机使能（进行状态安全验证，成功后正式允许向 485 总线发送功率力矩）
    void DisarmMotorAdapter();          // 电机断电/去使能（立刻让适配器下线，保护电机）

    // --- 适配器运行状态查询接口 ---
    bool IsMotorAdapterArmed() const;                   // 查询当前适配器是否处于安全的功率使能状态
    bool AdapterFaultLatched() const;                   // 查询系统当前是否由于总线断线/数据异常触发了“故障自锁”
    std::string AdapterFaultReason() const;             // 获取触发故障自锁的具体文字原因（用于离线日志排查）


private:
    // --- 共享初始化（两个公开构造函数都委托过来） ---
    // 没有这个空参构造函数，line 132 `RL_Real(shared_ptr) : RL_Real()` 会编译失败
    RL_Real();

    // --- 核心强化学习基类虚函数重载 ---
    std::vector<float> Forward() override;                                  // 神经网络前向推理
    void GetState(RobotState<float> *state) override;                       // 获取机器人当前传感器状态
    void SetCommand(const RobotCommand<float> *command) override;           // 下发电机控制指令

    // --- 独立的多线程循环处理函数 (Worker Loops) ---
    void KeyboardLoop();            // 键盘监听循环
    void ControlLoop();             // 底层高频电机控制循环 （200 hz）
    void ModelLoop();               // 神经网络中频推理循环（50 hz）
    void JoystickLoop();            // Xbox手柄事件高频监听循环（50 hz）

    // --- 核心控制解算与手柄映射 ---
    void RobotControl();            // 单步控制解算：读取最新传感器、运行当前状态机逻辑、下发 485 指令
    void RunModel();                // 单步模型推算：构建当前帧的 Observation，调用网络，并将结果塞入双端同步队列
    void JoystickInterface();       // Xbox手柄键位与摇杆物理量解析映射

    // --- 数据校验、物理防护与安全自锁辅助函数 ---
    // 从适配器线程安全地刷新 IMU 和各关节的物理读数
    bool RefreshStateForControl(RobotState<float> *state); 
    // 验证适配器接收的传感器状态是否合法
    bool ValidateAdapterState(const MotorAdapterState &state, int dofs, const std::vector<int> &mapping, std::string *error) const;
    // 验证发往电机的控制目标是否合法
    bool ValidateCommand(const RobotCommand<float> *command, int dofs, std::string *error) const;

    MotorAdapterInterface *AdapterLocked() const;                     // 互斥保护下获取当前的物理适配器后端指针
    bool HasNonOwningAdapterLocked() const;                           // 检查当前使用的适配器是否为非持有的裸指针形式
    void ClearNonOwningAdapterForThreadedRunLocked(const std::string &reason); // 在多线程启动前强行卸载不安全的裸指针适配器
    void SafeStopAdapterLocked(const std::string &reason);            // 触发安全停机：令适配器立刻停发数据并执行电机抱闸/脱机
    void LatchAdapterFaultLocked(const std::string &reason);          // 锁存一个致命硬件故障（带内部互斥锁版本）
    void LatchAdapterFault(const std::string &reason);               // 锁存一个致命硬件故障（供外部/无锁域调用的公开版本）
    void EnterPassiveSafeStateLocked();                               // 强行切入被动安全状态：清空指令队列，模拟发送紧急 P 键

    // --- 线程对象管理 ---
    std::thread keyboard_thread_;  // 异步键盘监视线程句柄
    std::thread control_thread_;   // 超高频 485 物理控制和状态机同步线程句柄
    std::thread model_thread_;     // 神经网络模型在线推理线程句柄
    std::thread joystick_thread_;  // 新增：Xbox 手柄事件流独立监听线程句柄

    // --- 线程同步锁与状态标志位 ---
    std::atomic<bool> running_{false};    // 核心全局原子运行状态（True代表线程正在运行，False代表启动终止流程）
    mutable std::mutex runner_mutex_;     // 控制层（ControlLoop）与推理层（ModelLoop）之间共享变量的同步锁
    mutable std::mutex adapter_mutex_;    // 485 电机适配器挂载、写指令、读状态时的专属保护互斥锁

    // --- 电机适配器底层后端管理变量 ---
    MotorAdapterInterface *raw_adapter_ = nullptr;          // 裸指针形式的适配器地址（不管理其生命周期）
    std::shared_ptr<MotorAdapterInterface> adapter_owner_; // 智能指针形式的适配器（完全持有其生命周期管理权）
    bool adapter_armed_ = false;                            // 标志位：电机当前是否处于功率输出使能状态
    bool adapter_fault_latched_ = false;                    // 标志位：当前系统是否由于总线异常处于“硬件故障挂起”状态
    std::string adapter_fault_reason_;                      // 记录触发故障挂起的具体软件/硬件错误文本信息
    bool fresh_state_for_control_ = false;                  // 握手状态标志：确保控制循环在拿到最新的有效状态后才允许写串口

    // --- 其它控制及 Xbox 手柄底层物理变量 ---
    int joystick_fd_ = -1;                                  // Linux 操作系统下 Xbox 手柄设备的文件描述符
    std::vector<float> pre_clamp_output_dof_tau;            // 物理限幅前原始计算力矩缓存（用于送入 TorqueProtect 进行危险力矩诊断）

};

#endif // RL_REAL_MY_ROBOT_HPP
