#ifndef MOTOR_ADAPTER_INTERFACE_HPP
#define MOTOR_ADAPTER_INTERFACE_HPP

#include <string>
#include <vector>


// ===================================================================
// 电机适配器统一接口（机器人硬件抽象层 / Hardware Abstraction Layer）
//
// 之所以单独抽出这一层，是为了让上层 rl_real_my_robot.cpp 完全不感知
//   - 电机走的是 USB转485 还是 EtherCAT 还是 DDS
//   - IMU 是单独串口的还是和电机一起广播的
// 上层只调 ReadState() / WriteCommand()，背后 USB转485 + UART IMU 的
// 物理拼装由具体子类 (例如 UnitreeMotorAdapter) 来承担。
// ===================================================================



// 整机一帧的传感器反馈：IMU + 12 路电机的位置/速度/估算力矩
// 注意：dof_pos / dof_vel / dof_tau_est 是 *硬件电机ID顺序* 的，
//       上层会用 joint_mapping 再把它重排为算法顺序。
struct MotorAdapterState
{
    std::vector<float> base_quaternion;     // 四元数，长度=4, [w, x, y, z]
    std::vector<float> gyro;                // 角速度，长度=3, [wx, wy, wz] rad/s
    std::vector<float> dof_pos;             // 关节位置，长度=硬件电机数, rad
    std::vector<float> dof_vel;             // 关节速度，长度=硬件电机数, rad/s
    std::vector<float> dof_tau_est;         // 估算输出力矩，长度=硬件电机数, N·m
};


// 整机一帧的电机控制指令：每个电机的工作模式 + PD + 前馈力矩
// 同样按 *硬件电机ID顺序*，由上层 SetCommand() 完成反向重排。
struct MotorAdapterCommand
{
    std::vector<int>   mode;                // 0 = BRAKE (软刹+阻尼)；1 = FOC (位置/速度/力矩闭环)
    std::vector<float> q;                   // 期望关节角 rad
    std::vector<float> dq;                  // 期望关节速度 rad/s
    std::vector<float> kp;                  // 位置增益
    std::vector<float> kd;                  // 速度增益
    std::vector<float> tau;                 // 前馈力矩 N·m
};


// 电机适配器纯虚基类
//
// 生命周期约定（必须严格遵守，否则会触发 rl_real_my_robot.cpp 内的故障自锁）:
//   1) 构造完成后 IsReady() 必须返回 true 才能 Arm()
//   2) Arm() 成功后 IsArmed() 才返回 true，此时才允许 WriteCommand()
//   3) Disarm() / SafeStop() 之后必须立刻使 IsArmed() 返回 false 并广播 BRAKE
//   4) ReadState() 必须线程安全；WriteCommand() 必须线程安全；
//      但允许实现内部串口操作互斥（不强制实时锁外）
class MotorAdapterInterface
{
public:
    virtual ~MotorAdapterInterface() = default;

    // 适配器是否已初始化、串口/总线是否可用
    virtual bool IsReady() const = 0;

    // 当前是否处于功率使能状态
    virtual bool IsArmed() const = 0;

    // 使能功率输出。失败应填充 LastError() 并返回 false
    virtual bool Arm() = 0;

    // 主动断电（变 BRAKE 模式）
    virtual bool Disarm() = 0;

    // 紧急停机：广播 BRAKE，必要时关闭底层资源
    virtual bool SafeStop() = 0;

    // 读取一帧整机状态：IMU + 12 路电机
    // 返回 false 表示通信故障，调用方会自锁
    virtual bool ReadState(MotorAdapterState *state) = 0;

    // 下发一帧整机指令（已由上层做过 NaN/力矩裁剪）
    virtual bool WriteCommand(const MotorAdapterCommand &cmd) = 0;

    // 最近一次失败的人类可读原因（用于日志/串口断线诊断）。
    // 没有错误时返回 nullptr 或空 C 字符串均可。
    virtual const char *LastError() const = 0;
};


#endif // MOTOR_ADAPTER_INTERFACE_HPP