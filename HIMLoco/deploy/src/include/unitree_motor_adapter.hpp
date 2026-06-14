#ifndef UNITREE_MOTOR_ADAPTER_HPP
#define UNITREE_MOTOR_ADAPTER_HPP

#include "motor_adapter_interface.hpp"
#include "imu_driver.hpp"

#include "unitreeMotor/unitreeMotor.h"
#include "serialPort/SerialPort.h"

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <vector>


// ===================================================================
// Unitree GO-M8010-6 USB转485 总线适配器
//
// 把 Unitree 官方 SDK 的 SerialPort + MotorCmd/MotorData 协议封装成
// 上层算法侧通用的 MotorAdapterInterface。
//
// 同时把 IMUDriver 拍进来一起，让上层 ReadState() 一次拿到 IMU + 12 路
// 电机的全部传感器数据，对上层 rl_real_my_robot.cpp 完全屏蔽底层走的是
// 两根不同物理线缆这件事。
// ===================================================================


class UnitreeMotorAdapter : public MotorAdapterInterface
{
public:
    struct Config                                           // 声明内部配置结构体 Config，用于初始化适配器
    {
        std::string serial_port      = "/dev/ttyUSB0";              // 定义串口设备路径，默认为 /dev/ttyUSB0
        int         baudrate         = 4000000;                     // 定义波特率，宇树 RS485 通常使用极高的 4M 波特率
        int         timeout_us       = 300;                         // 定义串口读写超时时间，单位微秒
        int         num_motors       = 12;                          // 定义总线上挂载的电机数量，默认为 12 个（四足机器人）
        MotorType   motor_type       = MotorType::GO_M8010_6;       // 定义电机型号，默认为 GO-M8010-6
        // 极性翻转：每路电机 ±1，用来吸收实际接线方向与算法约定方向的不一致
        // 如果留空，UnitreeMotorAdapter 内部默认全 +1
        std::vector<float> direction;                       // 定义电机正反转极性配置数组
        // 电机零位偏置：raw_q - zero_offset = 算法用的 q
        std::vector<float> zero_offset;                     // 定义电机机械零点偏置数组
    };

    UnitreeMotorAdapter(Config cfg, std::shared_ptr<IMUDriver> imu); // 构造函数：传入配置结构体和共享指针管理的 IMU 驱动对象
    ~UnitreeMotorAdapter() override;                        // 析构函数：加上 override 关键字表明重写基类虚函数

    // 禁拷贝
    UnitreeMotorAdapter(const UnitreeMotorAdapter &)            = delete; // 禁用拷贝构造函数，防止底层硬件资源被意外复制
    UnitreeMotorAdapter &operator=(const UnitreeMotorAdapter &) = delete; // 禁用赋值运算符，防止底层硬件资源被意外赋值

    // MotorAdapterInterface
    bool        IsReady() const override;                   // 重写 IsReady() 方法
    bool        IsArmed() const override;                   // 重写 IsArmed() 方法
    bool        Arm() override;                             // 重写 Arm() 方法
    bool        Disarm() override;                          // 重写 Disarm() 方法
    bool        SafeStop() override;                        // 重写 SafeStop() 方法
    bool        ReadState(MotorAdapterState *state) override; // 重写 ReadState() 方法
    bool        WriteCommand(const MotorAdapterCommand &cmd) override; // 重写 WriteCommand() 方法
    const char *LastError() const override;                 // 重写 LastError() 方法

private:
    void SetError(const std::string &msg);                          // 私有方法：用于在内部更新记录最近一次的错误信息
    void BuildBrakePacket(std::vector<MotorCmd> *out) const;        // 私有方法：构建给所有电机发送刹车(Brake)指令的数据包

    Config                       cfg_;                      // 存储构造时传入的配置参数
    std::shared_ptr<IMUDriver>   imu_;                      // 存储 IMU 驱动的共享指针
    std::unique_ptr<SerialPort>  port_;                     // 存储串口对象的独占指针，负责生命周期管理
    float                        gear_ratio_ = 1.0f;        // 存储电机的减速比，默认 1.0
    int                          mode_foc_   = 0;           // 存储当前电机对应的 FOC 模式控制字编码
    int                          mode_brake_ = 0;           // 存储当前电机对应的 BRAKE 模式控制字编码

    mutable std::mutex bus_mutex_;                          // 定义互斥锁，用于保护 RS485 串口总线，mutable 允许在 const 函数中加锁
    std::atomic<bool>  ready_{false};                       // 原子布尔量：标记串口是否成功打开且可用
    std::atomic<bool>  armed_{false};                       // 原子布尔量：标记电机是否已经上电使能

    mutable std::mutex last_error_mutex_;                   // 定义互斥锁，专门用于保护 last_error_ 字符串的读写
    std::string        last_error_;                         // 存储最近一次发生的错误信息文本
};


#endif // UNITREE_MOTOR_ADAPTER_HPP
