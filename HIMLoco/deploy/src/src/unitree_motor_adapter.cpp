#include "unitree_motor_adapter.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>


namespace
{
// 我们把 MotorAdapterCommand.mode 的语义统一成 0=BRAKE, 1=FOC（和 rl_real_my_robot.cpp 一致）
constexpr int kCmdModeBrake = 0;              // 编译期常量：自定义的控制指令模式 BRAKE 值为 0
constexpr int kCmdModeFoc   = 1;              // 编译期常量：自定义的控制指令模式 FOC 值为 1

bool IsFinite(float v) { return std::isfinite(v); }     // 辅助函数：判断传入的浮点数是否是有限值（即非 NaN 或无穷大）

} // namespace


// 构造函数实现
UnitreeMotorAdapter::UnitreeMotorAdapter(Config cfg, std::shared_ptr<IMUDriver> imu)
    : cfg_(std::move(cfg)),                 // 使用 std::move 高效初始化配置变量 cfg_
      imu_(std::move(imu))                  // 使用 std::move 转移 IMU 驱动指针的所有权
{
    // 检查配置中的电机数量是否合法
    if (cfg_.num_motors <= 0)
    {
        // 抛出无效参数异常
        throw std::invalid_argument("UnitreeMotorAdapter: num_motors must be > 0");
    }

    // 如果用户没有配置极性数组
    if (cfg_.direction.empty())
    {
        // 自动补全为全正极性 (1.0)
        cfg_.direction.assign(static_cast<size_t>(cfg_.num_motors), 1.0f);
    }   
    // 如果配置了极性，但长度与电机数不匹配
    else if (cfg_.direction.size() != static_cast<size_t>(cfg_.num_motors))
    {
        throw std::invalid_argument("UnitreeMotorAdapter: direction.size() != num_motors");
    }
    // 如果用户没有配置零点偏置数组
    if (cfg_.zero_offset.empty())
    {
        // 自动补全为全 0
        cfg_.zero_offset.assign(static_cast<size_t>(cfg_.num_motors), 0.0f);
    }
    // 如果配置了偏置，但长度与电机数不匹配
    else if (cfg_.zero_offset.size() != static_cast<size_t>(cfg_.num_motors))
    {
        // 抛出异常
        throw std::invalid_argument("UnitreeMotorAdapter: zero_offset.size() != num_motors");
    }

    // 查 SDK 拿到该型号电机的 FOC / BRAKE 模式整数编码 和减速比
    mode_foc_   = queryMotorMode(cfg_.motor_type, MotorMode::FOC);   // 调用宇树 SDK，获取当前型号电机 FOC 模式对应的十六进制指令字
    mode_brake_ = queryMotorMode(cfg_.motor_type, MotorMode::BRAKE); // 调用宇树 SDK，获取当前型号电机 BRAKE 模式对应的指令字
    gear_ratio_ = queryGearRatio(cfg_.motor_type);                   // 调用宇树 SDK，获取该型号电机的物理减速比
    
    // 检查减速比是否合理（防呆）
    if (gear_ratio_ <= 0.0f)    gear_ratio_ = 1.0f;         // 如果 SDK 返回异常值，默认设置为 1.0

    try
    {
        // 第三个参数 recvLength=16 是 SDK header 提供的 GO-M8010-6 报文长度
        port_ = std::make_unique<SerialPort>(                        // 创建独占型智能指针管理宇树串口对象
            cfg_.serial_port,                                        // 串口路径 (如 /dev/ttyUSB0)
            /*recvLength=*/16,                                       // 指定每帧接收的字节长度，GO-M8010 协议规定为 16 字节
            static_cast<uint32_t>(cfg_.baudrate),                    // 波特率转为无符号 32 位整型
            static_cast<size_t>(cfg_.timeout_us),                    // 超时时间转为 size_t
            BlockYN::NO);                                            // 非阻塞模式，保证实时性，收不到包立刻返回
        ready_ = true;                                               // 如果串口对象创建成功且未抛异常，将 ready 状态置为 true
    }
    // 捕获串口打开过程中可能抛出的标准异常
    catch (const std::exception &e)
    {
        ready_ = false;                                              // 初始化失败，ready 状态置为 false
        SetError(std::string("SerialPort open failed: ") + e.what());// 拼接错误原因并记录到内部错误信息中
    }
}

// 析构函数实现
UnitreeMotorAdapter::~UnitreeMotorAdapter()
{   
    // 对象被销毁前，安全地调用停止函数（刹车）
    SafeStop();
}

// IsReady 实现
bool UnitreeMotorAdapter::IsReady() const
{
    // 以线程安全的方式读取 ready_ 原子变量
    return ready_.load();
}

// IsArmed 实现
bool UnitreeMotorAdapter::IsArmed() const
{
    // 以线程安全的方式读取 armed_ 原子变量
    return armed_.load();
}

// Arm（使能）实现
bool UnitreeMotorAdapter::Arm()
{
    // 如果底层串口没准备好
    if (!ready_)
    {
        SetError("Arm() called but adapter not ready (serial open failed)");
        return false;
    }

    // 发一遍纯阻尼心跳包：所有电机 FOC + 0 增益 + 0 力矩；任何一路没回包都算失败
    std::vector<MotorCmd> tx(cfg_.num_motors);  // 创建用于发送(Transmit)的电机指令数组，长度为电机数
    std::vector<MotorData> rx(cfg_.num_motors); // 创建用于接收(Receive)的电机状态数组，长度为电机数

    for (int i = 0; i < cfg_.num_motors; ++i)   // 遍历每一路电机
    {
        tx[i].motorType = cfg_.motor_type;      // 写入期望的电机型号
        tx[i].id        = static_cast<unsigned short>(i); // 写入硬件电机 ID (0到11)
        tx[i].mode      = static_cast<unsigned short>(mode_foc_); // 设为 FOC 模式控制字
        tx[i].q  = 0.0f;                        // 期望位置置 0
        tx[i].dq = 0.0f;                        // 期望速度置 0
        tx[i].kp = 0.0f;                        // 位置刚度置 0
        tx[i].kd = 0.0f;                        // 速度阻尼置 0 (这里是0阻尼，实际上是松软模式)
        tx[i].tau = 0.0f;                       // 前馈力矩置 0

        rx[i].motorType = cfg_.motor_type;      // 为接收区结构体指明待接收的电机型号
    }

    std::lock_guard<std::mutex> lock(bus_mutex_); // 对串口总线加锁，防止其他线程（如 ReadState）同时抢占串口读写

    // 调用宇树串口 SDK 一次性发送并接收所有电机数据
    if (!port_->sendRecv(tx, rx))               
    {
        SetError("Arm(): sendRecv heartbeat failed"); // 如果通信失败，记录错误
        return false;                           // 使能失败
    }

    // 遍历检查返回的每个电机数据
    for (int i = 0; i < cfg_.num_motors; ++i)
    {   
        // 检查返回值里是否有 NaN (Not a Number) 或 Infinity
        if (!IsFinite(rx[i].q) || !IsFinite(rx[i].dq))
        {   
            // 记录异常电机的 ID
            SetError("Arm(): motor " + std::to_string(i) + " returned non-finite state");
            // 只要有一个电机回传乱码，就立刻判定使能失败，保障安全
            return false;
        }
    }

    armed_.store(true);                         // 全部检查通过，安全上电，修改 armed 状态为 true
    return true;                                // 返回使能成功
}

// Disarm (断电解使能) 实现
bool UnitreeMotorAdapter::Disarm()
{
    if (!ready_)                              // 如果连串口都没打开
    {
        armed_.store(false);                  // 保险起见依然把状态置 false
        return true;                          // 无需发指令，直接算断电成功
    }
    std::vector<MotorCmd> tx;                 // 创建发送指令包数组
    BuildBrakePacket(&tx);                    // 专门调用私有方法组装 "全车刹车" 报文

    std::vector<MotorData> rx(cfg_.num_motors); // 创建接收状态包数组
    for (int i = 0; i < cfg_.num_motors; ++i) 
        rx[i].motorType = cfg_.motor_type;      // 指明接收型号

    std::lock_guard<std::mutex> lock(bus_mutex_); // 总线加互斥锁
    const bool ok = port_->sendRecv(tx, rx);    // 下发刹车报文并收回状态

    armed_.store(false);                      // 无论硬件是否响应，软件侧立刻置为断电状态
    if (!ok) SetError("Disarm(): brake broadcast failed"); // 若通信失败，记录日志
    return ok;                                // 返回通信是否成功
}

// 紧急停止实现
bool UnitreeMotorAdapter::SafeStop()
{
    const bool ok = Disarm();            
    // 不关串口，留给 Arm() 再次复用；调用方析构时 port_ unique_ptr 自动 close
    return ok;
}

// 读取整机状态实现
bool UnitreeMotorAdapter::ReadState(MotorAdapterState *state)
{   
    // 防空指针校验
    if (!state)            return false;

    // 检查底层是否初始化成功
    if (!ready_)
    {
        SetError("ReadState(): adapter not ready");
        return false;
    }

    // 1) 读电机：发一遍当前应当下发的指令并同步收回数据。
    //    Arm 之后正常控制流程会用 WriteCommand() 把最新指令写下去，
    //    这里独立的 ReadState() 用纯阻尼心跳，效果等价于 RS-485 总线轮询。
    std::vector<MotorCmd>  tx(cfg_.num_motors);
    std::vector<MotorData> rx(cfg_.num_motors);
    for (int i = 0; i < cfg_.num_motors; ++i)
    {
        tx[i].motorType = cfg_.motor_type;
        tx[i].id        = static_cast<unsigned short>(i);
        tx[i].mode      = static_cast<unsigned short>(armed_ ? mode_foc_ : mode_brake_);
        tx[i].q  = 0.0f;
        tx[i].dq = 0.0f;
        tx[i].kp = 0.0f;
        tx[i].kd = 0.0f;
        tx[i].tau = 0.0f;
        rx[i].motorType = cfg_.motor_type;
    }

    {
        std::lock_guard<std::mutex> lock(bus_mutex_); // 锁定总线操作
        if (!port_->sendRecv(tx, rx))           // 硬件收发
        {
            SetError("ReadState(): sendRecv failed"); // 失败记录日志
            return false;                       // 读取失败触发自锁
        }
    }

    state->dof_pos.assign(static_cast<size_t>(cfg_.num_motors), 0.0f);     // 为输出 state 的 position 数组开辟并清空内存
    state->dof_vel.assign(static_cast<size_t>(cfg_.num_motors), 0.0f);     // 为输出 state 的 velocity 数组开辟并清空内存
    state->dof_tau_est.assign(static_cast<size_t>(cfg_.num_motors), 0.0f); // 为输出 state 的 torque 数组开辟并清空内存

    for (int i = 0; i < cfg_.num_motors; ++i)   // 对返回的每一路电机数据进行解析运算
    {
        const float dir = cfg_.direction[static_cast<size_t>(i)];          // 取出当前电机的极性方向（+1 或 -1）
        // 减速比折算：电机轴 q/dq 除以 gear_ratio 得到关节侧；电机轴 tau × gear_ratio 得到关节侧
        // 极性翻转保证算法侧约定的"伸展为正"方向与硬件一致
        state->dof_pos[i]     = dir * (rx[i].q  / gear_ratio_) - cfg_.zero_offset[static_cast<size_t>(i)]; // 计算关节端实际位置
        state->dof_vel[i]     = dir * rx[i].dq / gear_ratio_;                                             // 计算关节端实际速度
        state->dof_tau_est[i] = dir * rx[i].tau * gear_ratio_;                                            // 计算关节端实际力矩
    }

    // 2) 读 IMU：用最新一帧快照填进同一个 state 结构体
    std::array<float, 4> quat{1.0f, 0.0f, 0.0f, 0.0f}; // 预分配数组存四元数，默认填入无旋转的单位四元数
    std::array<float, 3> gyro{0.0f, 0.0f, 0.0f};       // 预分配数组存角速度，默认 0
    std::array<float, 3> accel{0.0f, 0.0f, 0.0f};      // 预分配数组存加速度，默认 0
    if (imu_ && imu_->Snapshot(quat, gyro, accel))     // 如果上层传入了 IMU 驱动且能够成功拿到快照
    {
        state->base_quaternion = {quat[0], quat[1], quat[2], quat[3]}; // 将四元数拷贝到状态结构体
        state->gyro            = {gyro[0], gyro[1], gyro[2]};          // 将角速度拷贝到状态结构体
    }

    else                                               // 拿不到 IMU 数据时的异常处理策略
    {
        // 没有 IMU 句柄就抛一个单位四元数 + 零角速度，让上层校验自己决定要不要拒绝
        // 上层的 ValidateAdapterState 会发现 quat 模长 == 1，gyro 全 0 是合法的；
        // 但如果 IsHealthy() 为 false，调用方应当走 SetError 自锁路径。
        if (imu_ && !imu_->IsHealthy())                // 如果 IMU 存在但报出不健康状态（线缆松动等硬件错误）
        {
            SetError("ReadState(): IMU stale (" + imu_->LastError() + ")"); // 提取 IMU 底层错误传递上来
            return false;                              // 因为失去平衡感知，直接返回 false 让机身锁死停机
        }
        state->base_quaternion = {1.0f, 0.0f, 0.0f, 0.0f}; // 对无物理 IMU 配置的兼容：假装是正的
        state->gyro            = {0.0f, 0.0f, 0.0f};       // 角速度 0
    }

    return true;                                       // 组装完成，返回成功
}

// 下发控制指令实现
bool UnitreeMotorAdapter::WriteCommand(const MotorAdapterCommand &cmd)
{
    if (!ready_)  { SetError("WriteCommand(): adapter not ready"); return false; } // 防御：未准备好
    if (!armed_)  { SetError("WriteCommand(): adapter disarmed");  return false; } // 防御：未处于使能上电状态

    const size_t n = static_cast<size_t>(cfg_.num_motors);          // 获取电机总数并转为 size_t

    // 严格检查算法层传下来的所有指令数组长度是否与硬件对齐
    if (cmd.q.size()   != n || cmd.dq.size()  != n ||
        cmd.kp.size()  != n || cmd.kd.size()  != n ||
        cmd.tau.size() != n || cmd.mode.size() != n)
    {
        SetError("WriteCommand(): command vector size mismatch");
        return false;
    }

    std::vector<MotorCmd>  tx(cfg_.num_motors);
    std::vector<MotorData> rx(cfg_.num_motors);

    for (int i = 0; i < cfg_.num_motors; ++i)
    {
        const float dir = cfg_.direction[static_cast<size_t>(i)]; // 取出当前电机的极性
        const int mode = (cmd.mode[static_cast<size_t>(i)] == kCmdModeFoc) ? mode_foc_ : mode_brake_; // 查表转换对应的控制字

        tx[i].motorType = cfg_.motor_type;                 // 写入目标电机类型
        tx[i].id        = static_cast<unsigned short>(i);  // 写入目标电机 ID
        tx[i].mode      = static_cast<unsigned short>(mode); // 写入工作模式控制字

        // 反向折算：算法侧关节量 → 电机轴量
        // 角度：先按零位偏置回填，再乘减速比、再翻极性
        const float q_joint = cmd.q[static_cast<size_t>(i)] + cfg_.zero_offset[static_cast<size_t>(i)]; // 第一步：将算法要求的关节角补上零位偏置
        tx[i].q   = dir * q_joint * gear_ratio_;           // 第二步：转换为转子期望角度
        tx[i].dq  = dir * cmd.dq[static_cast<size_t>(i)] * gear_ratio_; // 转子期望速度

        // PD 增益不需要乘减速比（电机驱动板按"电机轴空间"理解 PD），
        // 但物理意义上 kp_joint = kp_motor * gear_ratio^2；这里保留 yaml 里写的电机轴 kp，
        // yaml 已经按 GO-M8010-6 的电机轴空间标定（~0.8 量程）
        tx[i].kp  = cmd.kp[static_cast<size_t>(i)];        // 直接透传位置环刚度（SDK 期望的是转子域数值）
        tx[i].kd  = cmd.kd[static_cast<size_t>(i)];        // 直接透传速度环阻尼

        // 前馈力矩：算法侧关节力矩 → 电机轴 = joint_tau / gear_ratio
        tx[i].tau = dir * cmd.tau[static_cast<size_t>(i)] / gear_ratio_; // 将算法前馈的关节力矩缩小后丢给电机去输出

        rx[i].motorType = cfg_.motor_type;                 // 设定接收期盼类型
    }

    std::lock_guard<std::mutex> lock(bus_mutex_);          // 操作底层通信前加锁
    if (!port_->sendRecv(tx, rx))                          // 发送数据，并从 RS485 接收数据回放
    {
        SetError("WriteCommand(): sendRecv failed");       // 下发失败记录错误
        armed_.store(false); // 通信断了，立即解使能       // 关键安全机制：一旦通信丢包/断开，软件状态立刻强制降级为离线保护
        return false;
    }
    return true;
}

// 获取错误文本方法
const char *UnitreeMotorAdapter::LastError() const 
{
    std::lock_guard<std::mutex> lock(last_error_mutex_); // 给错误字符串读取加锁，防止一边写一边读发生数据竞争
    return last_error_.empty() ? nullptr : last_error_.c_str(); // 如果为空返回 nullptr，否则返回 C 格式字符串指针
}

// 私有方法：设置内部错误状态
void UnitreeMotorAdapter::SetError(const std::string &msg) 
{
    std::lock_guard<std::mutex> lock(last_error_mutex_); // 写入时加锁保护
    last_error_ = msg;                                   // 保存错误信息
}

// 构建纯刹车数据包的辅助方法
void UnitreeMotorAdapter::BuildBrakePacket(std::vector<MotorCmd> *out) const 
{
    out->assign(static_cast<size_t>(cfg_.num_motors), MotorCmd{}); // 确保输出数组拥有对应数量被清零的电机命令结构
    for (int i = 0; i < cfg_.num_motors; ++i)              // 遍历各电机填装包内容
    {
        auto &m = (*out)[static_cast<size_t>(i)];          // 取出引用
        m.motorType = cfg_.motor_type;                     // 填入型号
        m.id        = static_cast<unsigned short>(i);      // 填入 ID
        m.mode      = static_cast<unsigned short>(mode_brake_); // ★ 最关键：填入 SDK 的 BRAKE 模式控制字
        m.q = 0.0f; m.dq = 0.0f; m.kp = 0.0f; m.kd = 0.0f; m.tau = 0.0f; // 其他全部置零，断开能量输出
    }
}
