#include "rl_real_my_robot.hpp"

#include <chrono>
#include <cmath>
#include <limits>
#include <utility>
#include <iostream>

namespace
{

// 电机底层控制模式：0代表刹车模式/纯阻尼，1代表FOC力矩控制模式
constexpr int kMotorCommandModeBrake = 0;
constexpr int kMotorCommandModeFoc = 1;


// 辅助工具函数：安全裁剪或填充 float 数组到指定维度（防越界）
std::vector<float> Sized(const std::vector<float> &input, size_t n, float fallback)
{
    if (input.empty())
    {
        return std::vector<float>(n, fallback);
    }
    std::vector<float> out(n, fallback);
    for (size_t i = 0; i < n; ++i)
    {
        out[i] = input[std::min(i, input.size() - 1)];
    }
    return out;
}


// 辅助工具函数：安全裁剪或填充 int 数组到指定维度（防越界）
std::vector<int> SizedInt(const std::vector<int> &input, size_t n)
{
    std::vector<int> out(n, 0);
    for (size_t i = 0; i < n; ++i)
    {
        out[i] = input.empty() ? static_cast<int>(i) : input[std::min(i, input.size() - 1)];
    }
    return out;
}


// 辅助工具函数：计算底层硬件命令映射表的最大绝对边界大小
int HardwareCommandSize(const std::vector<int> &mapping, int dofs)
{
    int size = dofs;
    for (int index : mapping)
    {
        size = std::max(size, index + 1);
    }
    return size;
}


// 辅助工具函数：验证一个向量中的所有元素是否均为有限数（防 NaN 导致砸狗）
bool IsFiniteVector(const std::vector<float> &values)
{
    for (float value : values)
    {
        if (!std::isfinite(value))
        {
            return false;
        }
    }
    return true;
}


// 辅助工具函数：获取电机适配器最后的具体错误字符串
std::string AdapterError(const MotorAdapterInterface *adapter)
{
    if (!adapter)
    {
        return "";
    }
    const char *error = adapter->LastError();
    return error ? std::string(error) : std::string();
}


} // namespace


// =================================================================
// 1. 构造与析构函数模块
// =================================================================

// 构造函数1：接收外部注入的裸指针形式适配器（不管理其生命周期）
RL_Real::RL_Real(MotorAdapterInterface *adapter)
    : raw_adapter_(adapter)
{
    robot_name = "my_robot";
    config_name = "himloco";
    ang_vel_axis = "body"; // 设定角速度采用机体坐标系（满足真机和Mujoco需求）

    // 参数初始化预热
    params.Clear();
    ReadYaml(robot_name + "/" + config_name, "config.yaml");

    // 从读取到的参数中获取关节数量（默认12个自由度），并为基类的各项状态数组 resize 开辟内存
    InitJointNum(static_cast<size_t>(params.Get<int>("num_of_dofs", 12)));
    InitObservations();
    InitOutputs();
    InitControl();

    // 状态机工厂模式：根据机器人名字动态实例化对应的有限状态机
    if (FSMManager::GetInstance().IsTypeSupported(robot_name))
    {   
        // 动态创建状态机
        auto fsm_ptr = FSMManager::GetInstance().CreateFSM(robot_name, this);
        if (fsm_ptr)
        {
            fsm = *fsm_ptr;     // 成功后将状态机实体挂载到本类中
        }
    }
    else
    {
        // 如果没找到对应的状态机，抛出严重错误中断程序，防止裸跑机器狗导致危险
        throw std::runtime_error("FSM type not registered: " + robot_name);
    }

    // 将初始状态设置为配置中的默认关节站立零位，防初始化突变
    robot_state.motor_state.q = params.Get<std::vector<float>>("default_dof_pos");
    now_state = robot_state;   // 同步当前状态
    start_state = robot_state; // 同步起始插值状态
}


// 构造函数2：接收智能指针形式的适配器注入
RL_Real::RL_Real(std::shared_ptr<MotorAdapterInterface> adapter)
    : RL_Real()
{
    // 将智能指针所有权转移并挂载到系统后端
    AttachMotorAdapter(std::move(adapter));
}


// 析构函数：退出时执行安全下电注销
RL_Real::~RL_Real()
{
    // 对象生命周期结束时，强制终止所有工作线程并执行电机安全断电
    Stop();
}


// 电机适配器后台挂载与热插拔管理模块
void RL_Real::AttachMotorAdapter(MotorAdapterInterface *adapter)
{
    std::lock_guard<std::mutex> runner_lock(runner_mutex_);   // 锁住运行同步锁
    std::lock_guard<std::mutex> adapter_lock(adapter_mutex_); // 锁住硬件适配器锁

    // 防御性安全检查：如果系统线程已经在高频跑控制了，绝对拒绝中途挂载一个裸指针适配器，防止多线程悬空指针引发灾难
    if (running_ && adapter)
    {
        raw_adapter_ = nullptr;
        adapter_owner_.reset();
        adapter_armed_ = false;
        adapter_fault_latched_ = true;      // 强行触发自锁
        adapter_fault_reason_ = "refused non-owning motor adapter attach while runner is running";
        std::cout << LOGGER::ERROR << adapter_fault_reason_ << std::endl;
        return;
    }
    // 如果之前有智能指针形式的适配器正在运行，先安全将其断电停下
    if (adapter_owner_)
    {
        SafeStopAdapterLocked("adapter detach/replace");
    }
    raw_adapter_ = adapter;     // 挂载新的裸指针
    adapter_owner_.reset();     // 释放原有的智能指针
    adapter_armed_ = false;     // 重置使能状态
    adapter_fault_latched_ = false; // 清除历史锁存故障
    adapter_fault_reason_.clear();
}


// 动态挂载/替换智能指针形式的适配器
void RL_Real::AttachMotorAdapter(std::shared_ptr<MotorAdapterInterface> adapter)
{
    std::lock_guard<std::mutex> runner_lock(runner_mutex_);
    std::lock_guard<std::mutex> adapter_lock(adapter_mutex_);
    if (adapter_owner_)
    {
        SafeStopAdapterLocked("adapter detach/replace");
    }
    raw_adapter_ = nullptr;
    adapter_owner_ = std::move(adapter);
    adapter_armed_ = false;
    adapter_fault_latched_ = false;
    adapter_fault_reason_.clear();
    if (adapter_owner_)
    {
        SafeStopAdapterLocked("adapter attached");
    }
}


// 对 485 总线上的电机进行功率使能（安全验证上电）
bool RL_Real::ArmMotorAdapter()
{
    std::lock_guard<std::mutex> runner_lock(runner_mutex_);
    std::lock_guard<std::mutex> adapter_lock(adapter_mutex_);
    MotorAdapterInterface *adapter = AdapterLocked();                       // 互斥锁保护下获取适配器实体
    if (!adapter)   
    {
        // 无硬件适配器，使能失败
        return false;           
    }

    adapter_armed_ = false;                     // 预设为未使能

    // 检查串口驱动是否开好、总线是否通畅
    if (!adapter->IsReady())
    {
        LatchAdapterFaultLocked("adapter arm failed: adapter not ready");
        return false;
    }

    const int dofs = params.Get<int>("num_of_dofs", 12);
    const std::vector<int> mapping = SizedInt(params.Get<std::vector<int>>("joint_mapping"), dofs);
    MotorAdapterState state;

    // 预读一次电机的初始数据
    if (!adapter->ReadState(&state))
    {
        LatchAdapterFaultLocked("adapter arm failed: read state failed " + AdapterError(adapter));
        return false;
    }

    // 关键：上电前必须对初始的 IMU 和关节数据做一次彻底的合法性过滤（防错错交织引发暴冲）
    std::string error;

    // 极其关键的安全闭环：上电发力矩前，必须对读回来的初速度、角度、四元数进行深度合法性校验
    // 防止串口断线读回一堆垃圾数据 (如 NaN)，一旦带着垃圾数据使能电机，机器人上电瞬间就会由于巨大的 PD 误差发生暴冲伤人
    if (!ValidateAdapterState(state, dofs, mapping, &error))
    {
        LatchAdapterFaultLocked("adapter arm failed: " + error);
        return false;
    }

    // 调用底层的 485 SDK 发送使能广播包
    if (!adapter->Arm())
    {
        LatchAdapterFaultLocked("adapter arm failed: " + AdapterError(adapter));
        return false;
    }

    // 确认底层电机驱动板是否真的成功进入了使能模式
    if (!adapter->IsArmed())
    {
        LatchAdapterFaultLocked("adapter arm failed: adapter did not report armed");
        return false;
    }

    // 通过全部校验，正式解除系统自锁，打开总线功率开关
    adapter_fault_latched_ = false;
    adapter_fault_reason_.clear();
    adapter_armed_ = true;
    std::cout << LOGGER::INFO << "Motor adapter armed after fresh state validation" << std::endl;
    return true;
}


// 主动实施去使能（电机断电）
void RL_Real::DisarmMotorAdapter()
{
    std::lock_guard<std::mutex> runner_lock(runner_mutex_);
    std::lock_guard<std::mutex> adapter_lock(adapter_mutex_);
    SafeStopAdapterLocked("explicit adapter disarm");
}


// 外部查询：当前电机是否处于安全的功率输出使能状态
bool RL_Real::IsMotorAdapterArmed() const
{
    std::lock_guard<std::mutex> adapter_lock(adapter_mutex_);
    return AdapterLocked() && adapter_armed_ && !adapter_fault_latched_;
}


// 外部查询：当前总线是否处于异常自锁挂起状态
bool RL_Real::AdapterFaultLatched() const
{
    std::lock_guard<std::mutex> adapter_lock(adapter_mutex_);
    return adapter_fault_latched_;
}


// 外部查询：触发异常自锁的具体错误文字原因
std::string RL_Real::AdapterFaultReason() const
{
    std::lock_guard<std::mutex> adapter_lock(adapter_mutex_);
    return adapter_fault_reason_;
}

// =================================================================
// 3. 多线程生命周期流控管理模块（含新增的 Xbox 手柄接入）
// =================================================================
// 启动整个控制节点的四大异步后台线程
void RL_Real::Start()
{
    // C++11 原子操作比较并交换 (CAS)：确保多线程并发或者重复调用 Start() 时，只有第一个能通过，
    // 完美防止重复创建 std::thread 导致句柄覆盖重写而引发系统内核强制终止 (std::terminate) 的惨剧。
    bool expected = false;
    if (!running_.compare_exchange_strong(expected, true))
    {
        return;         // 若系统已经在跑，直接拦截退出
    }
    {
        std::lock_guard<std::mutex> runner_lock(runner_mutex_);
        std::lock_guard<std::mutex> adapter_lock(adapter_mutex_);

        // 检查是否使用了不持有的裸指针。为了线程绝对安全，启动并发前必须强制断开不安全的裸指针适配器
        if (HasNonOwningAdapterLocked())
        {
            ClearNonOwningAdapterForThreadedRunLocked(
                "non-owning motor adapter detached before threaded runner start; attach a std::shared_ptr adapter for hardware mode");
        }
        SafeStopAdapterLocked("runner start disarmed");                 // 确保上电时处于无功率安全初始态
    }

    // 模块化手柄嵌入：以非阻塞（O_NONBLOCK）模式尝试开启 Linux 系统第一个原生手柄物理设备文件
    joystick_fd_ = open("/dev/input/js0", O_RDONLY | O_NONBLOCK);
    if (joystick_fd_ < 0)
    {
        // 如果没插手柄，打印警告，但不崩溃，系统宽容允许开发者退回本地纯键盘调试模式
        std::cout << LOGGER::WARNING << "Xbox controller [ /dev/input/js0 ] not found. Gamepad control disabled." << std::endl;
    }
    else
    {
        std::cout << LOGGER::INFO << "Xbox controller connected successfully." << std::endl;
    }

    // 正式向 Linux 内核申请并派生四个独立的并发工作线程，它们将并列在后台永真循环运行
    keyboard_thread_ = std::thread(&RL_Real::KeyboardLoop, this);       // 线程1：低频终端键盘捕捉
    control_thread_  = std::thread(&RL_Real::ControlLoop, this);        // 线程2：超高频 485 电机收发与状态机同步
    model_thread_    = std::thread(&RL_Real::ModelLoop, this);          // 线程3：中频神经网络在线策略推理
    joystick_thread_ = std::thread(&RL_Real::JoystickLoop, this);       // 线程4：新增的手柄异步高频事件流监听冲刷

    std::cout << LOGGER::INFO << "RL real runner started" << std::endl;
}


// 停止控制节点，同步安全回收所有线程资源
void RL_Real::Stop()
{
    // 将原子变量 running_ 设为 false，这会让四个后台线程的 while(running_) 循环在下一帧全部自然结束
    const bool was_running = running_.exchange(false);
    
    // 同步等待并回收各个工作线程，确保所有子线程完全退出后，主线程才允许往下走，杜绝僵尸进程
    if (keyboard_thread_.joinable()) keyboard_thread_.join();
    if (control_thread_.joinable()) control_thread_.join();
    if (model_thread_.joinable()) model_thread_.join();
    if (joystick_thread_.joinable()) joystick_thread_.join();       // 同步回收手柄线程

    // 模块化手柄嵌入：回收完毕后，安全关闭手柄物理设备文件描述符，释放系统内核文件句柄
    if (joystick_fd_ >= 0)
    {
        close(joystick_fd_);
        joystick_fd_ = -1;
    }

    {
        std::lock_guard<std::mutex> adapter_lock(adapter_mutex_);
        if (adapter_owner_)
        {
            SafeStopAdapterLocked("runner stop");               // 终结底层的 485 串口驱动
        }
        else
        {
            adapter_armed_ = false;
            raw_adapter_ = nullptr;
        }
    }
    if (was_running)
    {
        std::cout << LOGGER::INFO << "RL real runner stopped" << std::endl;
    }
}

// =================================================================
// 4. 线程工作体内部独立微循环实现（Worker Loops）
// =================================================================
// 线程1工作体：终端键盘按键捕获微循环
void RL_Real::KeyboardLoop()
{
    while (running_)
    {
        {
            std::lock_guard<std::mutex> runner_lock(runner_mutex_); // 锁住运行同步锁
            KeyboardInterface(); // 调用键盘拦截函数
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50)); // ~20Hz 刷新率，低频对系统几乎零开销
    }
}


// 线程4工作体：新增的 Xbox 手柄专属事件高速事件同步循环
void RL_Real::JoystickLoop()
{
    if (joystick_fd_ < 0) return; // 若开机没连手柄硬件，此后台线程直接挂起，绝不空耗主板 CPU 算力

    while (running_)
    {
        {
            std::lock_guard<std::mutex> runner_lock(runner_mutex_); // 临界区加锁，防止与模型推理线程冲突
            JoystickInterface(); // 高效解析当前输入缓冲区里堆积的所有手柄模拟量和按键事件
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20)); // ~50Hz 高采样率，提供丝滑无延迟的手柄摇杆响应
    }
}


// 线程2工作体：底层极高频 485 控制和有限状态机（FSM）运转微循环
void RL_Real::ControlLoop()
{
    // 从参数字典中提取控制周期 dt (通常为 0.005 秒，即 200Hz)，设置防错底线不低于 1000Hz (0.001)
    const float dt = std::max(params.Get<float>("dt", 0.005f), 0.001f);
    const auto period = std::chrono::microseconds(static_cast<int>(dt * 1000000.0f)); // 换算为微秒

    while (running_)
    {
        const auto start = std::chrono::steady_clock::now(); // 记录本帧循环起始绝对时间点
        try
        {
            RobotControl(); // 执行单步控制解算：通过 485 总线轮询刷新收发电机状态、推进状态机逻辑
        }
        catch (const std::exception &e)
        {
            // 防御性异常捕获：如果控制过程中发生任何严重错误（比如总线爆满或硬件指针越界异常）
            {
                std::lock_guard<std::mutex> runner_lock(runner_mutex_);
                EnterPassiveSafeStateLocked(); // 强迫机器人模拟按下 P 键，迅速切入零力矩纯阻尼被动悬挂安全态
            }
            LatchAdapterFault(std::string("control loop exception: ") + e.what()); // 将错误文本锁存到自锁池
        }
        catch (...)
        {
            {
                std::lock_guard<std::mutex> runner_lock(runner_mutex_);
                EnterPassiveSafeStateLocked();
            }
            LatchAdapterFault("control loop unknown exception"); // 拦截捕获所有未知异常，兜底防疯狗
        }
        
        // 软件定时器机制：计算这一帧解算串口花掉了多少时间，如果比设定的 period 短，则让线程精准睡眠补齐差额，
        // 这能彻底消除 Linux 调度抖动，强行保证 485 串口收发频率绝对稳定在设定的周期（如 200Hz）。
        const auto elapsed = std::chrono::steady_clock::now() - start;
        if (elapsed < period)
        {
            std::this_thread::sleep_for(period - elapsed);
        }
    }
}


// 线程3工作体：中频神经网络在线策略推理微循环
void RL_Real::ModelLoop()
{
    const float dt = std::max(params.Get<float>("dt", 0.005f), 0.001f);
    const int decimation = std::max(params.Get<int>("decimation", 4), 1); // 提取步长抽取系数 (通常为 4)
    const auto period = std::chrono::microseconds(static_cast<int>(dt * decimation * 1000000.0f)); // 换算得出推理周期通常为 50Hz (20ms)
    while (running_)
    {
        const auto start = std::chrono::steady_clock::now();
        try
        {
            RunModel(); // 执行单步模型推理：吃进传感器 Observation 组装，调用网络 Forward，推送动作队列
        }
        catch (const std::exception &e)
        {
            {
                std::lock_guard<std::mutex> runner_lock(runner_mutex_);
                EnterPassiveSafeStateLocked();
            }
            LatchAdapterFault(std::string("model loop exception: ") + e.what());
        }
        catch (...)
        {
            {
                std::lock_guard<std::mutex> runner_lock(runner_mutex_);
                EnterPassiveSafeStateLocked();
            }
            LatchAdapterFault("model loop unknown exception");
        }
        const auto elapsed = std::chrono::steady_clock::now() - start;
        if (elapsed < period)
        {
            std::this_thread::sleep_for(period - elapsed);
        }
    }
}


// =================================================================
// 5. 核心控制解算与手柄映射映射模块
// =================================================================

// 新增模块：Linux 标准 Xbox 事件转换驱动内核
void RL_Real::JoystickInterface()
{
    if (joystick_fd_ < 0) return;

    js_event event;
    // 由于手柄是异步触发的，我们必须用一个 while 循环把操作系统手柄 FIFO 缓冲区里积累的所有积压事件全部排空、冲刷干净
    while (read(joystick_fd_, &event, sizeof(event)) > 0)
    {
        if (event.type & JS_EVENT_INIT) continue; // 强行剔除 Linux 初始化握手阶段发出来的多余扰动电平事件

        // --- 1. 处理物理摇杆事件 (Xbox Axis 模拟量控制移动速度) ---
        if (event.type == JS_EVENT_AXIS)
        {
            // Linux 驱动下，摇杆推满物理极值的输出范围是 -32767 到 32767
            // 我们通过取负号并将除以 32767，将其优雅地归一化到标准浮点数 [-1.0 到 1.0] 之间。
            float normalized_val = -static_cast<float>(event.value) / 32767.0f;
            
            // 8% 的物理死区滤波（Deadzone Control）。解决摇杆常年磨损导致的中央微幅漂移引起的机器人“自走抽风”现象
            if (std::fabs(normalized_val) < 0.08f) normalized_val = 0.0f;

            // 最大移动幅度系数（可根据神经网络训练时限制的 Action Space 幅度进行匹配缩放）
            constexpr float kMaxSpeedX = 1.0f;   // 前进最大期望速度 (m/s)
            constexpr float kMaxSpeedY = 1.0f;   // 侧移最大期望速度 (m/s)
            constexpr float kMaxSpeedYaw = 0.8f; // 绕Z轴原地偏航左转/右转最大期望角速度 (rad/s)

            switch (event.number)
            {
                case 1: // 左摇杆的前后推移 (Xbox Axis 1)：推上去原厂输出是负值，由于上面取了负号，这里正好正数对应前进速度
                    this->control.x = normalized_val * kMaxSpeedX;
                    break;
                case 0: // 左摇杆的左右推移 (Xbox Axis 0)：往左推是负值。在机器人标准右手法则中，向左侧移为正 Y
                    this->control.y = normalized_val * kMaxSpeedY;
                    break;
                case 3: // 右摇杆的左右推移 (Xbox Axis 3)：用于独立控制机器人的偏航角速度
                    this->control.yaw = normalized_val * kMaxSpeedYaw;
                    break;
                default:
                    break;
            }
        }

        // --- 2. 处理物理按键事件 (Xbox Button 状态机状态抢占切换) ---
        // 这里的各个分配按键，和你在 FSM 状态机 CheckChange() 里写的 Input::Gamepad 枚举按键完全一一对应
        if (event.type == JS_EVENT_BUTTON)
        {
            bool is_pressed = (event.value == 1); // 1 代表按下，0 代表松开
            if (is_pressed)
            {
                switch (event.number)
                {
                    case 0: // A 键 -> 触发 FSM 状态转移：命令机器人从地平躺状态【平滑起身站立】(GetUp)
                        this->control.current_gamepad = Input::Gamepad::A;
                        break;
                    case 1: // B 键 -> 触发 FSM 状态转移：命令站立状态下的机器人平缓【贴地趴下】(GetDown)
                        this->control.current_gamepad = Input::Gamepad::B;
                        break;
                    case 4: // LB 键 -> 触发紧急保护：无论处于什么状态，立刻强切到【安全断力阻尼软趴状态】(Passive Mode)
                        this->control.current_gamepad = Input::Gamepad::LB_X;
                        std::cout << std::endl << LOGGER::WARNING << "Emergency STOP triggered by Xbox LB button!" << std::endl;
                        break;
                    case 5: // RB 键 -> 站立完成后，按 RB 键正式激活推理环，将控制权移交给强化学习网络进行 Locomotion 行走
                        this->control.current_gamepad = Input::Gamepad::RB_DPadUp;
                        break;
                    default:
                        break;
                }
            }
            else
            {
                this->control.ClearGamepadInput(); // 按键弹起释放瞬间，立刻清除临时状态，防止逻辑在下一帧被无限重入触发
            }
        }
    }
}


/// 供核心控制环调用的顶层控制同步函数
void RL_Real::RobotControl()
{
    std::lock_guard<std::mutex> runner_lock(runner_mutex_); // 临界区上锁
    
    // 调用 485 电机数据刷新。如果串口通信丢包或者读取报错，立刻强制切断电机发力，退出防错
    if (!RefreshStateForControl(&robot_state))
    {
        EnterPassiveSafeStateLocked();
        control.ClearInput();
        return;
    }
    fresh_state_for_control_ = true; // 状态刷新握手标志置位，表明在这一控制帧中，我们拿到了新鲜且通过校验的硬件反馈
    try
    {
        StateController(&robot_state, &robot_command);  // 驱动业务层有限状态机
        SetCommand(&robot_command);                     // 将状态机输出的 PD 指令打成 485 数据包丢向总线
    }
    catch (...)
    {
        fresh_state_for_control_ = false; // 异常时握手复位
        throw; // 重新向外层控制线程抛出异常，触发外层的 catch 断力机制
    }
    fresh_state_for_control_ = false; // 消费结束，握手复位，等待下一帧串口通知
    control.ClearInput(); // 冲刷单帧瞬时按键数据
}


// 供模型推理环调用的策略正向解算函数
void RL_Real::RunModel()
{
    std::lock_guard<std::mutex> runner_lock(runner_mutex_);
    if (!rl_init_done)
    {
        return; // 若状态机尚未切入 RL 运动状态，不运行推理
    }

    ++episode_length_buf; // 记录策略网络生存步数
    
    // 同步物理控制环刚写进来的、来自 485 硬件总线的最新 IMU 惯导和关节角度数据
    obs.ang_vel = robot_state.imu.gyroscope;
    obs.commands = {control.x, control.y, control.yaw}; // 吃进由手柄线程或键盘线程刚刚异步改写好的期望速度指令
    obs.base_quat = robot_state.imu.quaternion;
    obs.dof_pos = robot_state.motor_state.q;
    obs.dof_vel = robot_state.motor_state.dq;

    obs.actions = Forward(); // 调用策略神经网络前向计算，推算 Actions
    
    // 根据阻抗控制范式，将网络 Actions 转换为各关节的期望位置、期望速度和前馈力矩
    ComputeOutput(obs.actions, output_dof_pos, output_dof_vel, output_dof_tau);
    
    // 推送入进程内线程安全同步双端队列，供高频控制环随时提取，无缝下发
    output_dof_pos_queue.push(output_dof_pos);
    output_dof_vel_queue.push(output_dof_vel);
    output_dof_tau_queue.push(output_dof_tau);

    // 强化学习部署双重软件防火墙
    TorqueProtect(pre_clamp_output_dof_tau);               // 1. 力矩越界过载熔断诊断
    AttitudeProtect(robot_state.imu.quaternion, 75.0f, 75.0f); // 2. 姿态横滚/俯仰倾角过大（如摔倒）物理熔断断电保护
}


// =================================================================
// 6. 底层数据验证与 485 总线输入输出解算模块
// =================================================================

void RL_Real::GetState(RobotState<float> *state)
{
    (void)RefreshStateForControl(state);            // 封装重载基类虚函数
}


// 从 485 适配器中线程安全地捕获当前的 IMU 和关节传感器读数并执行关节映射
bool RL_Real::RefreshStateForControl(RobotState<float> *state)
{
    if (!state)
    {
        return false;
    }

    const int dofs = params.Get<int>("num_of_dofs", 12);
    const bool state_was_empty = state->motor_state.q.empty();

    if (state->motor_state.q.size() != static_cast<size_t>(dofs))
    {
        // 数组预分配
        state->motor_state.resize(static_cast<size_t>(dofs));
    }

    // 底层 485 适配器专用的裸状态包
    MotorAdapterState adapter_state;
    {
        std::lock_guard<std::mutex> adapter_lock(adapter_mutex_);           // 锁定 485 硬件适配器操作锁
        MotorAdapterInterface *adapter = AdapterLocked();
        if (!adapter)
        {
            // 如果处于无硬件适配器仿真调试状态，用默认数据温和填充，防止空指针崩溃
            if (state_was_empty)
            {
                state->motor_state.q = params.Get<std::vector<float>>("default_dof_pos");
                state->motor_state.dq.assign(static_cast<size_t>(dofs), 0.0f);
                state->motor_state.tau_est.assign(static_cast<size_t>(dofs), 0.0f);
            }
            return true;
        }
        if (adapter_fault_latched_)
        {
            return false;               // 如果系统已自锁，拒绝执行读取
        }
        if (!adapter->IsReady())
        {
            LatchAdapterFaultLocked("adapter read failed: adapter not ready");
            return false;
        }

        // 调用 USB转485 驱动 SDK，执行一次物理串口数据收取与解包
        if (!adapter->ReadState(&adapter_state))
        {
            LatchAdapterFaultLocked("adapter read failed: " + AdapterError(adapter));
            return false;
        }

        // 获取参数里的关节映射数组 (例如：[3, 4, 5, 0, 1, 2...])
        const std::vector<int> mapping = SizedInt(params.Get<std::vector<int>>("joint_mapping"), dofs);
        std::string error;

        // 深入检验收到的 485 报文是否合法
        if (!ValidateAdapterState(adapter_state, dofs, mapping, &error))
        {
            LatchAdapterFaultLocked("adapter state invalid: " + error);
            return false;
        }

        // 提取真机 IMU 物理数据
        state->imu.quaternion = adapter_state.base_quaternion;
        state->imu.gyroscope = adapter_state.gyro;


        // 物理世界的电线接线顺序（如左前腿ID为0、1、2）通常和强化学习在 Isaac Gym 仿真训练时的顺序截然不同
        // 通过这个循环，利用 mapping 映射表，将底层 485 乱序的电机通道 ID 里的物理读数，优雅重排后精准塞入算法标准的对应关节项
        for (int i = 0; i < dofs; ++i)
        {
            const int hw = mapping[static_cast<size_t>(i)];                                         // 查表获取这个算法自由度对应的真机电机 ID
            state->motor_state.q[i] = adapter_state.dof_pos[static_cast<size_t>(hw)];               // 映射实际角度
            state->motor_state.dq[i] = adapter_state.dof_vel[static_cast<size_t>(hw)];              // 映射实际速度
            state->motor_state.tau_est[i] = adapter_state.dof_tau_est[static_cast<size_t>(hw)];     // 映射实际估计力矩
        }
        return true;
    }
    return true;
}


// 防御性校验：验证从 485 总线上抓取的原始传感器物理信号是否合法
bool RL_Real::ValidateAdapterState(const MotorAdapterState &state,
                                   int dofs,
                                   const std::vector<int> &mapping,
                                   std::string *error) const
{
    if (mapping.size() != static_cast<size_t>(dofs))
    {
        if (error) *error = "joint_mapping size mismatch";
        return false;
    }
    for (int index : mapping)
    {
        if (index < 0)
        {
            if (error) *error = "joint_mapping contains negative index";
            return false;
        }
    }

    const int hw_size = HardwareCommandSize(mapping, dofs);
    if (state.base_quaternion.size() != 4 || !IsFiniteVector(state.base_quaternion))
    {
        if (error) *error = "base_quaternion must contain 4 finite values";
        return false;
    }
    
    // 几何学防错：验证四元数的模长。如果模长为0，说明 IMU 反馈彻底损坏，若不拦截，后续的几何转换会除以 0 引发数学爆炸
    const float quat_norm = std::sqrt(state.base_quaternion[0] * state.base_quaternion[0] +
                                      state.base_quaternion[1] * state.base_quaternion[1] +
                                      state.base_quaternion[2] * state.base_quaternion[2] +
                                      state.base_quaternion[3] * state.base_quaternion[3]);
    if (quat_norm < 1e-6f)
    {
        if (error) *error = "base_quaternion norm is zero";
        return false;
    }
    if (state.gyro.size() != 3 || !IsFiniteVector(state.gyro))
    {
        if (error) *error = "gyro must contain 3 finite values";
        return false;
    }
    if (state.dof_pos.size() < static_cast<size_t>(hw_size) ||
        state.dof_vel.size() < static_cast<size_t>(hw_size) ||
        state.dof_tau_est.size() < static_cast<size_t>(hw_size))
    {
        if (error) *error = "motor state vectors do not cover all mapped joints";
        return false;
    }
    
    // 如果物理总线受到周围强电磁环境干扰产生严重乱码错位，导致读取的角度出现非有限数 (NaN)，立刻拉起防线拦截自锁
    for (int index : mapping)
    {
        const size_t hw = static_cast<size_t>(index);
        if (!std::isfinite(state.dof_pos[hw]) ||
            !std::isfinite(state.dof_vel[hw]) ||
            !std::isfinite(state.dof_tau_est[hw]))
        {
            if (error) *error = "motor state contains non-finite mapped joint value";
            return false;
        }
    }
    return true; // 完全安全
}


// 算法发送端控制命令合法性检验
bool RL_Real::ValidateCommand(const RobotCommand<float> *command, int dofs, std::string *error) const
{
    if (!command)
    {
        if (error) *error = "null robot command";
        return false;
    }
    const auto &motor = command->motor_command;
    const size_t expected = static_cast<size_t>(dofs);
    if (motor.q.size() != expected || motor.dq.size() != expected ||
        motor.kp.size() != expected || motor.kd.size() != expected ||
        motor.tau.size() != expected)
    {
        if (error) *error = "robot command vector size mismatch";
        return false;
    }
    
    // 极其重要：验证算法发出的指令是否包含 NaN。如果神经网络权重矩阵由于某种极端情况计算爆炸，
    // 输出的指令会变成 NaN，若不加拦截送给电机，电机会当场以最极限功率疯转自毁。
    if (!IsFiniteVector(motor.q) || !IsFiniteVector(motor.dq) ||
        !IsFiniteVector(motor.kp) || !IsFiniteVector(motor.kd) ||
        !IsFiniteVector(motor.tau))
    {
        if (error) *error = "robot command contains non-finite value";
        return false;
    }
    return true;
}


// 接收算法层下发的标准控制参数，通过映射表翻译并包装，下发给 485 电机总线客户端
// 接收算法层计算出的高层指令，将其反向通过映射表翻译打包，下发给 485 串口总线
void RL_Real::SetCommand(const RobotCommand<float> *command)
{
    const int dofs = params.Get<int>("num_of_dofs", 12);
    std::string error;
    // 下发控制前，执行严格的目标边界检查
    if (!ValidateCommand(command, dofs, &error))
    {
        LatchAdapterFault("command validation failed: " + error);
        return;
    }

    std::lock_guard<std::mutex> adapter_lock(adapter_mutex_); // 锁住硬件锁，独占串口写通道
    MotorAdapterInterface *adapter = AdapterLocked();
    if (!adapter || adapter_fault_latched_)
    {
        return; // 若未连接硬件或已自锁，直接拦截抛弃指令
    }
    if (!fresh_state_for_control_)
    {
        // 关键死锁防护：控制回路中必须先成功“读取”传感器，才允许“写入”控制。绝不允许盲目发指令
        LatchAdapterFaultLocked("blocked control write without fresh validated state");
        return;
    }
    if (!adapter->IsReady())
    {
        LatchAdapterFaultLocked("adapter write failed: adapter not ready");
        return;
    }
    if (!adapter_armed_)
    {
        return; // 未使能，拦截
    }
    if (!adapter->IsArmed())
    {
        LatchAdapterFaultLocked("adapter write failed: adapter disarmed unexpectedly");
        return;
    }

    const std::vector<int> mapping = SizedInt(params.Get<params.Get<std::vector<int>>("joint_mapping"), dofs);
    const int hw_size = HardwareCommandSize(mapping, dofs);
    // 从 YAML 中获取该型号电机的物理额定安全堵转力矩上限（如 23.7 牛米）
    const std::vector<float> torque_limits = Sized(params.Get<std::vector<float>>("torque_limits"), dofs, 23.7f);

    // 组装底层 485 驱动 SDK 要求的专属全通道 MotorAdapterCommand 报文实体
    MotorAdapterCommand out;
    out.mode.assign(static_cast<size_t>(hw_size), kMotorCommandModeBrake); // 全通道预设为安全阻尼刹车态
    out.q.assign(static_cast<size_t>(hw_size), 0.0f);
    out.dq.assign(static_cast<size_t>(hw_size), 0.0f);
    out.kp.assign(static_cast<size_t>(hw_size), 0.0f);
    out.kd.assign(static_cast<size_t>(hw_size), 0.0f);
    out.tau.assign(static_cast<size_t>(hw_size), 0.0f);

    // 反向翻译官逻辑：将强化学习算法标准顺序下的目标 q, dq, kp, kd, tau，
    // 根据映射表，分发重新填入底层硬件对应的物理电机通道 ID 位置中。
    for (int i = 0; i < dofs; ++i)
    {
        const int hw = mapping[static_cast<size_t>(i)]; // 换算得到真机物理电机 ID
        if (hw < 0 || hw >= hw_size)
        {
            LatchAdapterFaultLocked("adapter write failed: invalid joint mapping");
            return;
        }
        out.mode[static_cast<size_t>(hw)] = kMotorCommandModeFoc; // 将该电机的物理通道正式激活切换到 FOC 力矩控制闭环
        out.q[static_cast<size_t>(hw)] = command->motor_command.q[static_cast<size_t>(i)];
        out.dq[static_cast<size_t>(hw)] = command->motor_command.dq[static_cast<size_t>(i)];
        out.kp[static_cast<size_t>(hw)] = command->motor_command.kp[static_cast<size_t>(i)];
        out.kd[static_cast<size_t>(hw)] = command->motor_command.kd[static_cast<size_t>(i)];
        
        // 🛠️ 终极力矩硬限幅（ClampValue）：如果强化学习模型输出的力矩由于震动超出了电机的物理红线，
        // 在这里进行强制截断，限制在安全牛米范围内，保障电机内部齿轮箱绝对不会由于算法超调而崩齿。
        out.tau[static_cast<size_t>(hw)] = ClampValue(command->motor_command.tau[static_cast<size_t>(i)],
                                                      -torque_limits[static_cast<size_t>(i)],
                                                      torque_limits[static_cast<size_t>(i)]);
    }

    // 调用物理适配器，执行 USB转485 物理串口数据发送
    if (!adapter->WriteCommand(out))
    {
        LatchAdapterFaultLocked("adapter write failed: " + AdapterError(adapter));
    }
}


// =================================================================
// 7. 底层同步锁安全自锁及后备应急机制模块
// =================================================================


// 锁定保护状态下安全提取当前的物理适配器后端指针
MotorAdapterInterface *RL_Real::AdapterLocked() const
{
    return adapter_owner_ ? adapter_owner_.get() : raw_adapter_;
}

// 检查当前挂载的是否为非持有的纯裸指针适配器
bool RL_Real::HasNonOwningAdapterLocked() const
{
    return raw_adapter_ != nullptr && !adapter_owner_;
}

// 并发启动前强行卸载裸指针适配器自锁函数
void RL_Real::ClearNonOwningAdapterForThreadedRunLocked(const std::string &reason)
{
    raw_adapter_ = nullptr;
    adapter_armed_ = false;
    adapter_fault_latched_ = true;
    adapter_fault_reason_ = reason;
    fresh_state_for_control_ = false;
    std::cout << LOGGER::ERROR << "Motor adapter fault latched: " << adapter_fault_reason_ << std::endl;
}

// 触发安全保护停机：命令底层总线全断电，并调用 485 脱机抱闸原语
void RL_Real::SafeStopAdapterLocked(const std::string &reason)
{
    adapter_armed_ = false;
    MotorAdapterInterface *adapter = AdapterLocked();
    if (!adapter)
    {
        return;
    }
    adapter->Disarm(); // 断开 485 功率输出
    if (!adapter->SafeStop()) // 调用电机驱动 SDK 的紧急刹车原语
    {
        adapter_fault_latched_ = true;
        if (adapter_fault_reason_.empty())
        {
            adapter_fault_reason_ = reason + ": adapter SafeStop failed " + AdapterError(adapter);
        }
        std::cout << LOGGER::ERROR << adapter_fault_reason_ << std::endl;
    }
}

// 锁存一个致命的通信或数值级硬件故障（带互斥锁内部版本）
void RL_Real::LatchAdapterFaultLocked(const std::string &reason)
{
    adapter_fault_latched_ = true;
    adapter_fault_reason_ = reason;
    SafeStopAdapterLocked(reason); // 锁存异常的刹那间，必须十万火急强制电机断电脱机，避免摔坏机器人
    std::cout << LOGGER::ERROR << "Motor adapter fault latched: " << adapter_fault_reason_ << std::endl;
}

// 锁存一个致命硬件故障（供无锁外部作用域调用的公开版本）
void RL_Real::LatchAdapterFault(const std::string &reason)
{
    std::lock_guard<std::mutex> adapter_lock(adapter_mutex_);
    LatchAdapterFaultLocked(reason);
}

// 应急安全后备机制：控制异常或丢包严重时，强行清空队列并虚拟按下 P 键，令机器人优雅降落平躺
void RL_Real::EnterPassiveSafeStateLocked()
{
    rl_init_done = false;
    fresh_state_for_control_ = false;
    control.SetKeyboard(Input::Keyboard::P); // 强制发送模拟 P 键，通知状态机无缝切回 Passive 纯阻尼平躺模式
    const int dofs = params.Get<int>("num_of_dofs", 12);
    robot_command.motor_command.resize(static_cast<size_t>(dofs));
    
    // 清空双端等待分发执行的所有目标队列缓存，防止残留旧的位置偏差导致二次跳变
    output_dof_pos_queue.clear();
    output_dof_vel_queue.clear();
    output_dof_tau_queue.clear();
}

// 执行强化学习网络策略前向推理的具体内部逻辑
std::vector<float> RL_Real::Forward()
{
    std::lock_guard<std::mutex> lock(model_mutex); // 锁住神经网络推理池

    const std::vector<float> clamped_obs = ComputeObservation(); // 动态拼装并裁剪当前的 Observation 反馈
    const std::vector<int> history_ids = params.Get<std::vector<int>>("observations_history");
    std::vector<float> model_input = clamped_obs;
    
    // 处理时序帧缓存数据拼接堆叠 (Handling POMDP)
    if (!history_ids.empty())
    {
        history_obs_buf.Insert(clamped_obs); // 喂入最新一帧数据
        history_obs = history_obs_buf.Get(history_ids); // 提取提取历史序列
        if (!history_obs.empty())
        {
            model_input = history_obs; // 将历史状态流合并作为神经网络的巨型输入张量
        }
    }

    std::vector<float> actions;
    if (policy_forward_hook)
    {
        actions = policy_forward_hook(model_input); // 若挂载了仿真调试钩子，优先走钩子
    }
    else if (model && model->IsLoaded())
    {
        actions = model->Forward(model_input); // 核心：调用 LibTorch/ONNX 推理引擎，执行矩阵推导返回 Actions
    }
    else
    {
        throw std::runtime_error("policy model is not loaded");
    }

    const int dofs = params.Get<int>("num_of_dofs", 12);
    // 从 YAML 中获取网络动作的最大/最小合理边界（如 [-5.0, 5.0]）
    const std::vector<float> lower = Sized(params.Get<std::vector<float>>("clip_actions_lower"), dofs, -5.0f);
    const std::vector<float> upper = Sized(params.Get<std::vector<float>>("clip_actions_upper"), dofs, 5.0f);
    if (actions.size() != static_cast<size_t>(dofs))
    {
        throw std::runtime_error("policy action size mismatch: expected " +
                                 std::to_string(dofs) + ", got " +
                                 std::to_string(actions.size()));
    }
    return ClampVector(actions, lower, upper); // 将网络直接输出的粗糙 Actions 限制在安全动作限幅内并返回

}  

