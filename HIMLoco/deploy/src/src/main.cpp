// =================================================================
// HIMLoco my_robot 真机入口
//
// 启动顺序：
//   1. 读 robot_config.yaml 拿到串口路径/波特率
//   2. 启动 IMU 串口驱动
//   3. 构造 Unitree GO-M8010-6 USB转485 适配器（IMU 注入进去）
//   4. 构造 RL_Real 并挂载适配器，做一次 Arm() 状态校验
//   5. Start() 起 4 个后台线程
//   6. 等 SIGINT 优雅退出
// =================================================================

#include "rl_real_my_robot.hpp"
#include "unitree_motor_adapter.hpp"
#include "imu_driver.hpp"
#include "logger.hpp"

#include <yaml-cpp/yaml.h>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <thread>


namespace
{
std::atomic<bool> g_stop_requested{false};                      // 定义全局原子布尔变量，默认 false，用于标记是否收到了退出请求
void HandleSigint(int) { g_stop_requested.store(true); }        // 信号处理函数：当收到操作系统信号时，将标志位置为 true


// 定义模板函数 YamlGet，用于安全地读取单个值
template <typename T>
T YamlGet(const YAML::Node &node, const std::string &key, T fallback)
{
    if (!node || !node[key]) return fallback;               // 如果节点为空或键不存在，直接返回传入的默认值 (fallback)  
    try { return node[key].as<T>(); }                       // 尝试将 YAML 节点解析为指定的类型 T
    catch (...) { return fallback; }                        // 如果解析失败（例如类型不匹配），捕获异常并返回默认值
}

// 定义模板函数 YamlGetVec，用于安全地读取数组/列表
template <typename T>
std::vector<T> YamlGetVec(const YAML::Node &node, const std::string &key)
{
    std::vector<T> out;                       // 初始化空数组
    if (!node || !node[key]) return out;      // 如果节点为空或键不存在，返回空数组
    try { return node[key].as<std::vector<T>>(); } // 尝试解析为 std::vector<T>
    catch (...) { return out; }               // 解析失败则返回空数组
}

} // namespace


int main(int argc, char **argv)
{
    // 允许命令行覆盖 yaml 路径，方便不在 deploy 工作目录启动
    std::string cfg_path = (argc > 1) ? argv[1] : (std::string(SIM2REAL_CONFIG_DIR) + "/robot_config.yaml");

    // 声明 YAML 根节点
    YAML::Node cfg;
    try
    {
        // 尝试加载并解析指定路径的 yaml 文件
        cfg = YAML::LoadFile(cfg_path);
    }
    // 如果文件不存在或格式错误，捕获异常
    catch (const std::exception &e)
    {
        std::cerr << LOGGER::ERROR << "Failed to load " << cfg_path << ": " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    // ---------- （1) IMU ----------
    // 串口路径统一从 devices: 块读，imu:/unitree_motor_adapter: 只放协议层参数
    auto devices_node = cfg["devices"];                 // 获取 yaml 中的 devices 节点
    auto imu_node = cfg["imu"];                         // 获取 yaml 中的 imu 节点
    
    // 读取 IMU 的串口路径，默认 /dev/ttyUSB1
    const std::string imu_port = YamlGet<std::string>(devices_node, "imu_serial", "/dev/ttyUSB1");
    // 读取波特率，默认 9600
    const int         imu_baud = YamlGet<int>(imu_node, "baudrate", 9600);
    // 读取健康检查超时时间，默认 50ms
    const int         imu_health_ms = YamlGet<int>(imu_node, "health_timeout_ms", 50);

    // 利用配置参数创建一个 IMUDriver 实例的共享指针
    auto imu = std::make_shared<IMUDriver>(imu_port, imu_baud, imu_health_ms);

    // 调用 Start() 打开串口并开启后台读取线程
    if (!imu->Start())
    {   
        // 打印 IMU 启动失败的具体原因
        std::cerr << LOGGER::ERROR << "IMU start failed: " << imu->LastError() << std::endl;
        // 警告：没有 IMU 无法运行 RL 策略
        std::cerr << LOGGER::WARNING << "Continuing without IMU — robot WILL NOT be able to enter RL state." << std::endl;
        // 故意不退出：让用户能进入 Passive 看电机连通性；进 RL 时上层 ValidateAdapterState 会拦
    }

    else
    {
        // 启动成功，打印绿色的 INFO 日志
        std::cout << LOGGER::INFO << "IMU [" << imu_port << " @ " << imu_baud << "] started." << std::endl;
    }

    // ---------- 2) 电机适配器 ----------
    auto motor_node = cfg["unitree_motor_adapter"];                     // 获取电机配置节点
    UnitreeMotorAdapter::Config mc;                                     // 实例化一个电机配置结构体
    
    mc.serial_port = YamlGet<std::string>(devices_node, "motor_serial", "/dev/ttyUSB0");
    mc.baudrate    = YamlGet<int>(motor_node, "baudrate", 4000000);
    mc.timeout_us  = YamlGet<int>(motor_node, "timeout_us", 300);
    mc.num_motors  = YamlGet<int>(motor_node, "num_motors", 12);
    mc.motor_type  = MotorType::GO_M8010_6;

    // 极性 / 零位（按硬件电机ID 顺序，即 motor_id=0..11 的物理顺序）
    // robot_config.yaml::joint_hardware_params 是按"关节名→电机ID"组织的，
    // 我们把它倒推成"电机ID→极性 / 零位"两个 12 元素数组
    mc.direction.assign(static_cast<size_t>(mc.num_motors), 1.0f);
    mc.zero_offset.assign(static_cast<size_t>(mc.num_motors), 0.0f);
    if (auto hw = cfg["joint_hardware_params"])
    {
        for (const auto &kv : hw)
        {
            const auto &node = kv.second;
            const int idx = YamlGet<int>(node, "motor_idx", -1);
            const float dir = YamlGet<float>(node, "direction", 1.0f);
            if (idx >= 0 && idx < mc.num_motors)
            {
                mc.direction[static_cast<size_t>(idx)] = dir;
            }
        }
    }

    std::shared_ptr<UnitreeMotorAdapter> adapter;
    try
    {
        adapter = std::make_shared<UnitreeMotorAdapter>(mc, imu);
    }
    catch (const std::exception &e)
    {
        std::cerr << LOGGER::ERROR << "MotorAdapter construct failed: " << e.what() << std::endl;
        imu->Stop();
        return EXIT_FAILURE;
    }
    if (!adapter->IsReady())
    {
        std::cerr << LOGGER::ERROR << "MotorAdapter not ready: " << (adapter->LastError() ? adapter->LastError() : "") << std::endl;
        imu->Stop();
        return EXIT_FAILURE;
    }
    std::cout << LOGGER::INFO << "Motor adapter [" << mc.serial_port << " @ " << mc.baudrate << ", "
              << mc.num_motors << " motors] ready." << std::endl;


    // ---------- 3) RL_Real ----------
    std::unique_ptr<RL_Real> real;
    try
    {
        real = std::make_unique<RL_Real>(adapter);
    }
    catch (const std::exception &e)
    {
        std::cerr << LOGGER::ERROR << "RL_Real construct failed: " << e.what() << std::endl;
        adapter->SafeStop();
        imu->Stop();
        return EXIT_FAILURE;
    }

    // 上电前的安全校验：必须读到一帧合法 IMU + 电机状态才允许 Arm
    if (!real->ArmMotorAdapter())
    {
        std::cerr << LOGGER::ERROR << "ArmMotorAdapter failed: " << real->AdapterFaultReason() << std::endl;
        std::cerr << LOGGER::WARNING << "Continuing in unarmed mode for diagnostics." << std::endl;
    }
    else
    {
        std::cout << LOGGER::INFO << "Motor adapter armed." << std::endl;
    }


    // ---------- 4) 安装 SIGINT、起线程、等退出 ----------
    std::signal(SIGINT,  HandleSigint);
    std::signal(SIGTERM, HandleSigint);

    real->Start();
    std::cout << LOGGER::INFO << "RL real runner spawned. Press Ctrl+C to exit." << std::endl;
    std::cout << LOGGER::INFO << "Keyboard:  0=GetUp  1=RL Locomotion  9=GetDown  P=Passive(emergency)" << std::endl;
    std::cout << LOGGER::INFO << "Gamepad :  A=GetUp  RB=RL Locomotion  B=GetDown  LB+X=Passive" << std::endl;

    while (!g_stop_requested.load())
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    std::cout << std::endl << LOGGER::INFO << "Shutdown requested, stopping threads…" << std::endl;
    real->Stop();
    adapter->SafeStop();
    imu->Stop();
    std::cout << LOGGER::INFO << "Bye." << std::endl;
    return EXIT_SUCCESS;
}
