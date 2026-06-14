#include "rl_sdk.hpp"


// 状态控制器，接收当前机器人状态并输出控制指令
void RL::StateController(const RobotState<float>* state, RobotCommand<float>* command)
{
    // 定义一个Lambda函数，用于将最新的state和command指针传递给FSM（有限状态机）状态
    auto updateState = [&](std::shared_ptr<FSMState> statePtr)
    {
        // 尝试将通用FSMState转换为强化学习专用的RLFSMState
        if (auto rl_fsm_state = std::dynamic_pointer_cast<RLFSMState>(statePtr))
        {
            rl_fsm_state->fsm_state = state;                // 更新状态指针
            rl_fsm_state->fsm_command = command;            // 更新指令指针
        }
    };

    // 遍历状态机中的所有状态，并调用上面的Lambda函数更新数据
    for (auto& pair : fsm.states_)
    {
        updateState(pair.second);
    }

    // 运行一次状态机逻辑（执行当前状态的逻辑）
    fsm.Run();

    // 运动时间计数器递增
    this->motiontime++;

    // --- 以下为键盘控制逻辑（映射到X/Y轴平移和Yaw偏航） ---
    if (this->control.current_keyboard == Input::Keyboard::W)
    {
        this->control.x += 0.1f;
    }
    if (this->control.current_keyboard == Input::Keyboard::S)
    {
        this->control.x -= 0.1f;
    }
    if (this->control.current_keyboard == Input::Keyboard::A)
    {
        this->control.y += 0.1f;
    }
    if (this->control.current_keyboard == Input::Keyboard::D)
    {
        this->control.y -= 0.1f;
    }
    if (this->control.current_keyboard == Input::Keyboard::Q)
    {
        this->control.yaw += 0.1f;
    }
    if (this->control.current_keyboard == Input::Keyboard::E)
    {
        this->control.yaw -= 0.1f;
    }
    if (this->control.current_keyboard == Input::Keyboard::Space)
    {
        this->control.x = 0.0f;
        this->control.y = 0.0f;
        this->control.yaw = 0.0f;
    }
}


// 观测值计算
std::vector<float> RL::ComputeObservation()
{
    std::vector<std::vector<float>> obs_list;           // 存储各项观测值的二维数组（稍后会展平）

    // 遍历配置文件中定义需要的所有观测项
    for (const std::string &observation : this->params.Get<std::vector<std::string>>("observations"))
    {
        // ============= 基础观测值 =============
        // 用户指令（如期望的x, y速度和yaw角速度）
        if (observation == "commands")              
        {
            obs_list.push_back(this->obs.commands * this->params.Get<std::vector<float>>("commands_scale"));
        }
        // 角速度
        else if (observation == "ang_vel")
        {
            // In ROS1 Gazebo, the coordinate system for angular velocity is in the world coordinate system.
            // In ROS2 Gazebo, mujoco and real robot, the coordinate system for angular velocity is in the body coordinate system.
            if (this->ang_vel_axis == "body")
            {
                obs_list.push_back(this->obs.ang_vel * this->params.Get<float>("ang_vel_scale"));
            }
            else if (this->ang_vel_axis == "world")
            {
                obs_list.push_back(QuatRotateInverse(this->obs.base_quat, this->obs.ang_vel) * this->params.Get<float>("ang_vel_scale"));
            }
        }
        // 重力分量
        else if (observation == "gravity_vec")
        {
            obs_list.push_back(QuatRotateInverse(this->obs.base_quat, this->obs.gravity_vec));
        }
        // 关节位置（相对于默认位置的偏差）
        else if (observation == "dof_pos")
        {
            std::vector<float> dof_pos_rel = this->obs.dof_pos - this->params.Get<std::vector<float>>("default_dof_pos");
            for (int i : this->params.Get<std::vector<int>>("wheel_indices"))
            {
                dof_pos_rel[i] = 0.0f;
            }
            obs_list.push_back(dof_pos_rel * this->params.Get<float>("dof_pos_scale"));
        }
        // 关节速度
        else if (observation == "dof_vel")
        {
            obs_list.push_back(this->obs.dof_vel * this->params.Get<float>("dof_vel_scale"));
        }
        // 上一次的动作输出
        else if (observation == "actions")
        {
            obs_list.push_back(this->obs.actions);
        }
    }

    // 记录各个观测项的维度大小
    this->obs_dims.clear();
    for (const auto& obs : obs_list)
    {
       this->obs_dims.push_back(obs.size());
    }

    // 将二维数组展平为一维数组 (Flatten)
    std::vector<float> obs;
    for (const auto& obs_vec : obs_list)
    {
        obs.insert(obs.end(), obs_vec.begin(), obs_vec.end());
    }
    // 根据配置项 clip_obs 裁剪观测值，防止异常极大值输入网络导致输出崩溃
    std::vector<float> clamped_obs = clamp(obs, -this->params.Get<float>("clip_obs"), this->params.Get<float>("clip_obs"));
    return clamped_obs;     // 返回最终送入神经网络的张量
}


// 观测初始化
void RL::InitObservations()
{   
    this->obs.commands = {0.0f, 0.0f, 0.0f};
    this->obs.ang_vel = {0.0f, 0.0f, 0.0f};
    this->obs.gravity_vec = {0.0f, 0.0f, -1.0f};
    this->obs.base_quat = {0.0f, 0.0f, 0.0f, 1.0f};

    // 从配置文件中读取机器人的默认关节位置（比如四足机器人站立时的默认关节角度），赋给当前观测位置
    this->obs.dof_pos = this->params.Get<std::vector<float>>("default_dof_pos");
    
    // 清空关节速度数组，并根据配置文件中的关节数量(num_of_dofs)重新分配内存，初始值全部设为0.0f
    this->obs.dof_vel.clear();
    this->obs.dof_vel.resize(this->params.Get<int>("num_of_dofs"), 0.0f);
    
    // 清空上一帧的动作数组，重新分配内存，初始值全设为0.0f
    this->obs.actions.clear();
    this->obs.actions.resize(this->params.Get<int>("num_of_dofs"), 0.0f);
    
    // 强制执行一次观测值计算。这相当于“预热”，确保神经网络在第一次推理前，观测张量是有合法数据的
    this->ComputeObservation();
}


// 输出值初始化
void RL::InitOutputs()
{
    // 获取机器人关节总数
    int num_of_dofs = this->params.Get<int>("num_of_dofs");

    // 清空并重新分配前馈力矩数组，初始值为0
    this->output_dof_tau.clear();
    this->output_dof_tau.resize(num_of_dofs, 0.0f);

    // 将目标期望位置直接设置为机器人的默认姿态，防止启动瞬间去追踪原点(0)导致折叠/摔倒
    this->output_dof_pos = this->params.Get<std::vector<float>>("default_dof_pos");

    // 清空并重新分配目标期望速度数组，初始值为0
    this->output_dof_vel.clear();
    this->output_dof_vel.resize(num_of_dofs, 0.0f);
}


// 用户控制端初始化
void RL::InitControl()
{
    // 将手柄/键盘期望的速度指令全部归零，保证机器人上电/切入RL状态时是静止待命的
    this->control.x = 0.0f;
    this->control.y = 0.0f;
    this->control.yaw = 0.0f;
}


// 关节维度与内存初始化
void RL::InitJointNum(size_t num_joints)
{
    // 为底层硬件状态、起始状态、当前状态预分配内存。
    // 如果不在这里 resize，后续代码中直接使用 state.motor_state[i] 会直接导致越界报错 (Segfault)
    this->robot_state.motor_state.resize(num_joints);
    this->start_state.motor_state.resize(num_joints);
    this->now_state.motor_state.resize(num_joints);

    // 为发给硬件的指令数组预分配内存
    this->robot_command.motor_command.resize(num_joints);
}


// 初始化强化学习模块
void RL::InitRL(std::string robot_config_path)
{
    // 线程锁，保证加载过程安全
    std::lock_guard<std::mutex> lock(this->model_mutex);

    // 加载机器人的YAML配置
    this->ReadYaml(robot_config_path, "config.yaml");

    // 初始化关节数量
    this->InitJointNum(this->params.Get<int>("num_of_dofs"));

    // init rl
    this->InitObservations();
    this->InitOutputs();
    this->InitControl();


    // 如果配置了观测值历史（用于处理POMDP/部分可观测问题，比如盲走）
    const auto& observations_history = this->params.Get<std::vector<int>>("observations_history");  // avoid dangling reference
    if (!observations_history.empty())
    {   
        // 找到最长的历史帧数，初始化History Buffer
        int history_length = *std::max_element(observations_history.begin(), observations_history.end()) + 1;
        this->history_obs_buf = ObservationBuffer(1, this->obs_dims, history_length, this->params.Get<std::string>("observations_history_priority"));
    }

    // 加载基于ONNX/LibTorch/TensorRT推断引擎的模型
    std::string model_path = std::string(POLICY_DIR) + "/" + robot_config_path + "/" + this->params.Get<std::string>("model_name");
    this->model = InferenceRuntime::ModelFactory::load_model(model_path);
    if (!this->model)
    {
        throw std::runtime_error("Failed to load model from: " + model_path);       // 加载失败抛出异常
    }
}


// 将神经网络输出的 Actions 转为实际的控制量
void RL::ComputeOutput(const std::vector<float> &actions, std::vector<float> &output_dof_pos, std::vector<float> &output_dof_vel, std::vector<float> &output_dof_tau)
{
    // 将网络动作乘以缩放系数 (通常为正负1之间的输出映射到实际物理幅度)
    std::vector<float> actions_scaled = actions * this->params.Get<std::vector<float>>("action_scale");

    std::vector<float> pos_actions_scaled = actions_scaled;             // 用于位置控制的动作
    std::vector<float> vel_actions_scaled(actions.size(), 0.0f);        // 用于速度控制的动作

    // 如果存在轮子（比如轮腿机器人），轮子不使用位置控制，而使用速度控制
    for (int i : this->params.Get<std::vector<int>>("wheel_indices"))
    {
        pos_actions_scaled[i] = 0.0f;                           // 轮子位置增量设为0
        vel_actions_scaled[i] = actions_scaled[i];              // 将动作交给速度
    }

    std::vector<float> all_actions_scaled = pos_actions_scaled + vel_actions_scaled;

    // 期望位置 = 缩放动作 + 默认位置
    output_dof_pos = pos_actions_scaled + this->params.Get<std::vector<float>>("default_dof_pos");
    output_dof_vel = vel_actions_scaled;

    // 核心PD控制公式： Tau = Kp * (Target_P - Current_P) - Kd * Current_V
    // Target_P 在这里为: 网络动作值 + 默认姿势
    output_dof_tau = this->params.Get<std::vector<float>>("rl_kp") * (all_actions_scaled + this->params.Get<std::vector<float>>("default_dof_pos") - this->obs.dof_pos) - this->params.Get<std::vector<float>>("rl_kd") * this->obs.dof_vel;
    
    // 对最终力矩进行截断，保护硬件不被烧毁
    output_dof_tau = clamp(output_dof_tau, -this->params.Get<std::vector<float>>("torque_limits"), this->params.Get<std::vector<float>>("torque_limits"));
}


// 逆向查找关节映射表索引
int RL::InverseJointMapping(int idx) const
{
    auto joint_mapping = this->params.Get<std::vector<int>>("joint_mapping");
    for (size_t i = 0; i < joint_mapping.size(); ++i) {
        if (joint_mapping[i] == idx) return (int)i;
    }
    return -1;
}


// 力矩保护检查
void RL::TorqueProtect(const std::vector<float>& origin_output_dof_tau)
{
    std::vector<int> out_of_range_indices;
    std::vector<float> out_of_range_values;

    // 遍历所有力矩，检查是否超出配置中的物理极限
    for (size_t i = 0; i < origin_output_dof_tau.size(); ++i)
    {
        float torque_value = origin_output_dof_tau[i];
        float limit_lower = -this->params.Get<std::vector<float>>("torque_limits")[i];
        float limit_upper = this->params.Get<std::vector<float>>("torque_limits")[i];

        if (torque_value < limit_lower || torque_value > limit_upper)
        {
            out_of_range_indices.push_back(i);
            out_of_range_values.push_back(torque_value);
        }
    }
    if (!out_of_range_indices.empty())
    {
        // 如果超限，打印警告信息
        for (size_t i = 0; i < out_of_range_indices.size(); ++i)
        {
            int index = out_of_range_indices[i];
            float value = out_of_range_values[i];
            float limit_lower = -this->params.Get<std::vector<float>>("torque_limits")[index];
            float limit_upper = this->params.Get<std::vector<float>>("torque_limits")[index];

            std::cout << LOGGER::WARNING << "Torque(" << index + 1 << ")=" << value << " out of range(" << limit_lower << ", " << limit_upper << ")" << std::endl;
        }
        // 触发保护动作，比如切换到趴下（GETDOWN）状态
        std::cout << LOGGER::INFO << "Switching to STATE_POS_GETDOWN"<< std::endl;
    }
}


// 姿态倾倒保护
void RL::AttitudeProtect(const std::vector<float> &quaternion, float pitch_threshold, float roll_threshold)
{
    // 将四元数转为欧拉角，再将弧度转换为角度(*57.2958)
    std::vector<float> euler = QuaternionToEuler(quaternion);
    float roll = euler[0] * 57.2958f;   // Convert to degrees
    float pitch = euler[1] * 57.2958f;

    // 如果横滚角或俯仰角超过阈值（证明机器人可能摔倒了）
    if (std::fabs(roll) > roll_threshold)
    {
        this->control.SetKeyboard(Input::Keyboard::P);          // 发送模拟按键触发保护状态机
        std::cout << LOGGER::WARNING << "Roll exceeds " << roll_threshold << " degrees. Current: " << roll << " degrees." << std::endl;
    }
    if (std::fabs(pitch) > pitch_threshold)
    {
        this->control.SetKeyboard(Input::Keyboard::P);
        std::cout << LOGGER::WARNING << "Pitch exceeds " << pitch_threshold << " degrees. Current: " << pitch << " degrees." << std::endl;
    }
}


// --- Unix终端底层非阻塞键盘读取实现 ---
#include <termios.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <unistd.h>


// 实现类似Windows的kbhit()函数，无需按下回车即可捕捉按键
static int kbhit()
{
    static bool initialized = false;
    static termios original_term;

    // Initialize terminal to non-canonical mode on first call
    if (!initialized)
    {
        tcgetattr(STDIN_FILENO, &original_term);

        termios new_term = original_term;
        new_term.c_lflag &= ~(ICANON | ECHO);  // Disable canonical mode and echo
        new_term.c_cc[VMIN] = 0;   // Non-blocking read
        new_term.c_cc[VTIME] = 0;  // No timeout

        tcsetattr(STDIN_FILENO, TCSANOW, &new_term);

        // Register cleanup function to restore terminal on exit
        static bool cleanup_registered = false;
        if (!cleanup_registered)
        {
            std::atexit([]() {
                tcsetattr(STDIN_FILENO, TCSANOW, &original_term);
            });
            cleanup_registered = true;
        }

        initialized = true;
    }

    // Non-blocking read of a single character
    char c;
    int result = read(STDIN_FILENO, &c, 1);

    return (result == 1) ? (unsigned char)c : -1;
}


// 将ASCII字符映射为内部的 Input::Keyboard 枚举变量
void RL::KeyboardInterface()
{
    int c = kbhit();
    if (c > 0)
    {
        switch (c)
        {
        case '0': this->control.SetKeyboard(Input::Keyboard::Num0); break;
        case '1': this->control.SetKeyboard(Input::Keyboard::Num1); break;
        case '2': this->control.SetKeyboard(Input::Keyboard::Num2); break;
        case '3': this->control.SetKeyboard(Input::Keyboard::Num3); break;
        case '4': this->control.SetKeyboard(Input::Keyboard::Num4); break;
        case '5': this->control.SetKeyboard(Input::Keyboard::Num5); break;
        case '6': this->control.SetKeyboard(Input::Keyboard::Num6); break;
        case '7': this->control.SetKeyboard(Input::Keyboard::Num7); break;
        case '8': this->control.SetKeyboard(Input::Keyboard::Num8); break;
        case '9': this->control.SetKeyboard(Input::Keyboard::Num9); break;
        case 'a': case 'A': this->control.SetKeyboard(Input::Keyboard::A); break;
        case 'b': case 'B': this->control.SetKeyboard(Input::Keyboard::B); break;
        case 'c': case 'C': this->control.SetKeyboard(Input::Keyboard::C); break;
        case 'd': case 'D': this->control.SetKeyboard(Input::Keyboard::D); break;
        case 'e': case 'E': this->control.SetKeyboard(Input::Keyboard::E); break;
        case 'f': case 'F': this->control.SetKeyboard(Input::Keyboard::F); break;
        case 'g': case 'G': this->control.SetKeyboard(Input::Keyboard::G); break;
        case 'h': case 'H': this->control.SetKeyboard(Input::Keyboard::H); break;
        case 'i': case 'I': this->control.SetKeyboard(Input::Keyboard::I); break;
        case 'j': case 'J': this->control.SetKeyboard(Input::Keyboard::J); break;
        case 'k': case 'K': this->control.SetKeyboard(Input::Keyboard::K); break;
        case 'l': case 'L': this->control.SetKeyboard(Input::Keyboard::L); break;
        case 'm': case 'M': this->control.SetKeyboard(Input::Keyboard::M); break;
        case 'n': case 'N': this->control.SetKeyboard(Input::Keyboard::N); break;
        case 'o': case 'O': this->control.SetKeyboard(Input::Keyboard::O); break;
        case 'p': case 'P': this->control.SetKeyboard(Input::Keyboard::P); break;
        case 'q': case 'Q': this->control.SetKeyboard(Input::Keyboard::Q); break;
        case 'r': case 'R': this->control.SetKeyboard(Input::Keyboard::R); break;
        case 's': case 'S': this->control.SetKeyboard(Input::Keyboard::S); break;
        case 't': case 'T': this->control.SetKeyboard(Input::Keyboard::T); break;
        case 'u': case 'U': this->control.SetKeyboard(Input::Keyboard::U); break;
        case 'v': case 'V': this->control.SetKeyboard(Input::Keyboard::V); break;
        case 'w': case 'W': this->control.SetKeyboard(Input::Keyboard::W); break;
        case 'x': case 'X': this->control.SetKeyboard(Input::Keyboard::X); break;
        case 'y': case 'Y': this->control.SetKeyboard(Input::Keyboard::Y); break;
        case 'z': case 'Z': this->control.SetKeyboard(Input::Keyboard::Z); break;
        case ' ': this->control.SetKeyboard(Input::Keyboard::Space); break;
        case '\n': case '\r': this->control.SetKeyboard(Input::Keyboard::Enter); break;
        case 27:  // Escape sequence (for arrow keys on Unix/Linux/macOS)
        {
            char seq[2];
            // Try to read escape sequence non-blockingly
            if (read(STDIN_FILENO, &seq[0], 1) == 1)
            {
                if (seq[0] == '[')
                {
                    if (read(STDIN_FILENO, &seq[1], 1) == 1)
                    {
                        switch (seq[1])
                        {
                        case 'A': this->control.SetKeyboard(Input::Keyboard::Up); break;
                        case 'B': this->control.SetKeyboard(Input::Keyboard::Down); break;
                        case 'C': this->control.SetKeyboard(Input::Keyboard::Right); break;
                        case 'D': this->control.SetKeyboard(Input::Keyboard::Left); break;
                        default: break;
                        }
                    }
                }
                else
                {
                    // Plain escape key
                    this->control.SetKeyboard(Input::Keyboard::Escape);
                }
            }
            else
            {
                // Plain escape key
                this->control.SetKeyboard(Input::Keyboard::Escape);
            }
        } break;
        default:  break;
        }
    }
}


// 从 YAML 节点中读取数组
template <typename T>
std::vector<T> ReadVectorFromYaml(const YAML::Node &node)
{
    std::vector<T> values;                  // 创建一个指定类型 T 的空数组

    // 遍历 YAML 节点中的所有元素
    for (const auto &val : node)
    {   
        // 将 YAML 节点转换为指定的 C++ 类型 (如 float, int, string) 并压入数组
        values.push_back(val.as<T>());
    }

    return values;
}


// 加载并解析 config.yaml 文件
void RL::ReadYaml(const std::string& file_path, const std::string& file_name)
{
    // 拼接出完整的配置文件绝对路径 (POLICY_DIR 通常是一个宏定义，指向模型所在的根目录)
    std::string config_path = std::string(POLICY_DIR) + "/" + file_path + "/" + file_name;
    YAML::Node config;
    try
    {
        config = YAML::LoadFile(config_path)[file_path];
    }
    catch (YAML::BadFile &e)
    {
        std::cout << LOGGER::ERROR << "The file '" << config_path << "' does not exist" << std::endl;
        return;
    }

    for (auto it = config.begin(); it != config.end(); ++it)
    {
        std::string key = it->first.as<std::string>();
        this->params.config_node[key] = it->second;
    }
}


// 初始化 CSV 文件与表头
void RL::CSVInit(std::string robot_path)
{
    // 定义 CSV 文件的存储路径和前缀名
    csv_filename = std::string(POLICY_DIR) + "/" + robot_path + "/motor";

    // Uncomment these lines if need timestamp for file name
    // auto now = std::chrono::system_clock::now();
    // std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    // std::stringstream ss;
    // ss << std::put_time(std::localtime(&now_c), "%Y%m%d%H%M%S");
    // std::string timestamp = ss.str();
    // csv_filename += "_" + timestamp;

    csv_filename += ".csv";
    std::ofstream file(csv_filename.c_str());

    // 动态生成 CSV 表头 (第一行的数据列名)
    // 根据机器人的关节数量 (num_of_dofs)，依次生成 tau_cal_0, tau_cal_1...
    for(int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i) { file << "tau_cal_" << i << ","; }      // 计算力矩
    for(int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i) { file << "tau_est_" << i << ","; }      // 估计力矩 (传感器反馈)
    for(int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i) { file << "joint_pos_" << i << ","; }    // 实际关节位置
    for(int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i) { file << "joint_pos_target_" << i << ","; } // 目标期望位置
    for(int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i) { file << "joint_vel_" << i << ","; }    // 关节速度

    file << std::endl;

    file.close();
}


// 实时追加写入一帧数据
void RL::CSVLogger(const std::vector<float>& torque, const std::vector<float>& tau_est, const std::vector<float>& joint_pos, const std::vector<float>& joint_pos_target, const std::vector<float>& joint_vel)
{
    std::ofstream file(csv_filename.c_str(), std::ios_base::app);

    for(int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i) { file << torque[i] << ","; }
    for(int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i) { file << tau_est[i] << ","; }
    for(int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i) { file << joint_pos[i] << ","; }
    for(int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i) { file << joint_pos_target[i] << ","; }
    for(int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i) { file << joint_vel[i] << ","; }

    file << std::endl;

    file.close();
}


// 状态机状态中的线性插值功能
bool RLFSMState::Interpolate(
    float& percent,                        // 当前插值进度 (0.0 到 1.0)
    const std::vector<float>& start_pos,   // 起始关节角度
    const std::vector<float>& target_pos,  // 目标关节角度
    float duration_seconds,                // 期望经过的时间
    const std::string& description,        // 日志描述
    bool use_fixed_gains)                  // 是否使用固定的Kp/Kd增益
{
    // 插值已完成
    if (percent >= 1.0f)
        return false;

    // 刚开始插值时，如果发现其实已经在目标位置附近，直接跳到1.0完成
    if (percent == 0.0f)
    {
        float max_diff = 0.0f;
        for (size_t i = 0; i < start_pos.size() && i < target_pos.size(); ++i)
        {
            max_diff = std::max(max_diff, std::abs(start_pos[i] - target_pos[i]));
        }

        if (max_diff < 0.1f)
        {
            percent = 1.0f;
        }
    }

    // 根据dt计算走完所需时间所需的帧数，并得出每次步进百分比
    int required_frames = std::max(1, static_cast<int>(std::ceil(duration_seconds / rl.params.Get<float>("dt"))));
    float step = 1.0f / required_frames;

    percent += step;                            // 进度推进
    percent = std::min(percent, 1.0f);          // 钳制在最大1.0

    // 选择对应的PD增益
    auto kp = use_fixed_gains ? rl.params.Get<std::vector<float>>("fixed_kp") : rl.params.Get<std::vector<float>>("rl_kp");
    auto kd = use_fixed_gains ? rl.params.Get<std::vector<float>>("fixed_kd") : rl.params.Get<std::vector<float>>("rl_kd");

    // 遍历所有关节，运用线性插值公式：P_now = (1 - percent)*P_start + percent*P_target
    for (int i = 0; i < rl.params.Get<int>("num_of_dofs"); ++i)
    {
        fsm_command->motor_command.q[i] = (1 - percent) * start_pos[i] + percent * target_pos[i];
        fsm_command->motor_command.dq[i] = 0;                   // 插值过程中速度设0，纯位置控制
        fsm_command->motor_command.kp[i] = kp[i];
        fsm_command->motor_command.kd[i] = kd[i];
        fsm_command->motor_command.tau[i] = 0;                  // 前馈力矩为0
    }

    // 打印进度条
    if (!description.empty())
        LOGGER::PrintProgress(percent, description);

    // 如果还没到1.0，返回true表示仍在插值中
    if (percent >= 1.0f)
        return false;

    return true;
}


// 执行强化学习的主控制链路
void RLFSMState::RLControl()
{
    std::vector<float> _output_dof_pos, _output_dof_vel;

    // 从线程安全的队列 (try_pop) 获取神经网络推理出的最新期望位置和速度
    if (rl.output_dof_pos_queue.try_pop(_output_dof_pos) && rl.output_dof_vel_queue.try_pop(_output_dof_vel))
    {
        // 组装最终电机命令发送下去
        for (int i = 0; i < rl.params.Get<int>("num_of_dofs"); ++i)
        {
            if (!_output_dof_pos.empty())
            {
                fsm_command->motor_command.q[i] = _output_dof_pos[i];
            }
            if (!_output_dof_vel.empty())
            {
                fsm_command->motor_command.dq[i] = _output_dof_vel[i];
            }
            // 填入计算力矩用的PD参数
            fsm_command->motor_command.kp[i] = rl.params.Get<std::vector<float>>("rl_kp")[i];
            fsm_command->motor_command.kd[i] = rl.params.Get<std::vector<float>>("rl_kd")[i];
            fsm_command->motor_command.tau[i] = 0;
        }
    }
}
