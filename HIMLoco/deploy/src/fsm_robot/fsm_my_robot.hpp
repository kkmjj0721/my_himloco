#ifndef MY_ROBOT_FSM_HPP
#define MY_ROBOT_FSM_HPP

#include "fsm.hpp"
#include "rl_sdk.hpp"


namespace my_robot_fsm
{

/* 
 * @brief : 状态 1：被动/安全状态 (Passive State),此时电机不发力追位置，只提供阻尼，机器人处于“软趴”状态
 *
 */
class RLFSMStatePassive : public RLFSMState
{
public:
    // 构造函数，初始化父类并命名该状态为 "RLFSMStatePassive"
    RLFSMStatePassive(RL *rl) : RLFSMState(*rl, "RLFSMStatePassive") {}

    // 进入该状态时触发一次
    void Enter() override
    {
        // 打印提示信息，告诉操作员如何切换到起身状态
        std::cout << LOGGER::NOTE << "Entered passive mode. Press '0' (Keyboard) or 'A' (Gamepad) to switch to RLFSMStateGetUp." << std::endl;
    }

    // 该状态的主循环，每帧执行
    void Run() override
    {
        for (int i = 0; i < rl.params.Get<int>("num_of_dofs"); ++i)
        {
            // fsm_command->motor_command.q[i] = fsm_state->motor_state.q[i];

            fsm_command->motor_command.dq[i] = 0;  // 期望速度为0
            fsm_command->motor_command.kp[i] = 0;  // 位置刚度设为0 (完全不出力维持位置)
            fsm_command->motor_command.kd[i] = 8;  // 关键！阻尼设为8。这使得关节可以被外力掰动，但会有一定阻力，防止机器狗自由落体砸坏
            fsm_command->motor_command.tau[i] = 0; // 前馈力矩为0
        }
    }

    // 退出状态时的清理工作（此处为空）
    void Exit() override {}

    // 检查是否需要切换状态
    std::string CheckChange() override
    {
        // 如果按下键盘 '0' 或手柄 'A' 键
        if (rl.control.current_keyboard == Input::Keyboard::Num0 || rl.control.current_gamepad == Input::Gamepad::A)
        {
            // 切换到起身状态
            return "RLFSMStateGetUp";
        }
        // 否则保持当前状态不变
        return state_name_;
    }
};


// ==========================================
// 状态 2：起身状态 (GetUp State)
// 从地上平滑过渡到站立姿态
// ==========================================
class RLFSMStateGetUp : public RLFSMState
{
public:
    RLFSMStateGetUp(RL *rl) : RLFSMState(*rl, "RLFSMStateGetUp") {}

    float percent_pre_getup = 0.0f;                     // 第一阶段起身插值进度
    float percent_getup = 0.0f;                         // 第二阶段起身插值进度

    // 预备起身的过渡姿态（通常是收腿/半蹲的姿势，防止直接站立导致腿部卡住或打滑）
    std::vector<float> pre_running_pos = {
        0.00, 1.36, -2.5,              // FL
        0.00, 1.36, -2.5,              // FR
        0.00, 1.36, -2.5,              // RL
        0.00, 1.36, -2.5,              // RR
    };

    // 标记是否是从 Passive 状态过来的
    bool stand_from_passive = true;

    void Enter() override
    {
        percent_pre_getup = 0.0f;           // 重置进度条
        percent_getup = 0.0f;   

        // 判断上一个状态是不是 Passive
        if (rl.fsm.previous_state_->GetStateName() == "RLFSMStatePassive")
        {
            stand_from_passive = true;
        }
        else
        {
            stand_from_passive = false;
        }

        rl.now_state = *fsm_state;                  // 记录当前实际状态
        rl.start_state = rl.now_state;              // 将当前状态作为插值起点
    }

    void Run() override
    {
        // 如果是从地上软趴状态站起来（需要两段式起身）
        if(stand_from_passive)
        {
            // 第一段：从当前凌乱状态 -> 插值到规整的半蹲姿势 (耗时1秒)
            if (Interpolate(percent_pre_getup, rl.now_state.motor_state.q, pre_running_pos, 1.0f, "Pre Getting up", true)) return;
            // 第二段：从半蹲姿势 -> 插值到标准站立姿势 default_dof_pos (耗时2秒)
            if (Interpolate(percent_getup, pre_running_pos, rl.params.Get<std::vector<float>>("default_dof_pos"), 2.0f, "Getting up", true)) return;
        }
        // 如果不是从地上起来（可能是从趴下状态中断恢复），直接单段插值站立
        else
        {
            if (Interpolate(percent_getup, rl.now_state.motor_state.q, rl.params.Get<std::vector<float>>("default_dof_pos"), 1.0f, "Getting up", true)) return;
        }
    }
    
    // 退出状态时的清理工作（此处为空）
    void Exit() override {}

    std::string CheckChange() override
    {
        // 随时允许按下 P 键或手柄 LB+X 紧急切回被动状态
        if (rl.control.current_keyboard == Input::Keyboard::P || rl.control.current_gamepad == Input::Gamepad::LB_X)
            return "RLFSMStatePassive";

        // 只有当起身动作完全结束 (percent_getup >= 1.0f) 后，才允许切换到行走或趴下
        if (percent_getup >= 1.0f)
        {
            if (rl.control.current_keyboard == Input::Keyboard::Num1 || rl.control.current_gamepad == Input::Gamepad::RB_DPadUp)
                return "RLFSMStateRLLocomotion";            // 切换到强化学习控制状态

            else if (rl.control.current_keyboard == Input::Keyboard::Num9 || rl.control.current_gamepad == Input::Gamepad::B)
                return "RLFSMStateGetDown";                 // 切换到趴下状态
        }
        return state_name_;
    }
};


// ==========================================
// 状态 3：趴下状态 (GetDown State)
// 从站立平滑过渡到贴地
// ==========================================
class RLFSMStateGetDown : public RLFSMState
{
public:
    RLFSMStateGetDown(RL *rl) : RLFSMState(*rl, "RLFSMStateGetDown") {}

    // 趴下插值进度 
    float percent_getdown = 0.0f;

    void Enter() override
    {
        percent_getdown = 0.0f;
        rl.now_state = *fsm_state;          // 记录趴下瞬间的初始位置
    }


    void Run() override
    {
        // 从当前位置平滑插值到 rl.start_state (通常是机器人系统刚上电时的零位/趴下位)，耗时2秒
        Interpolate(percent_getdown, rl.now_state.motor_state.q, rl.start_state.motor_state.q, 2.0f, "Getting down", true);
    }

    // 退出状态时的清理工作（此处为空）
    void Exit() override {}

    std::string CheckChange() override
    {
        // 如果按下紧急 P 键，或者趴下的进度条走完了 (>= 1.0)，自动切入被动断力模式
        if (rl.control.current_keyboard == Input::Keyboard::P || rl.control.current_gamepad == Input::Gamepad::LB_X || percent_getdown >= 1.0f)
            return "RLFSMStatePassive";
        
        else if (rl.control.current_keyboard == Input::Keyboard::Num0 || rl.control.current_gamepad == Input::Gamepad::A)
            return "RLFSMStateGetUp";

        return state_name_;
    }
};


// ==========================================
// 状态 4：RL 运动状态 (Locomotion State)
// 将控制权交接给强化学习神经网络
// ==========================================
class RLFSMStateRLLocomotion : public RLFSMState
{
public:
    RLFSMStateRLLocomotion(RL *rl) : RLFSMState(*rl, "RLFSMStateRLLocomotion") {}

    float percent_transition = 0.0f;

    void Enter() override
    {
        percent_transition = 0.0f;
        rl.episode_length_buf = 0;              // 重置强化学习的步数计数器

        // 指定要加载的模型配置名称（这里写死为 himloco）
        rl.config_name = "himloco";
        std::string robot_config_path = rl.robot_name + "/" + rl.config_name;
        try
        {
            rl.InitRL(robot_config_path);               // 初始化强化学习模型（加载 ONNX/YAML 等）
            rl.now_state = *fsm_state;                  // 记录进入该状态时的身体状态
        }       
        catch (const std::exception& e)
        {
            // 如果模型加载失败，打印错误，并强制退回到安全被动状态
            std::cout << LOGGER::ERROR << "InitRL() failed: " << e.what() << std::endl;
            rl.rl_init_done = false;
            rl.fsm.RequestStateChange("RLFSMStatePassive");
        }
    }

    void Run() override
    {
        // position transition from last default_dof_pos to current default_dof_pos
        // if (Interpolate(percent_transition, rl.now_state.motor_state.q, rl.params.Get<std::vector<float>>("default_dof_pos"), 0.5f, "Policy transition", true)) return;

        if (!rl.rl_init_done) rl.rl_init_done = true;           // 标记初始化成功

        // 在终端同一行持续覆盖打印当前的期望速度控制量 (x, y, yaw)
        std::cout << "\r\033[K" << std::flush << LOGGER::INFO << "RL Controller [" << rl.config_name << "] x:" << rl.control.x << " y:" << rl.control.y << " yaw:" << rl.control.yaw << std::flush;

        // 执行核心的 RL 推理结果分发 (调用 rl_sdk 中的函数)
        RLControl();
    }

    // 退出状态时的清理工作（此处为空）
    void Exit() override
        rl.rl_init_done = false;        // 退出 RL 状态时，重置标志位


    std::string CheckChange() override
    {   
        // 允许通过多种按键切出 RL 运动模式
        if (rl.control.current_keyboard == Input::Keyboard::P || rl.control.current_gamepad == Input::Gamepad::LB_X)
            return "RLFSMStatePassive";             // 紧急停止 -> 被动软趴
        
        else if (rl.control.current_keyboard == Input::Keyboard::Num9 || rl.control.current_gamepad == Input::Gamepad::B)
            return "RLFSMStateGetDown";             // 平缓停下 -> 趴下

        else if (rl.control.current_keyboard == Input::Keyboard::Num0 || rl.control.current_gamepad == Input::Gamepad::A)
            return "RLFSMStateGetUp";               // 重新执行站立初始化
        
        else if (rl.control.current_keyboard == Input::Keyboard::Num1 || rl.control.current_gamepad == Input::Gamepad::RB_DPadUp)
            return "RLFSMStateRLLocomotion";            // 重复按 1 键可以重新进入本状态 (刷新 RL 模型)

        return state_name_;
    }
};

} // namespace go2_fsm


// ==========================================
// 状态机工厂 (FSM Factory)
// 负责根据字符串名称，动态实例化具体的 State 对象
// ==========================================
class MyRobotFSMFactory : public FSMFactory
{
public:
    MyRobotFSMFactory(const std::string& initial) : initial_state_(initial) {}

    // 工厂模式核心：传入一个名字，返回对应的状态对象指针
    std::shared_ptr<FSMState> CreateState(void *context, const std::string &state_name) override
    {
        // 将上下文指针强转回 RL 指针
        RL *rl = static_cast<RL *>(context);

        if (state_name == "RLFSMStatePassive")
            return std::make_shared<my_robot_fsm::RLFSMStatePassive>(rl);
        else if (state_name == "RLFSMStateGetUp")
            return std::make_shared<my_robot_fsm::RLFSMStateGetUp>(rl);
        else if (state_name == "RLFSMStateGetDown")
            return std::make_shared<my_robot_fsm::RLFSMStateGetDown>(rl);
        else if (state_name == "RLFSMStateRLLocomotion")
            return std::make_shared<my_robot_fsm::RLFSMStateRLLocomotion>(rl);
        return nullptr;
    }
    std::string GetType() const override { return "my_robot"; }                 // 表明该工厂是专为 my_robot 配置的

    std::vector<std::string> GetSupportedStates() const override                // 返回该工厂支持的所有状态列表
    {
        return {
            "RLFSMStatePassive",
            "RLFSMStateGetUp",
            "RLFSMStateGetDown",
            "RLFSMStateRLLocomotion"
        };
    }

    // 获取默认启动状态
    std::string GetInitialState() const override { return initial_state_; }

private:
    std::string initial_state_;             // 启动状态
};

REGISTER_FSM_FACTORY(MyRobotFSMFactory, "RLFSMStatePassive")

#endif // MY_ROBOT_FSM_HPP
