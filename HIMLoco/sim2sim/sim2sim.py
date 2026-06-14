import yaml
import torch
import mujoco
import mujoco.viewer
import time
import numpy as np
import threading
import os

try:
    import inputs
except ImportError:
    inputs = None

# ==========================================
# 1. 手柄控制类 
# ==========================================
class GamepadController:
    def __init__(self, cfg):
        self.command = np.zeros(3, dtype=np.float32)
        self.lock = threading.Lock()
        self.signs = cfg.get("command_signs", {"x": -1.0, "y": -1.0, "yaw": -1.0})
        self._running = False
        self.has_gamepad = inputs is not None and len(inputs.devices.gamepads) > 0

    def start(self):
        if not self.has_gamepad:
            print("⚠️ 警告: 未检测到物理手柄，控制指令将始终保持为 0.0")
            return
        self._running = True
        threading.Thread(target=self._read_loop, daemon=True).start()

    def get_command(self):
        with self.lock:
            return self.command.copy()

    def _read_loop(self):
        axis_states = {"ABS_Y": 0.0, "ABS_X": 0.0, "ABS_RX": 0.0, "ABS_RY": 0.0, "ABS_Z": 0.0, "ABS_RZ": 0.0}
        while self._running:
            try:
                events = inputs.get_gamepad()
                for event in events:
                    if event.code in axis_states:
                        if 0 <= event.state <= 255:
                            val = (event.state - 127.5) / 127.5
                        else:
                            val = event.state / 32768.0
                        axis_states[event.code] = max(-1.0, min(1.0, val))
                        
                with self.lock:
                    self.command[0] = axis_states["ABS_Y"] * self.signs.get("x", -1.0)
                    self.command[1] = axis_states["ABS_X"] * self.signs.get("y", -1.0)
                    self.command[2] = axis_states["ABS_Z"] * self.signs.get("yaw", -1.0) 
            except Exception as e:
                print(f"手柄连接异常: {e}。指令归零。")
                with self.lock:
                    self.command[:] = 0.0
                time.sleep(1.0)

# ==========================================
# 2. 仿真环境类
# ==========================================
class Sim2SimEnv:
    def __init__(self, cfg_path="config.yaml"):
        with open(cfg_path, "r") as f:
            self.cfg = yaml.safe_load(f)
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"✅ Using device: {self.device}")
        
        self.paths = self.cfg["paths"]
        self.joint_names = self.cfg["joint_names"]
        self.num_joints = len(self.joint_names)
        
        # 将静态参数提前转化为 Tensor 放于 GPU
        self.default_dof_pos = torch.tensor(self.cfg["default_dof_pos"], dtype=torch.float32, device=self.device)
        self.base_init_pos = np.array(self.cfg.get("base_init_pos", [0.0, 0.0, 0.35]), dtype=np.float32)
        self.base_init_quat = np.array(self.cfg.get("base_init_quat", [1.0, 0.0, 0.0, 0.0]), dtype=np.float32)
        
        self.p_gains = torch.tensor(self.cfg["p_gains"], dtype=torch.float32, device=self.device) 
        self.d_gains = torch.tensor(self.cfg["d_gains"], dtype=torch.float32, device=self.device)
        self.actions_scale = self._build_actions_scale()
        
        # 观测缩放因子预计算
        sf = self.cfg["scale_factors"]
        self.scale_ang_vel = sf["scale_ang_vel"]
        self.scale_dof_pos = sf["scale_dof_pos"]
        self.scale_dof_vel = sf["scale_dof_vel"]
        self.command_scale_tensor = torch.tensor([sf["scale_lin_vel"], sf["scale_lin_vel"], sf["scale_ang_vel"]], device=self.device)
        
        self.num_obs_current = self.cfg["num_obs_current"]
        self.num_obs_encoder = self.cfg["num_obs_encoder"]
        self.warmup_steps = int(self.cfg.get("warmup_steps", 200))

        # 初始化 MuJoCo
        if not os.path.exists(self.paths["scene_xml"]):
            raise FileNotFoundError(f"找不到 MuJoCo XML 文件: {self.paths['scene_xml']}")
            
        self.m = mujoco.MjModel.from_xml_path(self.paths["scene_xml"])
        self.d = mujoco.MjData(self.m)
        self.m.opt.timestep = self.cfg.get("timestep", 0.005)
        self.m.opt.gravity = (0, 0, -9.81)
        
        # 获取机身 Body ID 用于计算绝对安全的重力投影
        self.base_body_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        if self.base_body_id == -1:
            self.base_body_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "base")
        if self.base_body_id == -1:
            self.base_body_id = 1 # 默认回退到 Tracking 目标 ID

        # 🚨 核心修复：显式记录执行器 ID 列表，彻底废弃不稳定的 argsort 隐式映射
        self.actuator_ids = self._build_actuator_ids_list()
        self.tau_limit = self._get_actuator_tau_limit()
        
        # 加载 TorchScript 策略模型
        policy_path = self.paths["policy_path"]
        if not policy_path.endswith(".pt"):
            raise ValueError(f"策略模型必须是 TorchScript .pt 文件，当前路径: {policy_path}")

        try:
            self.policy = torch.jit.load(policy_path, map_location=self.device)
            self.policy.eval()
            print(f"✅ 成功加载 TorchScript 策略模型: {policy_path}")
        except Exception as e:
            raise RuntimeError(f"加载 TorchScript 策略模型失败: {policy_path}") from e

    def _build_actuator_ids_list(self):
        """显式获取与 config.yaml 中 joint_names 严格对应的 MuJoCo 执行器 ID 数组"""
        actuator_ids = []
        for joint_name in self.joint_names:
            actuator_name = joint_name[:-6] if joint_name.endswith("_joint") else joint_name
            actuator_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
            if actuator_id == -1: 
                actuator_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name)
                if actuator_id == -1:
                    raise RuntimeError(f"未找到执行器 (Actuator): '{actuator_name}' 或 '{joint_name}'")
            actuator_ids.append(actuator_id)
        return actuator_ids

    def _build_actions_scale(self):
        raw_scale = self.cfg["actions_scale"]
        if isinstance(raw_scale, (int, float)):
            return torch.full((self.num_joints,), float(raw_scale), dtype=torch.float32, device=self.device)

        if not isinstance(raw_scale, list):
            raise ValueError("actions_scale 必须是标量或与 joint_names 等长的列表")

        if len(raw_scale) != self.num_joints:
            raise ValueError(f"actions_scale 长度必须为 {self.num_joints}，实际为 {len(raw_scale)}")

        try:
            scale = torch.tensor(raw_scale, dtype=torch.float32, device=self.device)
        except (TypeError, ValueError) as exc:
            raise ValueError("actions_scale 列表必须只包含数值") from exc

        if not torch.isfinite(scale).all():
            raise ValueError("actions_scale 包含非有限数值 (NaN/Inf)")

        return scale

    def _get_actuator_tau_limit(self):
        default_limit = torch.full((self.num_joints,), 20.0, dtype=torch.float32, device=self.device)
        limits = []
        for joint_name in self.joint_names:
            joint_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id == -1: return default_limit
            
            actuator_ids = np.where(self.m.actuator_trnid[:, 0] == joint_id)[0]
            if len(actuator_ids) == 0: return default_limit
            
            ctrlrange = self.m.actuator_ctrlrange[actuator_ids[0]]
            limit = min(abs(float(ctrlrange[0])), abs(float(ctrlrange[1])))
            limits.append(limit if limit > 0 else 20.0)
            
        return torch.tensor(limits, dtype=torch.float32, device=self.device)

    def reset(self):
        mujoco.mj_resetData(self.m, self.d)
        for i, name in enumerate(self.joint_names):
            jnt_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, name)
            self.d.qpos[self.m.jnt_qposadr[jnt_id]] = self.default_dof_pos[i].item()
            self.d.qvel[self.m.jnt_dofadr[jnt_id]] = 0.0

        self.d.qpos[:3] = self.base_init_pos
        self.d.qpos[3:7] = self.base_init_quat
        self.d.qvel[:] = 0.0
        mujoco.mj_forward(self.m, self.d)

    def get_sensor_data(self, name):
        id_ = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SENSOR, name)
        if id_ == -1:
            raise RuntimeError(f"MuJoCo 模型中未找到传感器: '{name}'")
        adr, dim = self.m.sensor_adr[id_], self.m.sensor_dim[id_]
        return torch.tensor(self.d.sensordata[adr:adr+dim], device=self.device, dtype=torch.float32)

    def get_joint_state(self):
        pos_list, vel_list = [], []
        for n in self.joint_names:
            joint = self.m.joint(n)
            pos_list.append(self.d.qpos[joint.qposadr[0]])
            vel_list.append(self.d.qvel[joint.dofadr[0]])
        return (
            torch.tensor(pos_list, device=self.device, dtype=torch.float32),
            torch.tensor(vel_list, device=self.device, dtype=torch.float32),
        )

    def get_obs(self, last_actions, commands_tensor):
        imu_gyro = self.get_sensor_data("angular-velocity")
        dof_pos, dof_vel = self.get_joint_state()
        
        # 🚨 核心修复：使用旋转矩阵（xmat）绝对安全地计算重力投影
        R = self.d.xmat[self.base_body_id].reshape(3, 3)
        projected_gravity = torch.tensor([-R[2, 0], -R[2, 1], -R[2, 2]], device=self.device, dtype=torch.float32)                   

        return torch.cat([
            commands_tensor * self.command_scale_tensor,
            imu_gyro * self.scale_ang_vel,
            projected_gravity,
            (dof_pos - self.default_dof_pos) * self.scale_dof_pos, 
            dof_vel * self.scale_dof_vel,
            last_actions
        ], dim=-1)

    def compute_pd_torques(self, target_dof_pos):
        curr_pos, curr_vel = self.get_joint_state()
        torques = self.p_gains * (target_dof_pos - curr_pos) - self.d_gains * curr_vel
        return torch.clip(torques, -self.tau_limit, self.tau_limit)

    def run(self):
        self.reset()
        
        # 🎮 初始化并启动手柄控制器
        gamepad = GamepadController(self.cfg)
        gamepad.start()
        print("🎮 手柄控制模式已启用。请确保手柄已正确连接。")
        
        # 获取初始指令（此时一般全为0）
        cmd_np = gamepad.get_command()
        commands_tensor = torch.tensor(cmd_np, device=self.device, dtype=torch.float32)
        
        print("⏳ 正在进行预热 (Warmup)...")
        # 热身仿真 (Warmup)
        for _ in range(self.warmup_steps):
            torques = self.compute_pd_torques(self.default_dof_pos)
            self.d.ctrl[self.actuator_ids] = torques.detach().cpu().numpy()
            mujoco.mj_step(self.m, self.d)
            
        print("✅ 预热完成！")
        
        last_actions = torch.zeros(self.num_joints, device=self.device)
        
        # 获取第一帧真实观测并铺满历史 Queue
        obs_now_init = self.get_obs(last_actions, commands_tensor)
        obs_chunk_init = obs_now_init[:self.num_obs_current].detach().cpu().numpy()
        obs_encoder = np.tile(obs_chunk_init, int(self.num_obs_encoder / self.num_obs_current))

        target_dof_pos = self.default_dof_pos.clone()
        decimation = self.cfg.get("sim_steps_per_loop", 4)
        step = 0

        with mujoco.viewer.launch_passive(self.m, self.d) as viewer:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = self.base_body_id
            viewer.cam.distance, viewer.cam.elevation = 1.5, -20
            viewer.cam.lookat = np.array([0.0, 0.0, 0.5])

            while viewer.is_running():
                # --- 策略推理层 (50Hz) ---
                if step % decimation == 0:
                    # 🎮 从手柄获取最新的控制输入 [x, y, yaw]，并刷新 commands_tensor
                    cmd_np = gamepad.get_command()
                    commands_tensor = torch.tensor(cmd_np, device=self.device, dtype=torch.float32)

                    obs_now = self.get_obs(last_actions, commands_tensor)
                    obs_chunk = obs_now[:self.num_obs_current].detach().cpu().numpy()

                    # 滑动窗口更新历史缓存 (新观测插入头部，旧观测被挤压出尾部)
                    obs_encoder = np.concatenate((obs_chunk, obs_encoder[:-self.num_obs_current]))

                    # TorchScript 推理
                    obs_tensor = torch.as_tensor(
                        obs_encoder,
                        dtype=torch.float32,
                        device=self.device,
                    ).reshape(1, self.num_obs_encoder)
                    
                    with torch.no_grad():
                        actions = self.policy(obs_tensor)
                        
                    actions_np = actions.detach().cpu().numpy().reshape(-1)
                    if actions_np.shape[0] != self.num_joints:
                        raise RuntimeError(f"策略输出异常: 期望长度 {self.num_joints}, 实际得到 {actions_np.shape[0]}")
                        
                    if not np.isfinite(actions_np).all():
                        raise RuntimeError("策略输出包含非有限数值 (NaN/Inf)，物理系统已崩塌！")
                    
                    policy_actions = torch.tensor(actions_np, device=self.device, dtype=torch.float32)
                    policy_actions = torch.clip(policy_actions, -5.0, 5.0)
                    
                    # 映射目标位置
                    target_candidate = self.default_dof_pos + policy_actions * self.actions_scale
                    
                    last_actions = policy_actions.clone()
                    target_dof_pos = target_candidate

                # --- 物理控制层 (200Hz) ---
                torques = self.compute_pd_torques(target_dof_pos)
                self.d.ctrl[self.actuator_ids] = torques.detach().cpu().numpy()
                mujoco.mj_step(self.m, self.d)
                
                step += 1
                viewer.sync()

if __name__ == "__main__":
    env = Sim2SimEnv("config.yaml")
    env.run()
