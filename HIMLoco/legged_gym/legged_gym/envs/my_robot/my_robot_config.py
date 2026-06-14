from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class MyRobotCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env ):
        num_envs = 4096

    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = 'trimesh'# "heightfield" # none, plane, heightfield or trimesh
        curriculum = True
        measure_heights = True

    class commands( LeggedRobotCfg.commands ):
        curriculum = True
        max_curriculum = 1.0
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [-1.0, 1.0] # min max [m/s]
            lin_vel_y = [-1.0, 1.0]   # min max [m/s]
            ang_vel_yaw = [-3.14, 3.14]    # min max [rad/s]
            heading = [-3.14, 3.14]

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.35]
        default_joint_angles = { 
            'FL_hip_joint': 0.0,   # [rad]
            'RL_hip_joint': 0.0,   # [rad]
            'FR_hip_joint': -0.0 ,  # [rad]
            'RR_hip_joint': -0.0,   # [rad]

            'FL_thigh_joint': 0.9,     # [rad]
            'RL_thigh_joint': 0.9,   # [rad]
            'FR_thigh_joint': 0.9,     # [rad]
            'RR_thigh_joint': 0.9,   # [rad]

            'FL_calf_joint': -1.6,   # [rad]
            'RL_calf_joint': -1.6,    # [rad]
            'FR_calf_joint': -1.6,  # [rad]
            'RR_calf_joint': -1.6,    # [rad]
        }

    class control( LeggedRobotCfg.control ):
        control_type = 'P'
        stiffness = {'joint': 40.0}  
        damping = {'joint': 1.0}     
        action_scale = 0.25
        decimation = 4
        hip_reduction = 0.5

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/my_robot/urdf/my_robot.urdf'
        name = "a1"
        foot_name = "foot"
        penalize_contacts_on = ["FL_hip_link", "FR_hip_link", "RL_hip_link", "RR_hip_link", 
                        "FL_thigh_link", "FR_thigh_link", "RL_thigh_link", "RR_thigh_link", 
                        "FL_calf_link", "FR_calf_link", "RL_calf_link", "RR_calf_link", "base_link"]
        terminate_after_contacts_on = ["base_link" ,"FL_hip_link", "FR_hip_link", "RL_hip_link", "RR_hip_link"]
        privileged_contacts_on = [
            "base_link", 
            "FL_hip_link", "FR_hip_link", "RL_hip_link", "RR_hip_link",
            "FL_thigh_link", "FR_thigh_link", "RL_thigh_link", "RR_thigh_link",
            "FL_calf_link", "FR_calf_link", "RL_calf_link", "RR_calf_link"
        ]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False # Some .obj meshes must be flipped from y-up to z-up
        collapse_fixed_joints = False

    class domain_rand( LeggedRobotCfg.domain_rand ):
        # 基座负载质量
        randomize_payload_mass = True
        payload_mass_range = [-1.0, 3.0]

        # 质心偏移
        randomize_com_displacement = True
        com_displacement_range = [-0.15, 0.15]

        # 连杆质量
        randomize_link_mass = True
        link_mass_range = [0.95, 1.15]

        # 关节摩擦
        randomize_joint_friction = True
        joint_friction_range = [0.1, 2.0]
        
        # 关节阻尼
        randomize_joint_damping = True
        joint_damping_range = [0.1, 2.0]
        
        # 地面摩擦力
        randomize_friction = True
        friction_range = [0.8, 1.2]
        
        # 恢复系数
        randomize_restitution = False
        restitution_range = [0.9, 1.0]
        
        # 电机输出强度
        randomize_motor_strength = True
        motor_strength_range = [0.9, 1.1]
        
        # 比例增益
        randomize_kp = True
        kp_range = [0.9, 1.1]
        
        # 微分增益
        randomize_kd = True
        kd_range = [0.9, 1.1]
        
        # 持续外部扰动
        disturbance = True
        disturbance_range = [-30.0, 30.0]
        disturbance_interval = 8
        
        # 推力扰动
        push_robots = True
        push_interval_s = 16
        max_push_vel_xy = 1.

        # 动作延时
        action_delay = True

        # 观测延时
        obs_delay = False

    class rewards( LeggedRobotCfg.rewards ):
        class scales:
            termination = -100.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.1
            orientation = -0.5
            dof_acc = -2.5e-7
            joint_power = -2e-5
            base_height = -0.1
            foot_clearance = -0.1
            action_rate = -0.01
            smoothness = -0.01
            feet_air_time =  0.5
            collision = -1.0
            feet_stumble = -0.0
            stand_still = -0.5
            zero_command_feet_lift = -5.0
            torques = -0.0
            dof_vel = -0.0
            dof_pos_limits = -1.0
            dof_vel_limits = -0.0
            torque_limits = -0.0
            symmetry = -0.01
            feet_slip = -0.01
            hip_pos = -5.0

        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 0.9 # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        base_height_target = 0.25
        max_contact_force = 100. # forces above this value are penalized
        clearance_height_target = 0.08

    class noise( LeggedRobotCfg.noise ):
        add_noise = True
        noise_level = 1.0 # scales other values
        class noise_scales( LeggedRobotCfg.noise.noise_scales ):
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1
    
    class normalization( LeggedRobotCfg.normalization ):
        class obs_scales( LeggedRobotCfg.normalization.obs_scales ):
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 5.

class MyRobotCfgPPO( LeggedRobotCfgPPO ):
    class policy( LeggedRobotCfgPPO.policy ):
        init_noise_std = 1.0

    class algorithm( LeggedRobotCfgPPO.algorithm ):
        value_loss_coef = 1.0
        clip_param = 0.2
        num_mini_batches = 16
        learning_rate = 1.e-3
        desired_kl = 0.01
        max_grad_norm = 1.
        entropy_coef = 0.005

    class runner( LeggedRobotCfgPPO.runner ):
        max_iterations = 5000
        save_interval = 200
        experiment_name = 'rough_my_robot'
        resume = True
