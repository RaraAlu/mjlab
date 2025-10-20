"""用RSL-RL强化学习框架来运行和演示RL智能体的脚本。"""

import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Optional, cast

import gymnasium as gym
import torch
import tyro
from rsl_rl.runners import OnPolicyRunner
from typing_extensions import assert_never

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from mjlab.third_party.isaaclab.isaaclab_tasks.utils.parse_cfg import (
  load_cfg_from_registry,
)
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserViewer

import mjlab.tasks  # 触发任务注册
import pygame

@dataclass(frozen=True)
class PlayConfig:
  """
  播放/演示配置数据类,定义了运行智能体时的所有参数。
  frozen=True 表示这个数据类是不可变的。
  """
  # 智能体类型：zero(零动作), random(随机动作), trained(训练好的模型)
  agent: Literal["zero", "random", "trained", "joystick"] = "trained"
  # 检查点文件路径，用于加载训练好的模型
  checkpoint_file: str | None = None
  # 环境数量，如果指定则覆盖配置中的默认值
  num_envs: int | None = None
  # 计算设备：cuda:0(GPU) 或 cpu
  device: str | None = None
  # 是否录制视频
  video: bool = False
  # 视频长度(步数)
  video_length: int = 200
  # 视频高度(像素)
  video_height: int | None = None
  # 视频宽度(像素)
  video_width: int | None = None
  # 摄像机编号或名称
  camera: int | str | None = None
  # 查看器类型：native(原生MuJoCo查看器) 或 viser(Web查看器)
  viewer: Literal["native", "viser"] = "native"
  
  # 🎮 手柄配置参数
  joystick_id: int = 0  # 手柄设备ID（如果有多个手柄）
  joystick_deadzone: float = 0.1  # 摇杆死区（避免漂移）
  joystick_scale: float = 1.0  # 动作缩放系数
  debug_joystick: bool = True  # 🆕 是否启用手柄调试打印

class PolicyJoystick:
  """
  使用游戏手柄控制机器人的策略（混合控制模式）
  
  策略架构：
    手柄 → 高层运动指令(vx, vy, vyaw) → 训练好的RL策略 → 关节动作
  
  支持的手柄映射：
  - 左摇杆垂直轴 (Axis 1): 前进/后退速度 (vx)
  - 左摇杆水平轴 (Axis 0): 左右平移速度 (vy)  
  - 右摇杆水平轴 (Axis 2): 转向速度 (vyaw)
  """
  
  def __init__(
    self, 
    trained_policy,  # 🔑 关键：需要训练好的策略
    device: str,
    joystick_id: int = 0,
    deadzone: float = 0.15,
    max_lin_vel: float = 1.0,  # 最大线速度 (m/s)
    max_ang_vel: float = 1.0,  # 最大角速度 (rad/s)
    debug: bool = True,  # 🆕 调试模式开关
  ):
    
    self.trained_policy = trained_policy
    self.device = device
    self.deadzone = deadzone
    self.max_lin_vel = max_lin_vel
    self.max_ang_vel = max_ang_vel
    self.debug = debug  # 🆕
    self.step_count = 0  # 🆕 步数计数器
    
    # 初始化 pygame 和手柄
    pygame.init()
    pygame.joystick.init()
    
    # 检查手柄连接
    joystick_count = pygame.joystick.get_count()
    if joystick_count == 0:
        raise RuntimeError("❌ No joystick detected! Please connect a controller.")
    
    if joystick_id >= joystick_count:
        raise ValueError(
            f"❌ Joystick ID {joystick_id} not found. "
            f"Available IDs: 0-{joystick_count-1}"
        )
    
    # 连接手柄
    self.joystick = pygame.joystick.Joystick(joystick_id)
    self.joystick.init()
    
    # 🆕 获取手柄的轴和按钮数量
    self.num_axes = self.joystick.get_numaxes()
    self.num_buttons = self.joystick.get_numbuttons()
    
    # 打印手柄信息
    print(f"\n{'='*60}")
    print(f"🎮 Joystick Control Mode (Hybrid)")
    print(f"{'='*60}")
    print(f"  Controller: {self.joystick.get_name()}")
    print(f"  Axes: {self.num_axes}")
    print(f"  Buttons: {self.num_buttons}")
    print(f"")
    print(f"  📋 Control Mapping:")
    print(f"     Left Stick Y  (Axis 1) → Forward/Backward (vx)")
    print(f"     Left Stick X  (Axis 0) → Left/Right (vy)")
    print(f"     Right Stick X (Axis 2) → Rotate (vyaw)")
    print(f"")
    print(f"  ⚙️  Parameters:")
    print(f"     Max Linear Velocity:  {max_lin_vel} m/s")
    print(f"     Max Angular Velocity: {max_ang_vel} rad/s")
    print(f"     Deadzone: {deadzone}")
    print(f"     Debug Mode: {'✅ ENABLED' if debug else '❌ DISABLED'}")
    print(f"{'='*60}\n")
    
    # 🆕 测试手柄初始读取
    print("🔍 Testing initial joystick read...")
    self._test_joystick_read()
        
  def _apply_deadzone(self, value: float) -> float:
      """应用死区，避免摇杆漂移"""
      if abs(value) < self.deadzone:
          return 0.0
      # 重新映射到 [-1, 1] 范围
      sign = 1 if value > 0 else -1
      return sign * (abs(value) - self.deadzone) / (1.0 - self.deadzone)
  
  def _test_joystick_read(self):
    """🆕 测试手柄读取功能"""
    try:
      pygame.event.pump()
      print("  ✅ pygame.event.pump() successful")
      
      # 读取所有轴的值
      print(f"  📊 All axes values:")
      for i in range(self.num_axes):
        axis_val = self.joystick.get_axis(i)
        print(f"     Axis {i}: {axis_val:+.4f}")
      
      # 读取所有按钮状态
      print(f"  🔘 All button states:")
      pressed_buttons = [i for i in range(self.num_buttons) if self.joystick.get_button(i)]
      if pressed_buttons:
        print(f"     Pressed: {pressed_buttons}")
      else:
        print(f"     None pressed")
      
      print("  ✅ Joystick test completed\n")
    except Exception as e:
      print(f"  ❌ Joystick test failed: {e}\n")
  
  def _read_velocity_command(self) -> torch.Tensor:
    """
    读取手柄状态并转换为速度指令
    
    Returns:
        velocity_cmd: 形状为 (num_envs, 3) 的张量 [vx, vy, vyaw]
    """
    # 更新手柄状态
    pygame.event.pump()
    
    # 🆕 调试信息：显示当前步数
    if self.debug and self.step_count % 50 == 0:  # 每50步打印一次详细信息
      print(f"\n{'─'*60}")
      print(f"🔍 DEBUG [Step {self.step_count}] - Joystick State")
      print(f"{'─'*60}")
    
    # 读取摇杆轴（注意：某些手柄的Y轴是反的）
    # Axis 1: 左摇杆垂直（前进/后退）- 通常需要反转
    raw_vx = -self.joystick.get_axis(1) if self.num_axes > 1 else 0.0
    # Axis 0: 左摇杆水平（左右平移）
    raw_vy = self.joystick.get_axis(0) if self.num_axes > 0 else 0.0
    # Axis 2 或 3: 右摇杆水平（转向）
    raw_vyaw = self.joystick.get_axis(3) if self.num_axes > 2 else 0.0
    
    # 🆕 详细调试打印
    if self.debug and self.step_count % 50 == 0:
      print(f"  📥 Raw Axis Values:")
      print(f"     Axis 0 (Left X):  {self.joystick.get_axis(0):+.4f}")
      print(f"     Axis 1 (Left Y):  {self.joystick.get_axis(1):+.4f}")
      if self.num_axes > 2:
        print(f"     Axis 2 (Right X): {self.joystick.get_axis(2):+.4f}")
      if self.num_axes > 3:
        print(f"     Axis 3 (Right Y): {self.joystick.get_axis(3):+.4f}")
      
      print(f"\n  🎯 Mapped Raw Values (before deadzone):")
      print(f"     raw_vx:   {raw_vx:+.4f}")
      print(f"     raw_vy:   {raw_vy:+.4f}")
      print(f"     raw_vyaw: {raw_vyaw:+.4f}")
    
    # 应用死区
    vx_normalized = self._apply_deadzone(raw_vx)
    vy_normalized = self._apply_deadzone(raw_vy)
    vyaw_normalized = self._apply_deadzone(raw_vyaw)
    
    # 缩放到实际速度
    vx = vx_normalized * self.max_lin_vel
    vy = -vy_normalized * self.max_lin_vel
    vyaw = -vyaw_normalized * self.max_ang_vel
    
    # 🆕 详细调试打印
    if self.debug and self.step_count % 50 == 0:
      print(f"\n  ⚙️  After Deadzone ({self.deadzone}):")
      print(f"     vx_norm:   {vx_normalized:+.4f}")
      print(f"     vy_norm:   {vy_normalized:+.4f}")
      print(f"     vyaw_norm: {vyaw_normalized:+.4f}")
      
      print(f"\n  🚀 Final Velocity Commands:")
      print(f"     vx:   {vx:+.4f} m/s  (max: ±{self.max_lin_vel})")
      print(f"     vy:   {vy:+.4f} m/s  (max: ±{self.max_lin_vel})")
      print(f"     vyaw: {vyaw:+.4f} rad/s (max: ±{self.max_ang_vel})")
      print(f"{'─'*60}\n")
    
    # 🆕 实时简化打印（每步都显示，但只在有明显输入时）
    if self.debug and (abs(vx) > 0.01 or abs(vy) > 0.01 or abs(vyaw) > 0.01):
      print(f"\r🎮 [Step {self.step_count:4d}] Command: "
            f"vx={vx:+.2f} vy={vy:+.2f} vyaw={vyaw:+.2f}  ", end="")
    elif self.debug and self.step_count % 100 == 0:
      # 即使没有输入，也定期显示状态
      print(f"\r🎮 [Step {self.step_count:4d}] Command: "
            f"vx={vx:+.2f} vy={vy:+.2f} vyaw={vyaw:+.2f} (idle)", end="")
    
    return torch.tensor([vx, vy, vyaw], device=self.device)
    
  def __call__(self, obs: dict) -> torch.Tensor:
    """
    策略调用接口（混合控制模式）
    
    流程：
      1. 从手柄读取速度指令
      2. 修改观测中的 command 字段
      3. 调用训练好的策略生成关节动作
    
    Args:
        obs: 环境观测值字典，包含 'policy' 键
    
    Returns:
        动作张量，由训练好的策略生成
    """
    # 🆕 步数计数
    self.step_count += 1
    
    # 🆕 调试：显示原始观测信息
    if self.debug and self.step_count % 50 == 0:
      print(f"\n{'═'*60}")
      print(f"🧠 DEBUG [Step {self.step_count}] - Policy Call")
      print(f"{'═'*60}")
      print(f"  📊 Original Observation:")
      print(f"     obs['policy'] shape: {obs['policy'].shape}")
      print(f"     obs['policy'] device: {obs['policy'].device}")
      print(f"     Original command (last 3 dims): {obs['policy'][0, -3:].cpu().numpy()}")
    
    # 读取手柄指令
    velocity_cmd = self._read_velocity_command()
    
    # 🆕 调试：显示读取的速度指令
    if self.debug and self.step_count % 50 == 0:
      print(f"\n  🎮 Joystick Velocity Command:")
      print(f"     velocity_cmd: {velocity_cmd.cpu().numpy()}")
      print(f"     velocity_cmd device: {velocity_cmd.device}")
    
    # 🔑 关键：修改观测中的 command 部分
    obs_policy = obs['policy'].clone()  # 避免修改原始观测
    
    # 获取环境数量
    num_envs = obs_policy.shape[0]
    
    # 🆕 调试：显示环境信息
    if self.debug and self.step_count % 50 == 0:
      print(f"\n  🌍 Environment Info:")
      print(f"     num_envs: {num_envs}")
    
    # 将手柄指令广播到所有环境
    velocity_cmd_batch = velocity_cmd.unsqueeze(0).repeat(num_envs, 1)
    
    # 🆕 调试：显示广播后的指令
    if self.debug and self.step_count % 50 == 0:
      print(f"\n  📡 Broadcast Command:")
      print(f"     velocity_cmd_batch shape: {velocity_cmd_batch.shape}")
      print(f"     First env command: {velocity_cmd_batch[0].cpu().numpy()}")
      if num_envs > 1:
        print(f"     Last env command:  {velocity_cmd_batch[-1].cpu().numpy()}")
    
    # 替换观测中的 command 部分（最后3个维度）
    old_command = obs_policy[0, -3:].clone()  # 🆕 保存旧值用于对比
    obs_policy[:, -3:] = velocity_cmd_batch
    
    # 🆕 调试：对比修改前后
    if self.debug and self.step_count % 50 == 0:
      print(f"\n  🔄 Command Replacement:")
      print(f"     Old command: {old_command.cpu().numpy()}")
      print(f"     New command: {obs_policy[0, -3:].cpu().numpy()}")
      print(f"     ✅ Command replaced successfully!")
    
    # 使用修改后的观测调用训练好的策略
    modified_obs = {'policy': obs_policy}
    if 'critic' in obs:
      modified_obs['critic'] = obs['critic']
    
    # 🆕 调试：显示即将传入策略的观测
    if self.debug and self.step_count % 50 == 0:
      print(f"\n  🧪 Modified Observation to Policy:")
      print(f"     modified_obs keys: {list(modified_obs.keys())}")
      print(f"     Command in modified_obs: {modified_obs['policy'][0, -3:].cpu().numpy()}")
    
    # 调用训练好的策略
    action = self.trained_policy(modified_obs)
    
    # 🆕 调试：显示输出动作
    if self.debug and self.step_count % 50 == 0:
      print(f"\n  🎯 Policy Output:")
      print(f"     action shape: {action.shape}")
      print(f"     action range: [{action.min().item():.4f}, {action.max().item():.4f}]")
      print(f"     action mean: {action.mean().item():.4f}")
      print(f"     action std: {action.std().item():.4f}")
      print(f"{'═'*60}\n")
    
    return action
  
  def __del__(self):
    """析构函数：清理 pygame 资源"""
    if hasattr(self, 'joystick'):
      self.joystick.quit()
    pygame.quit()
    if self.debug:
      print("\n🎮 Joystick disconnected and cleaned up")


def run_play(task: str, cfg: PlayConfig):
  """
  主要函数：初始化环境，加载智能体策略，并运行演示循环。
  
  参数：
    task: 任务名称(如 "Mjlab-HumanoidTask-v0")
    cfg: PlayConfig对象，包含所有配置参数
  """
  # 配置PyTorch后端以获得最佳性能
  configure_torch_backends()

  # 确定使用的计算设备(GPU或CPU)
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  print(f"[INFO]: Using device: {device}")

  # 从注册表加载环境配置
  env_cfg = cast(
    ManagerBasedRlEnvCfg, load_cfg_from_registry(task, "env_cfg_entry_point")
  )
  # 从注册表加载强化学习智能体配置
  agent_cfg = cast(
    RslRlOnPolicyRunnerCfg, load_cfg_from_registry(task, "rl_cfg_entry_point")
  )

  # 判断是否使用虚拟模式(zero或random)
  DUMMY_MODE = cfg.agent in {"zero", "random"}
  # 判断是否使用训练好的模型
  TRAINED_MODE = not DUMMY_MODE
  # 🆕 判断是否使用手柄模式
  JOYSTICK_MODE = cfg.agent == "joystick"

  # 日志目录路径
  log_dir: Optional[Path] = None
  # 恢复/检查点路径
  resume_path: Optional[Path] = None
  
  # 如果使用训练好的模型或手柄模式，处理检查点加载逻辑
  if TRAINED_MODE or JOYSTICK_MODE:
    # 构建日志根目录路径(logs/rsl_rl/实验名称)
    log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
    print(f"[INFO]: Loading experiment from: {log_root_path}")
    
    # 如果指定了检查点文件
    if cfg.checkpoint_file is not None:
      resume_path = Path(cfg.checkpoint_file)
      # 检查文件是否存在
      if not resume_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
    else:
      # 使用训练模式但未指定检查点文件时，抛出错误
      raise ValueError(
        "`checkpoint_file` is required when using trained agent."
      )
    
    print(f"[INFO]: Loading checkpoint: {resume_path}")
    # 设置日志目录为检查点的父目录
    log_dir = resume_path.parent

  # 如果指定了环境数量，覆盖配置中的值
  if cfg.num_envs is not None:
    env_cfg.scene.num_envs = cfg.num_envs
  # 如果指定了视频高度，覆盖配置中的值
  if cfg.video_height is not None:
    env_cfg.viewer.height = cfg.video_height
  # 如果指定了视频宽度，覆盖配置中的值
  if cfg.video_width is not None:
    env_cfg.viewer.width = cfg.video_width

  # 确定渲染模式
  render_mode = "rgb_array" if ((TRAINED_MODE or JOYSTICK_MODE) and cfg.video) else None
  # 虚拟模式下不支持视频录制
  if cfg.video and DUMMY_MODE:
    print(
      "[WARN] Video recording with dummy agents is disabled (no checkpoint/log_dir)."
    )
  
  # 创建Gymnasium环境实例
  env = gym.make(task, cfg=env_cfg, device=device, render_mode=render_mode)

  # 如果需要录制视频，用RecordVideo包装环境
  if (TRAINED_MODE or JOYSTICK_MODE) and cfg.video:
    print("[INFO] Recording videos during play")
    env = gym.wrappers.RecordVideo(
      env,
      video_folder=str(Path(log_dir) / "videos" / "play"),  # type: ignore[arg-type]
      step_trigger=lambda step: step == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )

  # 用RSL-RL向量环境包装器包装环境
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  
  # 根据智能体类型创建相应的策略对象
  if DUMMY_MODE:
    # 获取环境的动作空间维度
    action_shape: tuple[int, ...] = env.unwrapped.action_space.shape  # type: ignore
    
    if cfg.agent == "zero":
      # 零策略：始终返回零动作
      class PolicyZero:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return torch.zeros(action_shape, device=env.unwrapped.device)

      policy = PolicyZero()
    elif cfg.agent == "random":
      class PolicyRandom:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1
      policy = PolicyRandom()
    else:
      raise ValueError(f"Unknown agent type: {cfg.agent}")
  elif JOYSTICK_MODE:
    # ✅ 手柄模式：加载训练好的策略后包装
    print("\n[INFO] Initializing joystick control mode...")
    runner = OnPolicyRunner(
      env, 
      asdict(agent_cfg),
      log_dir=str(log_dir), 
      device=device
    )
    print("[INFO] Loading trained policy...")
    runner.load(str(resume_path), map_location=device)
    trained_policy = runner.get_inference_policy(device=device)
    print("[INFO] Trained policy loaded successfully")
    
    print("[INFO] Creating joystick policy wrapper...")
    policy = PolicyJoystick(
      trained_policy=trained_policy,
      device=env.unwrapped.device,
      joystick_id=cfg.joystick_id,
      deadzone=cfg.joystick_deadzone,
      max_lin_vel=1.5,
      max_ang_vel=1.0,
      debug=cfg.debug_joystick,  # 🆕 使用配置中的调试开关
    )
    print("[INFO] ✅ Joystick policy wrapper created successfully\n")
  else:
    # 如果使用训练好的模型，加载训练好的策略
    print("\n[INFO] Loading trained policy...")
    # 创建OnPolicy训练器运行器实例
    runner = OnPolicyRunner(
      env, 
      asdict(agent_cfg),  # 将配置数据类转换为字典
      log_dir=str(log_dir), 
      device=device
    )
    # 从检查点文件加载训练好的权重
    runner.load(str(resume_path), map_location=device)
    # 提取推理策略(不需要梯度计算)
    policy = runner.get_inference_policy(device=device)
    print("[INFO] ✅ Trained policy loaded successfully\n")

  # 🆕 在运行前添加手柄测试提示
  if JOYSTICK_MODE:
    print("\n" + "="*60)
    print("🎮 JOYSTICK TEST MODE")
    print("="*60)
    print("  Please move the joystick sticks to test:")
    print("  - Left stick: Should control vx (forward/backward) and vy (left/right)")
    print("  - Right stick: Should control vyaw (rotation)")
    print("  ")
    print("  The debug output will show:")
    print("  1. Raw axis values from joystick")
    print("  2. Values after deadzone filtering")
    print("  3. Final velocity commands")
    print("  4. How observations are modified")
    print("  5. Actions generated by the policy")
    print("="*60)
    input("\n  Press ENTER to start... ")
    print("\n")

  # 根据配置选择使用的查看器运行演示
  if cfg.viewer == "native":
    # 使用原生MuJoCo查看器进行交互式可视化
    print("[INFO] Starting Native MuJoCo Viewer...")
    NativeMujocoViewer(env, policy).run()
  elif cfg.viewer == "viser":
    # 使用Viser Web查看器进行可视化
    print("[INFO] Starting Viser Web Viewer...")
    ViserViewer(env, policy).run()
  else:
    # 类型检查：如果viewer值无效，引发错误
    assert_never(cfg.viewer)

  # 关闭环境，释放资源
  env.close()


def main():
  """
  主入口点：解析命令行参数并运行演示。
  
  该函数分两步解析命令行参数：
  1. 第一步：选择任务(以 "Mjlab-" 开头的任务)
  2. 第二步：解析PlayConfig配置参数
  """
  # 任务名称前缀
  task_prefix = "Mjlab-"
  
  # 第一步：解析第一个参数作为任务选择
  # 只显示以 "Mjlab-" 开头的任务供用户选择
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(
      [k for k in gym.registry.keys() if k.startswith(task_prefix)]
    ),
    add_help=False,  # 不显示帮助信息(由主程序处理)
    return_unknown_args=True,  # 返回剩余未解析的参数
  )
  del task_prefix

  # 第二步：解析剩余的命令行参数为PlayConfig对象
  args = tyro.cli(
    PlayConfig,
    args=remaining_args,  # 使用剩余的参数
    default=PlayConfig(),  # 使用PlayConfig的默认值
    prog=sys.argv[0] + f" {chosen_task}",  # 程序帮助信息前缀
    config=(
      tyro.conf.AvoidSubcommands,  # 避免子命令
      tyro.conf.FlagConversionOff,  # 关闭标志转换
    ),
  )
  del remaining_args

  # 运行演示
  run_play(chosen_task, args)


# 脚本入口点
if __name__ == "__main__":
  main()
