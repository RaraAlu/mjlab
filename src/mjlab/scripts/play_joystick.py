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


@dataclass(frozen=True)
class PlayConfig:
  """
  播放/演示配置数据类，定义了运行智能体时的所有参数。
  frozen=True 表示这个数据类是不可变的。
  """
  # 智能体类型：zero(零动作), random(随机动作), trained(训练好的模型)
  agent: Literal["zero", "random", "trained"] = "trained"
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

  # 日志目录路径
  log_dir: Optional[Path] = None
  # 恢复/检查点路径
  resume_path: Optional[Path] = None
  
  # 如果使用训练好的模型，处理检查点加载逻辑
  if TRAINED_MODE:
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

  # 确定渲染模式：
  # - 训练模式且需要录制视频时，使用 "rgb_array"(用于视频保存)
  # - 否则为 None(不进行额外渲染)
  render_mode = "rgb_array" if (TRAINED_MODE and cfg.video) else None
  # 虚拟模式下不支持视频录制(因为没有真实的智能体)
  if cfg.video and DUMMY_MODE:
    print(
      "[WARN] Video recording with dummy agents is disabled (no checkpoint/log_dir)."
    )
  
  # 创建Gymnasium环境实例
  env = gym.make(task, cfg=env_cfg, device=device, render_mode=render_mode)

  # 如果需要录制视频，用RecordVideo包装环境
  if TRAINED_MODE and cfg.video:
    print("[INFO] Recording videos during play")
    env = gym.wrappers.RecordVideo(
      env,
      # 视频保存位置：logs/rsl_rl/实验名称/videos/play/
      video_folder=str(Path(log_dir) / "videos" / "play"),  # type: ignore[arg-type]
      # 在第0步时触发录制
      step_trigger=lambda step: step == 0,
      # 每段视频的长度(步数)
      video_length=cfg.video_length,
      # 禁用日志记录器
      disable_logger=True,
    )

  # 用RSL-RL向量环境包装器包装环境，用于处理向量化操作和动作裁剪
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  
  # 根据智能体类型创建相应的策略对象
  if DUMMY_MODE:
    # 获取环境的动作空间维度
    action_shape: tuple[int, ...] = env.unwrapped.action_space.shape  # type: ignore
    
    if cfg.agent == "zero":
      # 零策略：始终返回零动作
      class PolicyZero:
        def __call__(self, obs) -> torch.Tensor:
          # 忽略观测值，返回全零张量
          del obs
          return torch.zeros(action_shape, device=env.unwrapped.device)

      policy = PolicyZero()
    else:
      # 随机策略：返回[-1, 1]范围内的随机动作
      class PolicyRandom:
        def __call__(self, obs) -> torch.Tensor:
          # 忽略观测值，返回随机动作张量
          del obs
          # 生成[-1, 1]范围的均匀分布随机数
          return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1

      policy = PolicyRandom()
  else:
    # 如果使用训练好的模型，加载训练好的策略
    
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

  # 根据配置选择使用的查看器运行演示
  if cfg.viewer == "native":
    # 使用原生MuJoCo查看器进行交互式可视化
    NativeMujocoViewer(env, policy).run()
  elif cfg.viewer == "viser":
    # 使用Viser Web查看器进行可视化
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