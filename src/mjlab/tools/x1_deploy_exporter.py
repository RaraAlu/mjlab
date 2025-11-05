#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
X1/MJLab 部署参数导出器 - 简化版
仅导出 YAML 格式，包含：刚度 (stiffness)、阻尼 (damping)、缩放系数 (action_scale)

用法示例：
  python x1_deploy_exporter_yaml.py --out x1_joint_params.yaml
  python x1_deploy_exporter_yaml.py --out params.yaml --preview
"""

from __future__ import annotations
import argparse
import importlib
import math
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import mujoco
import yaml
from mjlab.entity.entity import Entity
from mjlab.utils.spec_config import ActuatorCfg


@dataclass
class JointData:
  """关节数据 - 仅包含需要的三个字段"""
  joint: str
  stiffness: float
  damping: float
  action_scale: float


# ============ 工具函数 ============

def _close(a: float, b: float, rel: float = 1e-6, abs_tol: float = 1e-9) -> bool:
  return abs(a - b) <= max(abs(a), abs(b)) * rel + abs_tol


def _collect_electric_actuators(mod) -> Dict[str, object]:
  """收集模块中所有 ElectricActuator 实例"""
  try:
    from mjlab.utils.actuator import ElectricActuator
  except Exception:
    ElectricActuator = None
  out = {}
  for name, obj in vars(mod).items():
    if ElectricActuator and isinstance(obj, ElectricActuator):
      out[name] = obj
  return out


def _label_from_name(var_name: str) -> str:
  """从变量名提取标签"""
  m = re.search(r"R(\d+)_?(\d+)?", var_name.upper())
  if m:
    a = m.group(1)
    b = m.group(2)
    return f"R{a}-{b}" if b else f"R{a}"
  if var_name.upper().startswith("ACTUATOR_"):
    return var_name[len("ACTUATOR_"):]
  if var_name.upper().startswith("X1_ACTUATOR_"):
    return var_name[len("X1_ACTUATOR_"):]
  return var_name


def _match_group_velocity_limit(a_cfg: ActuatorCfg, elec_map: Dict[str, object]) -> Tuple[Optional[str], Optional[float]]:
  """通过数值匹配 ElectricActuator 返回 velocity_limit"""
  effort = float(a_cfg.effort_limit)
  arm = float(getattr(a_cfg, "armature", 0.0))
  for var_name, ea in elec_map.items():
    e_eff = float(getattr(ea, "effort_limit", float("nan")))
    e_arm = float(getattr(ea, "reflected_inertia", float("nan")))
    if _close(effort, e_eff) and _close(arm, e_arm):
      return var_name, None
  return None, None


def _joint_type_name(m: mujoco.MjModel, j: int) -> str:
  """获取关节类型名称"""
  t = int(m.jnt_type[j])
  return {0: "free", 1: "ball", 2: "slide", 3: "hinge"}.get(t, str(t))


def _expand_groups(mod, soft_factor: float) -> List[JointData]:
  """从 x1_constants 展开所有关节数据"""
  # 1) 构建模型
  robot_cfg = getattr(mod, "X1_ROBOT_CFG", None)
  if robot_cfg is None:
    raise RuntimeError("未找到 X1_ROBOT_CFG")
  ent = Entity(robot_cfg)
  m = ent.spec.compile()

  # 2) 读取组配置
  articulation = getattr(mod, "X1_ARTICULATION", None)
  if articulation is None or not hasattr(articulation, "actuators"):
    raise RuntimeError("未找到 X1_ARTICULATION.actuators")
  a_cfgs: Sequence[ActuatorCfg] = articulation.actuators

  # 3) 构建组信息
  groups_info = []
  elec_map = _collect_electric_actuators(mod)
  
  for idx, a in enumerate(a_cfgs):
    var_name, _ = _match_group_velocity_limit(a, elec_map)
    label = _label_from_name(var_name) if var_name else f"group-{idx}"
    groups_info.append({
        "index": idx,
        "label": label,
        "stiffness": float(a.stiffness),
        "damping": float(a.damping),
        "effort_limit": float(a.effort_limit),
        "regex": list(a.joint_names_expr),
    })

  # 4) 获取所有关节
  jnames: List[str] = []
  for j in range(m.njnt):
    nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
    jnames.append(nm)

  # 5) 正则匹配
  compiled = [[re.compile(p) for p in g["regex"]] for g in groups_info]
  joint_hit: Dict[str, List[int]] = {jn: [] for jn in jnames}
  for gi, plist in enumerate(compiled):
    for jn in jnames:
      if any(p.fullmatch(jn) or p.match(jn) for p in plist):
        joint_hit[jn].append(gi)

  conflicts = [jn for jn, hits in joint_hit.items() if len(hits) > 1]
  misses = [jn for jn, hits in joint_hit.items() if len(hits) == 0]
  if conflicts:
    raise RuntimeError(f"正则冲突：{conflicts[:10]}")
  if misses:
    print(f"[WARN] {len(misses)} 个关节未匹配 (示例: {misses[:5]})")

  # 6) 生成关节数据
  rows: List[JointData] = []
  for j in range(m.njnt):
    jn = jnames[j]
    hits = joint_hit.get(jn, [])
    if len(hits) != 1:
      continue

    gi = hits[0]
    g = groups_info[gi]
    jtype = _joint_type_name(m, j)

    # 计算缩放系数（与 x1_constants.py 一致）
    if g["stiffness"] <= 0.0:
      scale = float("inf")
    else:
      scale = 0.25 * g["effort_limit"] / g["stiffness"]

    rows.append(JointData(
        joint=jn,
        stiffness=g["stiffness"],
        damping=g["damping"],
        action_scale=scale,
    ))

  return rows


# ============ 导出函数 ============

def export_yaml(path: str, rows: List[JointData]):
  """导出为 YAML 格式（仅包含刚度、阻尼、缩放系数）"""
  # 构建字典：关节名 -> {stiffness, damping, action_scale}
  data = {}
  for r in rows:
    # 格式化 action_scale：如果是 inf 则输出 "inf"，否则保留足够精度
    scale_val = "inf" if math.isinf(r.action_scale) else round(r.action_scale, 10)
    
    data[r.joint] = {
        "stiffness": round(r.stiffness, 6),
        "damping": round(r.damping, 6),
        "action_scale": scale_val,
    }
  
  # 使用 PyYAML 导出，保持字典顺序
  with open(path, "w", encoding="utf-8") as f:
    yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
  
  print(f"[OK] 写出 YAML: {path} ({len(rows)} 个关节)")


# ============ CLI ============

def main():
  ap = argparse.ArgumentParser(description="X1/MJLab 部署参数导出器 - YAML 简化版")
  ap.add_argument("--robot-mod", type=str,
                  default="mjlab.asset_zoo.robots.agibot_x1.x1_constants",
                  help="配置模块名")
  ap.add_argument("--out", type=str, default="x1_joint_params.yaml",
                  help="输出 YAML 文件路径")
  ap.add_argument("--preview", action="store_true",
                  help="打印前 20 条关节摘要")
  args = ap.parse_args()

  try:
    mod = importlib.import_module(args.robot_mod)
  except Exception as e:
    print(f"[FATAL] 无法 import 模块 {args.robot_mod}: {e}")
    sys.exit(2)

  soft_factor = float(getattr(
    getattr(mod, "X1_ARTICULATION", None), "soft_joint_pos_limit_factor", 0.9))
  
  rows = _expand_groups(mod, soft_factor)

  if args.preview:
    print("[PREVIEW] 前 20 个关节：")
    for r in rows[:20]:
      scale_txt = f"{r.action_scale:.5f}" if math.isfinite(r.action_scale) else "inf"
      print(f"  - {r.joint:>28s} | Kp={r.stiffness:.2f} Kd={r.damping:.2f} | scale={scale_txt}")

  export_yaml(args.out, rows)


if __name__ == "__main__":
  main()