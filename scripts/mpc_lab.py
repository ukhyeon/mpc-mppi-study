"""Franka A->B with cuRobo MPPI MPC in Isaac Sim (no camera, no nvblox).

Primitive collision world: table + back wall + one cuboid obstacle between A and B.
Simplified version of franka-depth-mppi/experiment/hand_experiment.py for study.

Run (from ~/isaacsim):
  PYTHONPATH=$HOME/curobo/src ./python.sh study_franka_mppi/scripts/mpc_lab.py --headless \
      --eta 0.025 --out study_franka_mppi/scripts/results/eta025.jsonl
"""
# ---------------------------------------------------------------------------
# 1. 인자 파싱 -> SimulationApp 생성 (omni/isaacsim 모듈은 반드시 이 뒤에 import)
# ---------------------------------------------------------------------------
import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--headless', action='store_true')
parser.add_argument('--steps', type=int, default=600, help='physics steps (1/60 s each)')
parser.add_argument('--eta', type=float, default=None, help='environment collision activation distance [m]; default 0.025 from particle_mpc.yml')
parser.add_argument('--buffer', type=float, default=0.02, help='collision_sphere_buffer [m] (hand_experiment uses 0.02)')
parser.add_argument('--particles', type=int, default=None, help='MPPI num_particles (default 400)')
parser.add_argument('--beta', type=float, default=None, help='MPPI beta (default 0.1)')
parser.add_argument('--step-dt', type=float, default=1 / 60, help='MPC model base dt (hand replay used 1/60)')
parser.add_argument('--obstacle', type=float, nargs=6, default=[0.5, 0.0, 0.4, 0.12, 0.06, 0.16],
                    metavar=('X', 'Y', 'Z', 'DX', 'DY', 'DZ'), help='obstacle cuboid center and size [m]')
parser.add_argument('--no-obstacle', action='store_true')
parser.add_argument('--out', default=None, help='JSONL output path')
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

from isaacsim import SimulationApp

app = SimulationApp({'headless': args.headless, 'width': 1280, 'height': 720})

from isaacsim.core.api import World
from isaacsim.core.api.objects import VisualCuboid
from isaacsim.core.utils.types import ArticulationAction

from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import Cuboid, WorldConfig
from curobo.rollout.rollout_base import Goal
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.state import JointState
from curobo.util.logger import setup_curobo_logger
from curobo.util.usd_helper import UsdHelper
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
from curobo.wrap.reacher.mpc import MpcSolver, MpcSolverConfig

sys.path.insert(0, str(Path.home() / 'curobo' / 'examples' / 'isaac_sim'))
from helper import add_robot_to_scene  # noqa: E402

setup_curobo_logger('warn')
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# hand_experiment.py 와 같은 목표 (x, y, z, qw, qx, qy, qz)
GOALS = {'A': [0.4, -0.5, 0.4, 0, 1, 0, 0], 'B': [0.4, 0.5, 0.4, 0, 1, 0, 0]}
PLAN = ['A', 'B']                 # start -> A -> B, B 도달 후 유지
POS_TOL, ROT_TOL = 0.01, 0.05     # 실제 EE 도달 판정 (10 mm, 0.05 rad)
PHYSICS_DT = 1 / 60

# ---------------------------------------------------------------------------
# 2. Isaac Sim 장면: 지면, Franka, 목표 표시, 장애물 표시
# ---------------------------------------------------------------------------
ta = TensorDeviceType()
world = World(stage_units_in_meters=1.0)
world.set_simulation_dt(physics_dt=PHYSICS_DT, rendering_dt=PHYSICS_DT)
stage = world.stage
stage.SetDefaultPrim(stage.DefinePrim('/World', 'Xform'))
world.scene.add_default_ground_plane()

robot_cfg = load_yaml(join_path(get_robot_configs_path(), 'franka.yml'))['robot_cfg']
robot_cfg['kinematics']['collision_sphere_buffer'] = args.buffer
robot = add_robot_to_scene(robot_cfg, world)[0]  # 로컬 helper 는 (robot, path, view) 를 반환
joint_names = robot_cfg['kinematics']['cspace']['joint_names']
q0 = robot_cfg['kinematics']['cspace']['retract_config']

for name, g in GOALS.items():
    VisualCuboid(f'/World/goal_{name}', position=np.array(g[:3]), orientation=np.array(g[3:]),
                 size=0.04, color=np.array([0.2, 0.8, 0.2]))

# ---------------------------------------------------------------------------
# 3. cuRobo 충돌 세계 (PRIMITIVE = cuboid OBB 만, nvblox 없음)
# ---------------------------------------------------------------------------
fixed = WorldConfig.from_dict(load_yaml(join_path(get_world_configs_path(), 'collision_wall.yml')))
fixed.cuboid[0].pose[2] -= 0.01   # 테이블 윗면을 지면보다 1 cm 아래로 (hand_experiment.py 와 동일)
cuboids = list(fixed.cuboid[:2])  # table, back wall
obs = None
if not args.no_obstacle:
    x, y, z, dx, dy, dz = args.obstacle
    obs = Cuboid(name='obstacle', pose=[x, y, z, 1, 0, 0, 0], dims=[dx, dy, dz])
    cuboids.append(obs)
    VisualCuboid('/World/obstacle_vis', position=np.array([x, y, z]), scale=np.array([dx, dy, dz]),
                 size=1.0, color=np.array([1.0, 0.6, 0.1]))
world_cfg = WorldConfig(cuboid=cuboids)
usd = UsdHelper()
usd.load_stage(stage)
usd.add_world_to_stage(WorldConfig(cuboid=list(fixed.cuboid[:2])).get_mesh_world(), base_frame='/World')

# ---------------------------------------------------------------------------
# 4. MPC solver (MPPI only). 모든 설정은 CUDA graph capture(첫 mpc.step) 전에 끝낸다.
# ---------------------------------------------------------------------------
override = {'mppi': {}}
if args.particles is not None:
    override['mppi']['num_particles'] = args.particles
if args.beta is not None:
    override['mppi']['beta'] = args.beta
override_file = None
if override['mppi']:
    import yaml
    override_file = os.path.join(tempfile.mkdtemp(), 'mppi_override.yml')
    Path(override_file).write_text(yaml.safe_dump(override))

mpc_cfg = MpcSolverConfig.load_from_robot_config(
    robot_cfg, world_cfg,
    use_cuda_graph=True, use_cuda_graph_metrics=True, use_cuda_graph_full_step=False,
    self_collision_check=True,
    collision_checker_type=CollisionCheckerType.PRIMITIVE,
    collision_cache={'obb': 10},
    collision_activation_distance=args.eta,   # None 이면 particle_mpc.yml 의 0.025
    use_mppi=True, use_lbfgs=False, use_es=False,
    store_rollouts=True,
    step_dt=args.step_dt,
    override_particle_file=override_file,
)
mpc = MpcSolver(mpc_cfg)

opt = mpc.solver.optimizers[0]
settings = {
    'optimizer': type(opt).__name__,
    'num_particles': opt.total_num_particles, 'horizon': opt.horizon,
    'n_iters': opt.n_iters, 'beta': opt.beta, 'gamma': opt.gamma,
    'activation_distance_m': float(opt.rollout_fn.primitive_collision_cost.activation_distance[0]),
    'collision_weight': float(opt.rollout_fn.primitive_collision_cost.weight[0]),
    'collision_sphere_buffer_m': args.buffer,
    'traj_dt': mpc.rollout_fn.dynamics_model.traj_dt.cpu().tolist(),
    'horizon_s': float(mpc.rollout_fn.dynamics_model.traj_dt.sum()),
    'physics_dt': PHYSICS_DT, 'obstacle': None if obs is None else {'pose': obs.pose, 'dims': obs.dims},
}
print('MPC_SETTINGS', json.dumps({k: v for k, v in settings.items() if k != 'traj_dt'}), flush=True)

retract = mpc.rollout_fn.dynamics_model.retract_config.clone().unsqueeze(0)
js = JointState.from_position(retract, joint_names=mpc.joint_names)
goal = Goal(current_state=js, goal_state=js, goal_pose=Pose.from_list(GOALS[PLAN[0]], ta))
goal_buffer = mpc.setup_solve_single(goal, 1)
mpc.update_goal(goal_buffer)


def obstacle_clearance(spheres):
    """구 표면 - 축정렬 cuboid 최소 거리 [m] (반지름에 buffer 포함). 음수 = 겹침."""
    if obs is None:
        return None
    c = np.array(obs.pose[:3])
    h = np.array(obs.dims) / 2
    q = np.abs(spheres[:, :3] - c) - h
    outside = np.linalg.norm(np.maximum(q, 0), axis=1)
    inside = np.minimum(q.max(axis=1), 0)
    return float((outside + inside - spheres[:, 3]).min())


# ---------------------------------------------------------------------------
# 5. 제어 루프: physics step -> 측정 상태 -> mpc.step -> 위치 명령
# ---------------------------------------------------------------------------
world.reset()
world.play()
idx = [robot.get_dof_index(n) for n in joint_names]
robot.set_joint_positions(q0, idx)
robot._articulation_view.set_max_efforts(np.full(len(idx), 5000), joint_indices=idx)
controller = robot.get_articulation_controller()
cmd_idx = [robot.get_dof_index(n) for n in mpc.joint_names]

out = open(args.out, 'w') if args.out else None
if out:
    out.write(json.dumps({'type': 'settings', **settings}) + '\n')
plan_i = 0
events = [{'event': 'command', 'goal': PLAN[0], 'step': 0, 'sim_s': 0.0}]
t0 = world.current_time

for step in range(1, args.steps + 1):
    if not app.is_running():
        break
    world.step(render=not args.headless)
    now = world.current_time - t0

    # 측정 관절 상태 (위치 + 속도), cuRobo 관절 순서로 정렬
    s = robot.get_joints_state()
    measured = JointState(position=ta.to_device(s.positions), velocity=ta.to_device(s.velocities),
                          acceleration=ta.to_device(s.velocities) * 0, jerk=ta.to_device(s.velocities) * 0,
                          joint_names=robot.dof_names).get_ordered_joint_state(mpc.joint_names).unsqueeze(0)

    # 실제 EE 오차 -> 도달하면 다음 목표
    kin = mpc.rollout_fn.compute_kinematics(measured)
    ee = kin.ee_pos_seq.reshape(-1, 3)[0]
    quat = kin.ee_quat_seq.reshape(-1, 4)[0]
    gp = goal_buffer.goal_pose.position.reshape(-1, 3)[0]
    gq = goal_buffer.goal_pose.quaternion.reshape(-1, 4)[0]
    pos_err = float(torch.linalg.norm(ee - gp))
    rot_err = float(2 * torch.acos(torch.dot(quat, gq).abs().clamp(0, 1)))
    if pos_err <= POS_TOL and rot_err <= ROT_TOL and not any(e['event'] == 'arrive' and e['goal'] == PLAN[plan_i] for e in events):
        events.append({'event': 'arrive', 'goal': PLAN[plan_i], 'step': step, 'sim_s': now,
                       'pos_err_m': pos_err, 'rot_err_rad': rot_err})
        print('ARRIVE', json.dumps(events[-1]), flush=True)
        if plan_i + 1 < len(PLAN):
            plan_i += 1
            goal_buffer.goal_pose.copy_(Pose.from_list(GOALS[PLAN[plan_i]], ta))
            mpc.update_goal(goal_buffer)
            events.append({'event': 'command', 'goal': PLAN[plan_i], 'step': step, 'sim_s': now})

    # MPPI 최적화 1회 (내부 n_iters 반복, warm start)
    torch.cuda.synchronize()
    tic = time.perf_counter()
    result = mpc.step(measured, max_attempts=2)
    torch.cuda.synchronize()
    solver_s = time.perf_counter() - tic
    cmd = result.js_action.get_ordered_joint_state(mpc.joint_names)
    feasible = bool(result.metrics.feasible.all())

    # 위치 명령 -> Isaac Sim 관절 PD 드라이브가 추종
    controller.apply_action(ArticulationAction(cmd.position.reshape(-1).cpu().numpy(), joint_indices=cmd_idx))

    spheres = kin.robot_spheres.reshape(-1, 4)
    spheres = spheres[spheres[:, 3] > 0].cpu().numpy()
    clr = obstacle_clearance(spheres)
    row = {'type': 'step', 'step': step, 'sim_s': now, 'goal': PLAN[plan_i],
           'ee_pos': ee.cpu().tolist(), 'pos_err_m': pos_err, 'rot_err_rad': rot_err,
           'q': measured.position.reshape(-1).cpu().tolist(),
           'clearance_buffered_m': clr, 'clearance_bare_m': None if clr is None else clr + args.buffer,
           'feasible': feasible, 'solver_wall_s': solver_s}
    if out:
        out.write(json.dumps(row) + '\n')
    if step % 60 == 0:
        print('STATUS', json.dumps({k: row[k] for k in ['step', 'sim_s', 'goal', 'pos_err_m', 'clearance_buffered_m', 'feasible']}), flush=True)

summary = {'type': 'summary', 'events': events, 'steps': step,
           'reached': [e['goal'] for e in events if e['event'] == 'arrive']}
arr = {e['goal']: e['sim_s'] for e in events if e['event'] == 'arrive'}
if 'A' in arr and 'B' in arr:
    summary['AB_duration_s'] = arr['B'] - arr['A']
print('SUMMARY', json.dumps(summary), flush=True)
if out:
    out.write(json.dumps(summary) + '\n')
    out.close()
app.close()
