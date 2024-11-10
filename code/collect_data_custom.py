"""
    For panda (two-finger) gripper: pushing, pushing-left, pushing-up, pulling, pulling-left, pulling-up
        50% all parts closed, 50% middle (for each part, 50% prob. closed, 50% prob. middle)
        Simulate until static before starting
"""

import os
import sys
import shutil
import numpy as np
from PIL import Image
from utils import get_global_position_from_camera, save_h5
import cv2
import json
from argparse import ArgumentParser

from sapien.core import Pose
from env_custom import Env, ContactError
from camera import Camera
from robots.panda_robot import Robot
# from robots.ur5_robot import Robot
import pyvista as pv
import pcl
import pcl.pcl_visualization


parser = ArgumentParser()
parser.add_argument('category', type=str) # StorageFurniture
parser.add_argument('--out_dir', type=str)
parser.add_argument('--trial_id', type=int, default=0, help='trial id')
parser.add_argument('--random_seed', type=int, default=None)
parser.add_argument('--no_gui', action='store_true', default=False, help='no_gui [default: False]')
args = parser.parse_args()

print("start collect data")
trial_id = args.trial_id
if args.no_gui:
    out_dir = os.path.join(args.out_dir, '%s_%d' % (args.category, trial_id))
else:
    out_dir = os.path.join('results', '%s_%d' % (args.category, trial_id))
if os.path.exists(out_dir):
    shutil.rmtree(out_dir)
print(out_dir)
os.makedirs(out_dir)
flog = open(os.path.join(out_dir, 'log.txt'), 'w')
out_info = dict() # 创建一个空字典

# set random seed
if args.random_seed is not None:
    np.random.seed(args.random_seed)
    out_info['random_seed'] = args.random_seed

# setup env
env = Env(flog=flog, show_gui=(not args.no_gui))

# setup camera
cam = Camera(env, dist=1, random_position=True)
out_info['camera_metadata'] = cam.get_metadata_json()
if not args.no_gui:
    env.set_controller_camera_pose(cam.pos[0], cam.pos[1], cam.pos[2], np.pi+cam.theta, -cam.phi)

# load shape
object_urdf_fn = '../data/grasp/%s/model.urdf' % args.category
flog.write('object_urdf_fn: %s\n' % object_urdf_fn)
object_material = env.get_material(4, 4, 0.01)
state = 'random-closed-middle'
if np.random.random() < 0.5:
    state = 'closed'
flog.write('Object State: %s\n' % state)
out_info['object_state'] = state
joint_angles = env.load_object(object_urdf_fn, object_material, state=state, custom=False)
out_info['joint_angles'] = joint_angles
out_info['joint_angles_lower'] = env.joint_angles_lower
out_info['joint_angles_upper'] = env.joint_angles_upper
cur_qpos = env.get_object_qpos()

# simulate some steps for the object to stay rest
still_timesteps = 0
wait_timesteps = 0
while still_timesteps < 1000:
    env.step(custom=False)
    env.render()
    still_timesteps += 1
'''
while still_timesteps < 5000 and wait_timesteps < 20000:
    env.step()
    env.render()
    cur_new_qpos = env.get_object_qpos()
    invalid_contact = False
    for c in env.scene.get_contacts():
        for p in c.points:
            if abs(p.impulse @ p.impulse) > 1e-4:
                invalid_contact = True
                break
        if invalid_contact:
            break
    if np.max(np.abs(cur_new_qpos - cur_qpos)) < 1e-6 and (not invalid_contact):
        still_timesteps += 1
    else:
        still_timesteps = 0
    cur_qpos = cur_new_qpos
    wait_timesteps += 1

if still_timesteps < 5000:
    flog.write('Object Not Still!\n')
    flog.close()
    env.close()
    exit(1)
'''
### use the GT vision
rgb, depth = cam.get_observation()
Image.fromarray((rgb*255).astype(np.uint8)).save(os.path.join(out_dir, 'rgb.png'))

# 根据深度图（depth）和相机的内参矩阵来计算相机坐标系中的三维点
cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts = cam.compute_camera_XYZA(depth) # 返回有效深度值的像素位置（y, x）和计算出的三维点坐标（points）。
''' show
pv.plot(
    cam_XYZA_pts,
    scalars=cam_XYZA_pts[:, 2],
    render_points_as_spheres=True,
    point_size=5,
    show_scalar_bar=False,
)
'''

cloud = pcl.PointCloud(cam_XYZA_pts.astype(np.float32))
# 创建SAC-IA分割对象
seg = cloud.make_segmenter()
seg.set_optimize_coefficients(True)
seg.set_model_type(pcl.SACMODEL_PLANE)
seg.set_method_type(pcl.SAC_RANSAC)
seg.set_distance_threshold(0.02)
# 执行分割
inliers, coefficients = seg.segment()
# 获取地面点云和非地面点云
ground_points = cloud.extract(inliers, negative=False)
non_ground_points = cloud.extract(inliers, negative=True)
# 转换为array
cam_XYZA_filter_pts = non_ground_points.to_array()
''' show
pv.plot(
    cam_XYZA_filter_pts,
    scalars=cam_XYZA_filter_pts[:, 2],
    render_points_as_spheres=True,
    point_size=5,
    show_scalar_bar=False,
)
'''
# cam_XYZA_pts_tmp = np.around(np.array(cam_XYZA_pts), decimals=6)
# cam_XYZA_filter_pts_tem = np.around(np.array(cam_XYZA_filter_pts), decimals=6)

cam_XYZA_pts_tmp = np.array(cam_XYZA_pts).astype(np.float32)
cam_XYZA_filter_pts_tem = np.array(cam_XYZA_filter_pts).astype(np.float32)

# filter_pc_index = np.array([np.where(cam_XYZA_pts_tmp == a)[0][0] for a in cam_XYZA_filter_pts_tem]).astype(np.uint8)

# match_matrix = (cam_XYZA_pts_tmp[:, np.newaxis, :] == cam_XYZA_filter_pts_tem).all(axis=2)
# filter_pc_index = np.argmax(match_matrix, axis=0)

# match_matrix = np.any(np.abs(cam_XYZA_pts_tmp[:, np.newaxis, :] - cam_XYZA_filter_pts_tem) <= 1e-4, axis=2)
# filter_pc_index = np.argmax(match_matrix, axis=0)

index_inliers_set = set(inliers)
cam_XYZA_filter_idx = []
cam_XYZA_pts_idx = np.arange(cam_XYZA_pts.shape[0])
for idx in range(len(cam_XYZA_pts_idx)):
    if idx not in index_inliers_set:
        cam_XYZA_filter_idx.append(cam_XYZA_pts_idx[idx])
cam_XYZA_filter_idx = np.array(cam_XYZA_filter_idx)
cam_XYZA_filter_idx = cam_XYZA_filter_idx.astype(int)
cam_XYZA_filter_id1 = cam_XYZA_id1[cam_XYZA_filter_idx]
cam_XYZA_filter_id2 = cam_XYZA_id2[cam_XYZA_filter_idx]

# 将计算出的三维点信息组织成一个矩阵格式。
cam_XYZA = cam.compute_XYZA_matrix(cam_XYZA_filter_id1, cam_XYZA_filter_id2, cam_XYZA_filter_pts, depth.shape[0], depth.shape[1])
save_h5(os.path.join(out_dir, 'cam_XYZA.h5'), \
        [(cam_XYZA_filter_id1.astype(np.uint64), 'id1', 'uint64'), \
         (cam_XYZA_filter_id2.astype(np.uint64), 'id2', 'uint64'), \
         (cam_XYZA_filter_pts.astype(np.float32), 'pc', 'float32'), \
        ])

gt_nor = cam.get_normal_map()
Image.fromarray(((gt_nor+1)/2*255).astype(np.uint8)).save(os.path.join(out_dir, 'gt_nor.png'))

object_link_ids = env.movable_link_ids
# gt_movable_link_mask = cam.get_movable_link_mask(object_link_ids) # gt_movable_link_mask 表示为：像素图中可抓取link对应其link_id
# Image.fromarray((gt_movable_link_mask>0).astype(np.uint8)*255).save(os.path.join(out_dir, 'interaction_mask.png')) # 将gt_movable_link_mask转为二值图进行保存

gt_movable_link_mask = cam.get_grasp_regien_mask(cam_XYZA_filter_id1, cam_XYZA_filter_id2, depth.shape[0], depth.shape[1]) # gt_movable_link_mask 表示为：像素图中可抓取link对应其link_id
Image.fromarray((gt_movable_link_mask>0).astype(np.uint8)*255).save(os.path.join(out_dir, 'interaction_mask.png')) # 将gt_movable_link_mask转为二值图进行保存

# sample a pixel to interact
# object_mask = cam.get_object_mask()
xs, ys = np.where(gt_movable_link_mask==1)
if len(xs) == 0:
    flog.write('No Movable Pixel! Quit!\n')
    flog.close()
    env.close()
    exit(1)
idx = np.random.randint(len(xs)) # sample interaction pixels random
x, y = xs[idx], ys[idx]
out_info['pixel_locs'] = [int(x), int(y)] # 采样到的像素位置
# 随机设置一个可移动关节作为 actor_id
env.set_target_object_part_actor_id(object_link_ids[gt_movable_link_mask[x, y]-1], custom=False) # [gt_movable_link_mask[x, y]-1] represent pixel coordinate(x,y) correspond to movable link id
out_info['target_object_part_actor_id'] = env.target_object_part_actor_id
out_info['target_object_part_joint_id'] = env.target_object_part_joint_id

# get pixel 3D pulling direction (cam/world)
direction_cam = gt_nor[x, y, :3]
direction_cam /= np.linalg.norm(direction_cam)
out_info['direction_camera'] = direction_cam.tolist()
flog.write('Direction Camera: %f %f %f\n' % (direction_cam[0], direction_cam[1], direction_cam[2]))
direction_world = cam.get_metadata()['mat44'][:3, :3] @ direction_cam
out_info['direction_world'] = direction_world.tolist()
flog.write('Direction World: %f %f %f\n' % (direction_world[0], direction_world[1], direction_world[2]))
flog.write('mat44: %s\n' % str(cam.get_metadata()['mat44']))

# sample a random direction in the hemisphere (cam/world)
action_direction_cam = np.random.randn(3).astype(np.float32)
action_direction_cam /= np.linalg.norm(action_direction_cam)
if action_direction_cam @ direction_cam > 0: # 两个向量的夹角小于90度
    action_direction_cam = -action_direction_cam
out_info['gripper_direction_camera'] = action_direction_cam.tolist() # position p
action_direction_world = cam.get_metadata()['mat44'][:3, :3] @ action_direction_cam
out_info['gripper_direction_world'] = action_direction_world.tolist()
print("angle between cam to grasp", action_direction_cam @ direction_cam)

# get pixel 3D position (cam/world)
position_cam = cam_XYZA[x, y, :3]
if (np.sum(position_cam) == 0):
    print("position_cam : ", position_cam)
    sys.exit
out_info['position_cam'] = position_cam.tolist()
position_cam_xyz1 = np.ones((4), dtype=np.float32)
position_cam_xyz1[:3] = position_cam
position_world_xyz1 = cam.get_metadata()['mat44'] @ position_cam_xyz1
position_world = position_world_xyz1[:3]
out_info['position_world'] = position_world.tolist()

# compute final pose
up = np.array(action_direction_world, dtype=np.float32)
forward = np.random.randn(3).astype(np.float32)
while abs(up @ forward) > 0.99:
    forward = np.random.randn(3).astype(np.float32)
left = np.cross(up, forward)
left /= np.linalg.norm(left)
forward = np.cross(left, up)
forward /= np.linalg.norm(forward)
out_info['gripper_forward_direction_world'] = forward.tolist()
forward_cam = np.linalg.inv(cam.get_metadata()['mat44'][:3, :3]) @ forward
out_info['gripper_forward_direction_camera'] = forward_cam.tolist() # orientation
rotmat = np.eye(4).astype(np.float32) # 旋转矩阵
rotmat[:3, 0] = forward
rotmat[:3, 1] = left
rotmat[:3, 2] = up

# final_dist = 0.13 # ur5 grasp
final_dist = 0.1

final_rotmat = np.array(rotmat, dtype=np.float32)
final_rotmat[:3, 3] = position_world - action_direction_world * final_dist # 以齐次坐标形式添加 平移向量
final_pose = Pose().from_transformation_matrix(final_rotmat) # 变换矩阵转位置和旋转（四元数）
out_info['target_rotmat_world'] = final_rotmat.tolist()

start_rotmat = np.array(rotmat, dtype=np.float32)
# start_rotmat[:3, 3] = position_world - action_direction_world * 0.2 # 以齐次坐标形式添加 平移向量  ur5 grasp
start_rotmat[:3, 3] = position_world - action_direction_world * 0.15 # 以齐次坐标形式添加 平移向量
start_pose = Pose().from_transformation_matrix(start_rotmat) # 变换矩阵转位置和旋转（四元数）
out_info['start_rotmat_world'] = start_rotmat.tolist()

action_direction = None

if action_direction is not None:
    end_rotmat = np.array(rotmat, dtype=np.float32)
    end_rotmat[:3, 3] = position_world - action_direction_world * final_dist + action_direction * 0.05
    out_info['end_rotmat_world'] = end_rotmat.tolist()


### viz the EE gripper position
# setup robot
robot_urdf_fn = './robots/panda_gripper.urdf'
# robot_urdf_fn = './robots/robotiq85/robots/robotiq_arg85_description.urdf'
# robot_urdf_fn = './robots/Robotiq85/urdf/robotiq_85.urdf'
# robot_urdf_fn = './robots/robotiq_85/urdf/robotiq_85_gripper_simple.urdf'
# robot_urdf_fn = './robots/ur5_description/urdf/ur5_robotiq_85.urdf'
robot_material = env.get_material(4, 4, 0.01)
robot = Robot(env, robot_urdf_fn, robot_material)

# move to the final pose
robot.robot.set_root_pose(final_pose)
env.render()
rgb_final_pose, _ = cam.get_observation()
Image.fromarray((rgb_final_pose*255).astype(np.uint8)).save(os.path.join(out_dir, 'viz_target_pose.png'))

# move back
robot.robot.set_root_pose(start_pose)
env.render()
# activate contact checking
env.start_checking_contact(robot.hand_actor_id, robot.gripper_actor_ids)

if not args.no_gui:
    ### wait to start
    env.wait_to_start()

### main steps
out_info['start_target_part_qpos'] = env.get_target_part_qpos(custom=False)

target_link_mat44 = env.get_target_part_pose(custom=False).to_transformation_matrix() # 得到世界坐标系下物体Link的位姿
# 某一点云坐标相对于物体Link坐标发生的位姿变换, 该点云坐标不随Link运动而变化          inv: 从物体Link坐标系变换回全局坐标系的逆变换，表示将点从世界坐标系变换到物体Link坐标系
position_local_xyz1 = np.linalg.inv(target_link_mat44) @ position_world_xyz1 # position_world_xyz1: 世界坐标系下物体上某一点云坐标

success = True
try:
    robot.open_gripper()

    # approach
    robot.move_to_target_pose(final_rotmat, 500, custom=False) # final_rotmat 齐次坐标形式的位姿矩阵4✖4
    print("move to start pose end")
    robot.wait_n_steps(500, custom=False)
    env.end_checking_contact(custom=False)

    #### 计算位置变化 ####
    target_link_pose = env.get_target_part_pose(custom=False).p # 得到世界坐标系下物体Link的位姿
    # position_world_xyz1_start = target_link_mat44 @ position_local_xyz1 # position_world_xyz1_end: 某一点云坐标相对于世界坐标系下发生的位姿变换
    mov_dir = np.array(target_link_pose[:2].tolist(), dtype=np.float32) - \
            np.array([0,0], dtype=np.float32)
    mov_dir = np.linalg.norm(mov_dir, ord=2)
    print("mov_dir", mov_dir)
    if mov_dir > 0.01:
        success = False
        print("move start contact: ", mov_dir)
        raise ContactError

    robot.close_gripper()
    robot.wait_n_steps(500, custom=False)

    # activate contact checking
    print("move end")
    env.start_checking_contact(robot.hand_actor_id, robot.gripper_actor_ids, custom=False)
    robot.move_to_target_pose(start_rotmat, 500, custom=False)
    print("move to end pose end")
    env.end_checking_contact(custom=False)
    robot.wait_n_steps(500, custom=False)
    print("move finish")
    
except ContactError:
    success = False

target_link_mat44 = env.get_target_part_pose(custom=False).to_transformation_matrix() # 得到世界坐标系下物体Link的位姿
position_world_xyz1_end = target_link_mat44 @ position_local_xyz1 # position_world_xyz1_end: 某一点云坐标相对于世界坐标系下发生的位姿变换
flog.write('touch_position_world_xyz_start: %s\n' % str(position_world_xyz1))
flog.write('touch_position_world_xyz_end: %s\n' % str(position_world_xyz1_end))
out_info['touch_position_world_xyz_start'] = position_world_xyz1[:3].tolist()
out_info['touch_position_world_xyz_end'] = position_world_xyz1_end[:3].tolist()

#### 计算位置变化 ####
# mov_dir = np.array(out_info['touch_position_world_xyz_end'], dtype=np.float32) - \
#         np.array(out_info['touch_position_world_xyz_start'], dtype=np.float32)
# mov_dir /= np.linalg.norm(mov_dir)
# intended_dir = -np.array(out_info['gripper_direction_world'], dtype=np.float32)
# print("move end distance: ", intended_dir @ mov_dir)
# success = (intended_dir @ mov_dir > 0.01) and (intended_dir @ mov_dir < 0.5) 

if success:
    out_info['result'] = 'VALID'
    out_info['final_target_part_qpos'] = env.get_target_part_qpos(custom=False)
else:
    out_info['result'] = 'CONTACT_ERROR'

# save results
with open(os.path.join(out_dir, 'result.json'), 'w') as fout:
    json.dump(out_info, fout)

#close the file
flog.close()

if args.no_gui:
    # close env
    env.close()
else:
    if success:
        print('[Successful Interaction] Done. Ctrl-C to quit.')
        ### wait forever
        robot.wait_n_steps(100000)
    else:
        print('[Unsuccessful Interaction] invalid gripper-object contact.')
        # close env
        env.close()

