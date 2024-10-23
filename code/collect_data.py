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
from env import Env, ContactError
from camera import Camera
from robots.panda_robot import Robot

parser = ArgumentParser()
parser.add_argument('shape_id', type=str) # 40147
parser.add_argument('category', type=str) # StorageFurniture
parser.add_argument('cnt_id', type=int) # 0
parser.add_argument('primact_type', type=str) # pushing
parser.add_argument('--out_dir', type=str)
parser.add_argument('--trial_id', type=int, default=0, help='trial id')
parser.add_argument('--random_seed', type=int, default=None)
parser.add_argument('--no_gui', action='store_true', default=False, help='no_gui [default: False]')
args = parser.parse_args()

shape_id = args.shape_id
trial_id = args.trial_id
primact_type = args.primact_type
if args.no_gui:
    out_dir = os.path.join(args.out_dir, '%s_%s_%d_%s_%d' % (shape_id, args.category, args.cnt_id, primact_type, trial_id))
else:
    out_dir = os.path.join('results', '%s_%s_%d_%s_%d' % (shape_id, args.category, args.cnt_id, primact_type, trial_id))
if os.path.exists(out_dir):
    shutil.rmtree(out_dir)
os.mkdir(out_dir)
flog = open(os.path.join(out_dir, 'log.txt'), 'w')
out_info = dict() # 创建一个空字典

# set random seed
if args.random_seed is not None:
    np.random.seed(args.random_seed)
    out_info['random_seed'] = args.random_seed

# setup env
env = Env(flog=flog, show_gui=(not args.no_gui))

# setup camera
cam = Camera(env, random_position=True)
out_info['camera_metadata'] = cam.get_metadata_json()
if not args.no_gui:
    env.set_controller_camera_pose(cam.pos[0], cam.pos[1], cam.pos[2], np.pi+cam.theta, -cam.phi)

# load shape
object_urdf_fn = '../data/where2act_original_sapien_dataset/%s/mobility_vhacd.urdf' % shape_id
flog.write('object_urdf_fn: %s\n' % object_urdf_fn)
object_material = env.get_material(4, 4, 0.01)
state = 'random-closed-middle'
if np.random.random() < 0.5:
    state = 'closed'
flog.write('Object State: %s\n' % state)
out_info['object_state'] = state
joint_angles = env.load_object(object_urdf_fn, object_material, state=state)
out_info['joint_angles'] = joint_angles
out_info['joint_angles_lower'] = env.joint_angles_lower
out_info['joint_angles_upper'] = env.joint_angles_upper
cur_qpos = env.get_object_qpos()

# simulate some steps for the object to stay rest
still_timesteps = 0
wait_timesteps = 0
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

### use the GT vision
rgb, depth = cam.get_observation()
Image.fromarray((rgb*255).astype(np.uint8)).save(os.path.join(out_dir, 'rgb.png'))

# 根据深度图（depth）和相机的内参矩阵来计算相机坐标系中的三维点
cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts = cam.compute_camera_XYZA(depth) # 返回有效深度值的像素位置（y, x）和计算出的三维点坐标（points）。
# 将计算出的三维点信息组织成一个矩阵格式。
cam_XYZA = cam.compute_XYZA_matrix(cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts, depth.shape[0], depth.shape[1])
save_h5(os.path.join(out_dir, 'cam_XYZA.h5'), \
        [(cam_XYZA_id1.astype(np.uint64), 'id1', 'uint64'), \
         (cam_XYZA_id2.astype(np.uint64), 'id2', 'uint64'), \
         (cam_XYZA_pts.astype(np.float32), 'pc', 'float32'), \
        ])

gt_nor = cam.get_normal_map()
Image.fromarray(((gt_nor+1)/2*255).astype(np.uint8)).save(os.path.join(out_dir, 'gt_nor.png'))

object_link_ids = env.movable_link_ids
gt_movable_link_mask = cam.get_movable_link_mask(object_link_ids)
Image.fromarray((gt_movable_link_mask>0).astype(np.uint8)*255).save(os.path.join(out_dir, 'interaction_mask.png'))

# sample a pixel to interact
xs, ys = np.where(gt_movable_link_mask>0)
if len(xs) == 0:
    flog.write('No Movable Pixel! Quit!\n')
    flog.close()
    env.close()
    exit(1)
idx = np.random.randint(len(xs))
x, y = xs[idx], ys[idx]
out_info['pixel_locs'] = [int(x), int(y)]
env.set_target_object_part_actor_id(object_link_ids[gt_movable_link_mask[x, y]-1])
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
if action_direction_cam @ direction_cam > 0:
    action_direction_cam = -action_direction_cam
out_info['gripper_direction_camera'] = action_direction_cam.tolist()
action_direction_world = cam.get_metadata()['mat44'][:3, :3] @ action_direction_cam
out_info['gripper_direction_world'] = action_direction_world.tolist()

# get pixel 3D position (cam/world)
position_cam = cam_XYZA[x, y, :3]
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
out_info['gripper_forward_direction_camera'] = forward_cam.tolist()
rotmat = np.eye(4).astype(np.float32)
rotmat[:3, 0] = forward
rotmat[:3, 1] = left
rotmat[:3, 2] = up

final_dist = 0.1
if primact_type == 'pushing-left' or primact_type == 'pushing-up':
    final_dist = 0.11

final_rotmat = np.array(rotmat, dtype=np.float32)
final_rotmat[:3, 3] = position_world - action_direction_world * final_dist
final_pose = Pose().from_transformation_matrix(final_rotmat)
out_info['target_rotmat_world'] = final_rotmat.tolist()

start_rotmat = np.array(rotmat, dtype=np.float32)
start_rotmat[:3, 3] = position_world - action_direction_world * 0.15
start_pose = Pose().from_transformation_matrix(start_rotmat)
out_info['start_rotmat_world'] = start_rotmat.tolist()

action_direction = None
if 'left' in primact_type:
    action_direction = forward
elif 'up' in primact_type:
    action_direction = left

if action_direction is not None:
    end_rotmat = np.array(rotmat, dtype=np.float32)
    end_rotmat[:3, 3] = position_world - action_direction_world * final_dist + action_direction * 0.05
    out_info['end_rotmat_world'] = end_rotmat.tolist()


### viz the EE gripper position
# setup robot
robot_urdf_fn = './robots/panda_gripper.urdf'
robot_material = env.get_material(4, 4, 0.01)
robot = Robot(env, robot_urdf_fn, robot_material, open_gripper=('pulling' in primact_type))

# move to the final pose
robot.robot.set_root_pose(final_pose)
env.render()
rgb_final_pose, _ = cam.get_observation()
Image.fromarray((rgb_final_pose*255).astype(np.uint8)).save(os.path.join(out_dir, 'viz_target_pose.png'))

# move back
robot.robot.set_root_pose(start_pose)
env.render()

# activate contact checking
env.start_checking_contact(robot.hand_actor_id, robot.gripper_actor_ids, 'pushing' in primact_type)

if not args.no_gui:
    ### wait to start
    env.wait_to_start()

### main steps
out_info['start_target_part_qpos'] = env.get_target_part_qpos()

target_link_mat44 = env.get_target_part_pose().to_transformation_matrix()
position_local_xyz1 = np.linalg.inv(target_link_mat44) @ position_world_xyz1

success = True
try:
    if 'pushing' in primact_type:
        robot.close_gripper()
    elif 'pulling' in primact_type:
        robot.open_gripper()

    # approach
    robot.move_to_target_pose(final_rotmat, 2000)
    robot.wait_n_steps(2000)

    if 'pulling' in primact_type:
        robot.close_gripper()
        robot.wait_n_steps(2000)
    
    if 'left' in primact_type or 'up' in primact_type:
        robot.move_to_target_pose(end_rotmat, 2000)
        robot.wait_n_steps(2000)
    
    if primact_type == 'pulling':
        robot.move_to_target_pose(start_rotmat, 2000)
        robot.wait_n_steps(2000)

except ContactError:
    success = False

target_link_mat44 = env.get_target_part_pose().to_transformation_matrix()
position_world_xyz1_end = target_link_mat44 @ position_local_xyz1
flog.write('touch_position_world_xyz_start: %s\n' % str(position_world_xyz1))
flog.write('touch_position_world_xyz_end: %s\n' % str(position_world_xyz1_end))
out_info['touch_position_world_xyz_start'] = position_world_xyz1[:3].tolist()
out_info['touch_position_world_xyz_end'] = position_world_xyz1_end[:3].tolist()

if success:
    out_info['result'] = 'VALID'
    out_info['final_target_part_qpos'] = env.get_target_part_qpos()
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
        robot.wait_n_steps(100000000000)
    else:
        print('[Unsuccessful Interaction] invalid gripper-object contact.')
        # close env
        env.close()

