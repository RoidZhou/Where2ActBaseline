"""
    Environment with one object at center
        external: one robot, one camera
"""

from __future__ import division
import sapien.core as sapien
from sapien.core import Pose, SceneConfig, OptifuserConfig
from transforms3d.quaternions import axangle2quat, qmult
import numpy as np
from utils import process_angle_limit, get_random_number


class ContactError(Exception):
    pass


class Env(object):
    
    def __init__(self, flog=None, show_gui=True, render_rate=20, timestep=1/500, \
            object_position_offset=0.0, succ_ratio=0.1):
        self.current_step = 0

        self.flog = flog
        self.show_gui = show_gui
        self.render_rate = render_rate
        self.timestep = timestep
        self.succ_ratio = succ_ratio
        self.object_position_offset = object_position_offset

        # engine and renderer
        self.engine = sapien.Engine(0, 0.001, 0.005)
        
        render_config = OptifuserConfig()
        render_config.shadow_map_size = 8192
        render_config.shadow_frustum_size = 10
        render_config.use_shadow = False
        render_config.use_ao = True
        
        self.renderer = sapien.OptifuserRenderer(config=render_config)
        self.renderer.enable_global_axes(False)
        
        self.engine.set_renderer(self.renderer)

        # GUI
        self.window = False
        if show_gui:
            self.renderer_controller = sapien.OptifuserController(self.renderer)
            self.renderer_controller.set_camera_position(-3.0+object_position_offset, 1.0, 3.0)
            self.renderer_controller.set_camera_rotation(-0.4, -0.8)

        # scene
        scene_config = SceneConfig()
        scene_config.gravity = [0, 0, -9.81]
        scene_config.solver_iterations = 20
        scene_config.enable_pcm = False
        scene_config.sleep_threshold = 0.0

        self.scene = self.engine.create_scene(config=scene_config)
        if show_gui:
            self.renderer_controller.set_current_scene(self.scene)

        self.scene.set_timestep(timestep)

        # add lights
        self.scene.set_ambient_light([0.5, 0.5, 0.5])
        self.scene.set_shadow_light([0, 1, -1], [0.5, 0.5, 0.5])
        self.scene.add_point_light([1+object_position_offset, 2, 2], [1, 1, 1])
        self.scene.add_point_light([1+object_position_offset, -2, 2], [1, 1, 1])
        self.scene.add_point_light([-1+object_position_offset, 0, 1], [1, 1, 1])

        # self.scene.add_ground(altitude=-0.1, render=False) # 添加地面
        self.scene.add_ground(altitude=-0.1) # 添加地面
        # default Nones
        self.object = None
        self.object_target_joint = None

        # check contact
        self.check_contact = False
        self.step_length = 0
        self.finger1_contact = False
        self.finger2_contact = False

    def set_controller_camera_pose(self, x, y, z, yaw, pitch):
        self.renderer_controller.set_camera_position(x, y, z)
        self.renderer_controller.set_camera_rotation(yaw, pitch)
        self.renderer_controller.render()

    def load_object(self, urdf, material, state='closed', custom = True):
        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = False
        self.object = loader.load(urdf, {"material": material})
        #self.object = loader.load(urdf, material)
        pose = Pose([self.object_position_offset, 0, 0], [1, 0, 0, 0])
        self.object.set_root_pose(pose)

        # compute link actor information
        self.all_link_ids = [l.get_id() for l in self.object.get_links()]
        self.movable_link_ids = []
        if custom:
            for j in self.object.get_joints():
                if j.get_dof() == 1:
                    self.movable_link_ids.append(j.get_child_link().get_id())
        else:
            dummy_id = 1
            self.movable_link_ids.append(dummy_id)
            self.target_object_part_joint_id = dummy_id
        if self.flog is not None:
            self.flog.write('All Actor Link IDs: %s\n' % str(self.all_link_ids))
            self.flog.write('All Movable Actor Link IDs: %s\n' % str(self.movable_link_ids))

        # set joint property
        for joint in self.object.get_joints():
            joint.set_drive_property(stiffness=0, damping=10)

        # set initial qpos
        joint_angles = []
        self.joint_angles_lower = []
        self.joint_angles_upper = []
        for j in self.object.get_joints():
            if j.get_dof() == 1:
                l = process_angle_limit(j.get_limits()[0, 0])
                self.joint_angles_lower.append(float(l))
                r = process_angle_limit(j.get_limits()[0, 1])
                self.joint_angles_upper.append(float(r))
                if state == 'closed':
                    joint_angles.append(float(l))
                elif state == 'open':
                    joint_angles.append(float(r))
                elif state == 'random-middle':
                    joint_angles.append(float(get_random_number(l, r)))
                elif state == 'random-closed-middle':
                    if np.random.random() < 0.5:
                        joint_angles.append(float(get_random_number(l, r)))
                    else:
                        joint_angles.append(float(l))
                else:
                    raise ValueError('ERROR: object init state %s unknown!' % state)
        self.object.set_qpos(joint_angles)
        return joint_angles

    def set_object_joint_angles(self, joint_angles):
        self.object.set_qpos(joint_angles)

    def set_target_object_part_actor_id(self, actor_id, custom=True):
        if self.flog is not None:
            self.flog.write('Set Target Object Part Actor ID: %d\n' % actor_id)
        self.target_object_part_actor_id = actor_id
        self.non_target_object_part_actor_id = list(set(self.all_link_ids) - set([actor_id])) # 转换成一个集合（set），set是一个无序的不重复元素集

        # get the link handler
        if custom:
            for j in self.object.get_joints():
                if j.get_dof() == 1:
                    if j.get_child_link().get_id() == actor_id:
                        self.target_object_part_actor_link = j.get_child_link()
        else:
            self.target_object_part_actor_link  = self.object.get_links()
        # moniter the target joint
        idx = 0
        for j in self.object.get_joints():
            if j.get_dof() == 1:
                if j.get_child_link().get_id() == actor_id:
                    self.target_object_part_joint_id = idx
                idx += 1

    def get_object_qpos(self):
        return self.object.get_qpos()

    def get_target_part_qpos(self, custom=True):
        if custom:
            qpos = self.object.get_qpos()
            return float(qpos[self.target_object_part_joint_id])
        else:
            qpos = [1,1]
            return float(qpos[self.target_object_part_joint_id])
    
    def get_target_part_pose(self, custom=True):
        if custom:
            return self.target_object_part_actor_link.get_pose()
        else:
            return self.target_object_part_actor_link[0].get_pose()

    def start_checking_contact(self, robot_hand_actor_id, robot_gripper_actor_ids, strict=False, custom=True):
        self.check_contact = True
        self.check_contact_strict = strict
        if custom:
            self.first_timestep_check_contact = True
        self.robot_hand_actor_id = robot_hand_actor_id
        self.robot_gripper_actor_ids = robot_gripper_actor_ids

    def end_checking_contact(self, custom=True):
        self.check_contact = False
        if not custom:
            self.first_timestep_check_contact = False
            self.step_length = 0

    def get_material(self, static_friction, dynamic_friction, restitution):
        return self.engine.create_physical_material(static_friction, dynamic_friction, restitution)

    def render(self):
        if self.show_gui and (not self.window):
            self.window = True
            self.renderer_controller.show_window()
        self.scene.update_render()
        if self.show_gui and (self.current_step % self.render_rate == 0):
            self.renderer_controller.render()

    def step(self, custom=True):
        self.current_step += 1
        self.scene.step()
        if self.check_contact:
            if not self.check_contact_is_valid(custom):
                raise ContactError()

    # check the first contact: only gripper links can touch the target object part link
    def check_contact_is_valid(self, custom=True):
        self.contacts = self.scene.get_contacts()
        contact = False; valid = False; 
        if custom:
            for c in self.contacts:
                aid1 = c.actor1.get_id()
                aid2 = c.actor2.get_id()
                has_impulse = False
                for p in c.points:
                    if abs(p.impulse @ p.impulse) > 1e-4:
                        has_impulse = True
                        break
                if has_impulse:
                    if (aid1 in self.robot_gripper_actor_ids and aid2 == self.target_object_part_actor_id) or \
                    (aid2 in self.robot_gripper_actor_ids and aid1 == self.target_object_part_actor_id): # aid1 and 2 local in robot and object respectively
                        contact, valid = True, True
                    if (aid1 in self.robot_gripper_actor_ids and aid2 in self.non_target_object_part_actor_id) or \
                    (aid2 in self.robot_gripper_actor_ids and aid1 in self.non_target_object_part_actor_id):
                        if self.check_contact_strict: # 检查接触是否严格 "pushing" coordinate to true
                            return False
                        else:
                            contact, valid = True, True
                    if (aid1 == self.robot_hand_actor_id or aid2 == self.robot_hand_actor_id):
                        if self.check_contact_strict:
                            return False
                        else:
                            contact, valid = True, True
                    # starting pose should have no collision at all
                    if (aid1 in self.robot_gripper_actor_ids or aid1 == self.robot_hand_actor_id or \
                        aid2 in self.robot_gripper_actor_ids or aid2 == self.robot_hand_actor_id) and self.first_timestep_check_contact:
                            return False

            self.first_timestep_check_contact = False
            if contact and valid:
                self.check_contact = False # check only at first time
            return True
        else:
            print("all contact ", self.contacts)
            for c in self.contacts:
                aid1 = c.actor1.get_id()
                aid2 = c.actor2.get_id()
                has_impulse = False

                for p in c.points:
                    if abs(p.impulse @ p.impulse) > 1e-4:
                        has_impulse = True
                        break
                if has_impulse and self.first_timestep_check_contact:
                    print("first contact object")

                    return False
                    # return True
                elif has_impulse and not self.first_timestep_check_contact:
                    print("last contact object", self.step_length)
                    self.step_length += 1
                    if (aid1 in self.robot_gripper_actor_ids and aid2 in self.robot_gripper_actor_ids): # robot gripper Link(panda_leftfinger/panda_rightfinger)
                        return False

                    else:
                        # print("c.actor name", c.actor1.get_name(), c.actor2.get_name())
                        if (c.actor1.get_name()=='ground' and aid2 == self.robot_hand_actor_id) or (c.actor2.get_name()=='ground' and aid1 == self.robot_hand_actor_id) or \
                           (c.actor1.get_name()=='ground' and aid2 in self.robot_gripper_actor_ids) or (c.actor2.get_name()=='ground' and aid1 in self.robot_gripper_actor_ids):
                            print("contact ground")
                            return False
                        elif (c.actor1.get_name()=='baseLink' and aid2 in self.robot_gripper_actor_ids):
                            self.finger1_contact = True
                            return True
                        elif (c.actor2.get_name()=='baseLink' and aid1 in self.robot_gripper_actor_ids):
                            self.finger2_contact = True
                            return True
                        
                        # elif (c.actor1.get_name()=='baseLink' and aid2 == self.robot_hand_actor_id) or (c.actor2.get_name()=='baseLink' and aid1 == self.robot_hand_actor_id):
                        #     return True
                elif not has_impulse and self.first_timestep_check_contact:
                    print("first not contact object", self.step_length)
                    self.step_length += 1
                    return True
            if (self.finger1_contact and self.finger2_contact):
                return True
            else:
                print("no contact succ")
                return False

    def close_render(self):
        if self.window:
            self.renderer_controller.hide_window()
        self.window = False
    
    def wait_to_start(self):
        print('press q to start\n')
        while not self.renderer_controller.should_quit:
            self.scene.update_render()
            if self.show_gui:
                self.renderer_controller.render()

    def close(self):
        if self.show_gui:
            self.renderer_controller.set_current_scene(None)
        self.scene = None

