# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
from enum import Enum
import numpy as np
import torch
import yaml
import json

from isaacgym import gymapi
from isaacgym import gymtorch

from env.tasks.humanoid import Humanoid, dof_to_obs
from utils import gym_util
from utils.motion_lib import MotionLib
from isaacgym.torch_utils import *

from utils import torch_utils

class HumanoidSit(Humanoid):
    class StateInit(Enum):
        Default = 0
        Start = 1
        Random = 2
        Hybrid = 3

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):

        # configs for task
        self._enable_task_obs = cfg["env"]["enableTaskObs"]

        self._enable_conditional_disc = cfg["env"]["enableConditionalDisc"]

        self._sit_vel_penalty = cfg["env"]["sit_vel_penalty"]
        self._sit_vel_pen_coeff = cfg["env"]["sit_vel_pen_coeff"]
        self._sit_vel_pen_thre = cfg["env"]["sit_vel_pen_threshold"]
        self._sit_ang_vel_pen_coeff = cfg["env"]["sit_ang_vel_pen_coeff"]
        self._sit_ang_vel_pen_thre = cfg["env"]["sit_ang_vel_pen_threshold"]

        self._mode = cfg["env"]["mode"] # determine which set of objects to use (train or test)
        assert self._mode in ["train", "test"]

        if cfg["args"].eval:
            self._mode = "test"

        # configs for amp
        state_init = cfg["env"]["stateInit"]
        self._state_init = HumanoidSit.StateInit[state_init]
        self._hybrid_init_prob = cfg["env"]["hybridInitProb"]
        self._num_amp_obs_steps = cfg["env"]["numAMPObsSteps"]
        assert(self._num_amp_obs_steps >= 2)

        self._power_reward = cfg["env"]["power_reward"]
        self._power_coefficient = cfg["env"]["power_coefficient"]

        self._reset_default_env_ids = []
        self._reset_ref_env_ids = {} # to enable multi-skill reference init, use dict instead of list
        self._reset_ref_motion_ids = {}
        self._reset_ref_motion_times = {}

        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)

        self._skill = cfg["env"]["skill"]
        self._skill_init_prob = torch.tensor(cfg["env"]["skillInitProb"], device=self.device, dtype=torch.float) # probs for state init
        self._skill_disc_prob = torch.tensor(cfg["env"]["skillDiscProb"], device=self.device, dtype=torch.float) # probs for amp obs demo fetch

        motion_file = cfg['env']['motion_file']
        self._load_motion(motion_file)

        self._amp_obs_buf = torch.zeros((self.num_envs, self._num_amp_obs_steps, self._num_amp_obs_per_step), device=self.device, dtype=torch.float)
        self._curr_amp_obs_buf = self._amp_obs_buf[:, 0]
        self._hist_amp_obs_buf = self._amp_obs_buf[:, 1:]
        
        self._amp_obs_demo_buf = None

        # tensors for task
        self._prev_root_pos = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)
        self._prev_root_rot = torch.zeros([self.num_envs, 4], device=self.device, dtype=torch.float)
        self._tar_pos = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float) # target location of the object, 3d xyz

        # Interaction Early Termination (IET)
        self._enable_IET = cfg["env"]["enableIET"]
        self._success_threshold = cfg["env"]["successThreshold"]
        if self._enable_IET:
            self._max_IET_steps = cfg["env"]["maxIETSteps"]
            self._IET_step_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
            self._IET_triggered_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        if (not self.headless):
            self._build_marker_state_tensors()

        # tensors for object
        self._build_object_tensors()

        # tensors for enableTrackInitState
        self._every_env_init_dof_pos = torch.zeros((self.num_envs, self.num_dof), device=self.device, dtype=torch.float)

        # tensors for fixing obs bug
        self._kinematic_humanoid_rigid_body_states = torch.zeros((self.num_envs, self.num_bodies, 13), device=self.device, dtype=torch.float)

        ###### evaluation!!!
        self._is_eval = cfg["args"].eval
        if self._is_eval:

            self._enable_IET = False # as default, we disable this feature

            self._success_buf = torch.zeros((self.num_envs), device=self.device, dtype=torch.long)
            self._precision_buf = torch.zeros((self.num_envs), device=self.device, dtype=torch.float)

            self._success_threshold = cfg["env"]["eval"]["successThreshold"]

            self._skill = cfg["env"]["eval"]["skill"]
            self._skill_init_prob = torch.tensor(cfg["env"]["eval"]["skillInitProb"], device=self.device, dtype=torch.float) # probs for state init

        return
    
    def get_obs_size(self):
        obs_size = super().get_obs_size()
        if (self._enable_task_obs):
            task_obs_size = self.get_task_obs_size()
            obs_size += task_obs_size
        return obs_size
    
    def get_task_obs_size(self):
        obs_size = 0
        if (self._enable_task_obs):
            obs_size += 3 # target sit position
            obs_size += 3 + 6 # object pos + 6d rot
            obs_size += 2 # object 2d facing dir
            obs_size += 8 * 3 # object bps
        return obs_size

    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
        self._prev_root_pos[:] = self._humanoid_root_states[..., 0:3]
        self._prev_root_rot[:] = self._humanoid_root_states[..., 3:7]
        return

    def _update_marker(self):
        self._marker_pos[:] = self._tar_pos # 3d xyz
        env_ids_int32 = torch.cat([self._marker_actor_ids, self._object_actor_ids], dim=0)

        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        return
    
    def _create_envs(self, num_envs, spacing, num_per_row):
        if (not self.headless):
            self._marker_handles = []
            self._load_marker_asset()
        
        # load object dataset
        dataset_root = os.path.join(os.path.dirname(self.cfg['env']['motion_file']), "objects", self._mode)
        dataset_categories = self.cfg["env"]["objCategories"]
        dataset_categories_count = []
        obj_urdfs = []
        obj_bbox_centers = []
        obj_bbox_lengths = []
        obj_facings = []
        obj_tar_sit_pos = []
        obj_on_ground_trans = []
        for cat in dataset_categories:
            obj_list = os.listdir(os.path.join(dataset_root, cat))
            dataset_categories_count.append(len(obj_list))
            for obj_name in obj_list:
                curr_dir = os.path.join(dataset_root, cat, obj_name)
                obj_urdfs.append(os.path.join(curr_dir, "asset.urdf"))

                with open(os.path.join(os.getcwd(), curr_dir, "config.json"), "r") as f:
                    object_cfg = json.load(f)
                    assert not np.sum(np.abs(object_cfg["center"])) > 0.0 
                    obj_bbox_centers.append(object_cfg["center"])
                    obj_bbox_lengths.append(object_cfg["bbox"])
                    obj_facings.append(object_cfg["facing"])
                    obj_tar_sit_pos.append(object_cfg["tarSitPos"])
                    obj_on_ground_trans.append(-1 * (obj_bbox_centers[-1][2] - obj_bbox_lengths[-1][2] / 2))

        # randomly sample a fixed object for each simulation env, due to the limitation of IsaacGym
        base_prob = 1.0 / len(dataset_categories)
        averaged_probs = []
        for i in range(len(dataset_categories)):
            averaged_probs += [base_prob * (1.0 / dataset_categories_count[i])] * dataset_categories_count[i]
        weights = torch.tensor(averaged_probs, device=self.device)
        self._every_env_object_ids = torch.multinomial(weights, num_samples=self.num_envs, replacement=True).squeeze(-1)
        if self.num_envs == 1:
            self._every_env_object_ids = self._every_env_object_ids.unsqueeze(0)
        self._every_env_object_bbox_centers = to_torch(obj_bbox_centers, dtype=torch.float, device=self.device)[self._every_env_object_ids]
        self._every_env_object_bbox_lengths = to_torch(obj_bbox_lengths, dtype=torch.float, device=self.device)[self._every_env_object_ids]
        self._every_env_object_facings = to_torch(obj_facings, dtype=torch.float, device=self.device)[self._every_env_object_ids]
        self._every_env_object_tar_sit_pos = to_torch(obj_tar_sit_pos, dtype=torch.float, device=self.device)[self._every_env_object_ids]
        self._every_env_object_on_ground_trans = to_torch(obj_on_ground_trans, dtype=torch.float, device=self.device)[self._every_env_object_ids]

        # load physical assets
        self._object_handles = []
        self._load_object_asset(obj_urdfs)

        super()._create_envs(num_envs, spacing, num_per_row)
        return
    
    def _load_marker_asset(self):
        asset_root = "tokenhsi/data/assets/mjcf/"
        asset_file = "location_marker.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.linear_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.density = 1.0
        asset_options.fix_base_link = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        self._marker_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        return

    def _load_object_asset(self, object_urdfs):
        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.linear_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.fix_base_link = True # fix it
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        # Load materials from meshes
        asset_options.use_mesh_materials = True
        asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX

        # use default convex decomposition params
        asset_options.vhacd_enabled = True
        asset_options.vhacd_params = gymapi.VhacdParams()
        asset_options.vhacd_params.resolution = 100000
        asset_options.vhacd_params.max_convex_hulls = 128
        asset_options.vhacd_params.max_num_vertices_per_ch = 64

        asset_options.replace_cylinder_with_capsule = False # support cylinder

        asset_root = "./"
        self._object_assets = []
        for urdf in object_urdfs:
            self._object_assets.append(self.gym.load_asset(self.sim, asset_root, urdf, asset_options))

        return

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        super()._build_env(env_id, env_ptr, humanoid_asset)
        
        self._build_object(env_id, env_ptr)

        if (not self.headless):
            self._build_marker(env_id, env_ptr)

        return
    
    def _build_object(self, env_id, env_ptr):
        col_group = env_id
        col_filter = 0
        segmentation_id = 0

        default_pose = gymapi.Transform()
        default_pose.p.x = 1
        default_pose.p.y = 0
        default_pose.p.z = self._every_env_object_on_ground_trans[env_id] # ensure no penetration between object and ground plane
        
        object_handle = self.gym.create_actor(env_ptr, self._object_assets[self._every_env_object_ids[env_id]], default_pose, "object", col_group, col_filter, segmentation_id)
        self._object_handles.append(object_handle)

        return

    def _build_marker(self, env_id, env_ptr):
        col_group = self.num_envs + 1
        col_filter = 0
        segmentation_id = 0
        default_pose = gymapi.Transform()
        
        marker_handle = self.gym.create_actor(env_ptr, self._marker_asset, default_pose, "marker", col_group, col_filter, segmentation_id)
        self.gym.set_rigid_body_color(env_ptr, marker_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.8, 0.0, 0.0))
        self.gym.set_actor_scale(env_ptr, marker_handle, 0.5)
        self._marker_handles.append(marker_handle)

        return
    
    def _build_object_tensors(self):
        num_actors = self.get_num_actors_per_env()
        self._object_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., 1, :]
        
        self._object_actor_ids = self._humanoid_actor_ids + 1

        self._initial_object_states = self._object_states.clone()
        self._initial_object_states[:, 7:13] = 0

        self._build_object_bps()
        
        return

    def _build_object_bps(self):

        bps_0 = torch.cat([     self._every_env_object_bbox_lengths[:, 0].unsqueeze(-1) / 2,      self._every_env_object_bbox_lengths[:, 1].unsqueeze(-1) / 2, -1 * self._every_env_object_bbox_lengths[:, 2].unsqueeze(-1) / 2], dim=-1)
        bps_1 = torch.cat([-1 * self._every_env_object_bbox_lengths[:, 0].unsqueeze(-1) / 2,      self._every_env_object_bbox_lengths[:, 1].unsqueeze(-1) / 2, -1 * self._every_env_object_bbox_lengths[:, 2].unsqueeze(-1) / 2], dim=-1)
        bps_2 = torch.cat([-1 * self._every_env_object_bbox_lengths[:, 0].unsqueeze(-1) / 2, -1 * self._every_env_object_bbox_lengths[:, 1].unsqueeze(-1) / 2, -1 * self._every_env_object_bbox_lengths[:, 2].unsqueeze(-1) / 2], dim=-1)
        bps_3 = torch.cat([     self._every_env_object_bbox_lengths[:, 0].unsqueeze(-1) / 2, -1 * self._every_env_object_bbox_lengths[:, 1].unsqueeze(-1) / 2, -1 * self._every_env_object_bbox_lengths[:, 2].unsqueeze(-1) / 2], dim=-1)
        bps_4 = torch.cat([     self._every_env_object_bbox_lengths[:, 0].unsqueeze(-1) / 2,      self._every_env_object_bbox_lengths[:, 1].unsqueeze(-1) / 2,      self._every_env_object_bbox_lengths[:, 2].unsqueeze(-1) / 2], dim=-1)
        bps_5 = torch.cat([-1 * self._every_env_object_bbox_lengths[:, 0].unsqueeze(-1) / 2,      self._every_env_object_bbox_lengths[:, 1].unsqueeze(-1) / 2,      self._every_env_object_bbox_lengths[:, 2].unsqueeze(-1) / 2], dim=-1)
        bps_6 = torch.cat([-1 * self._every_env_object_bbox_lengths[:, 0].unsqueeze(-1) / 2, -1 * self._every_env_object_bbox_lengths[:, 1].unsqueeze(-1) / 2,      self._every_env_object_bbox_lengths[:, 2].unsqueeze(-1) / 2], dim=-1)
        bps_7 = torch.cat([     self._every_env_object_bbox_lengths[:, 0].unsqueeze(-1) / 2, -1 * self._every_env_object_bbox_lengths[:, 1].unsqueeze(-1) / 2,      self._every_env_object_bbox_lengths[:, 2].unsqueeze(-1) / 2], dim=-1)
        
        self._every_env_object_bps = torch.cat([
            bps_0.unsqueeze(1),
            bps_1.unsqueeze(1),
            bps_2.unsqueeze(1),
            bps_3.unsqueeze(1),
            bps_4.unsqueeze(1),
            bps_5.unsqueeze(1),
            bps_6.unsqueeze(1),
            bps_7.unsqueeze(1)]
        , dim=1)
            
        self._every_env_object_bps += self._every_env_object_bbox_centers.unsqueeze(1) # (num_envs, 8, 3)

        return

    def _build_marker_state_tensors(self):
        num_actors = self._root_states.shape[0] // self.num_envs
        self._marker_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., 2, :] # humanoid, object, marker
        self._marker_pos = self._marker_states[..., :3]
        
        self._marker_actor_ids = self._humanoid_actor_ids + 2

        return

    def _reset_task(self, env_ids):
        target_sit_locations = self._every_env_object_tar_sit_pos[env_ids]

        # transform from object local space to world space
        translation_to_world = self._object_states[env_ids, 0:3]
        rotation_to_world = self._object_states[env_ids, 3:7]
        target_sit_locations = quat_rotate(rotation_to_world, target_sit_locations) + translation_to_world

        self._tar_pos[env_ids] = target_sit_locations

        return
    
    def _compute_observations(self, env_ids=None):
        humanoid_obs = self._compute_humanoid_obs(env_ids)
        
        if (self._enable_task_obs):
            task_obs = self._compute_task_obs(env_ids)
            obs = torch.cat([humanoid_obs, task_obs], dim=-1)
        else:
            obs = humanoid_obs

        if (env_ids is None):
            self.obs_buf[:] = obs
        else:
            self.obs_buf[env_ids] = obs
        return

    def _compute_task_obs(self, env_ids=None):
        if (env_ids is None):
            root_states = self._humanoid_root_states
            object_states = self._object_states
            tar_pos = self._tar_pos
            
            object_bps = self._every_env_object_bps
            object_facings = self._every_env_object_facings
        else:
            root_states = self._humanoid_root_states[env_ids]
            object_states = self._object_states[env_ids]
            tar_pos = self._tar_pos[env_ids]

            object_bps = self._every_env_object_bps[env_ids]
            object_facings = self._every_env_object_facings[env_ids]

        obs = compute_location_observations(root_states, tar_pos, object_states, object_bps, object_facings)

        return obs

    def _compute_reward(self, actions):
        root_pos = self._humanoid_root_states[..., 0:3]
        root_rot = self._humanoid_root_states[..., 3:7]
        object_pos = self._object_states[..., 0:3]
        object_rot = self._object_states[..., 3:7]

        location_reward = compute_location_reward(root_pos, self._prev_root_pos, root_rot, self._prev_root_rot, object_pos, self._tar_pos, 1.5, self.dt,
                                                  self._sit_vel_penalty, self._sit_vel_pen_coeff, self._sit_vel_pen_thre, self._sit_ang_vel_pen_coeff, self._sit_ang_vel_pen_thre,
                                                  )

        power = torch.abs(torch.multiply(self.dof_force_tensor, self._dof_vel)).sum(dim = -1)
        power_reward = -self._power_coefficient * power

        if self._power_reward:
            self.rew_buf[:] = location_reward + power_reward
        else:
            self.rew_buf[:] = location_reward
        
        if self._enable_IET:
            mask = compute_finish_state(root_pos, self._tar_pos, self._success_threshold)
            self._IET_step_buf[mask] += 1

        return
    
    def _compute_reset(self):
        if self._enable_IET:
            self.reset_buf[:], self._terminate_buf[:], self._IET_triggered_buf[:] = compute_humanoid_reset_with_IET(self.reset_buf, self.progress_buf,
                                                    self._contact_forces, self._contact_body_ids,
                                                    self._rigid_body_pos, self.max_episode_length,
                                                    self._enable_early_termination, self._termination_heights,
                                                    self._max_IET_steps, self._IET_step_buf)
        else:
            self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(self.reset_buf, self.progress_buf,
                                                   self._contact_forces, self._contact_body_ids,
                                                   self._rigid_body_pos, self.max_episode_length,
                                                   self._enable_early_termination, self._termination_heights)
        return
    
    def render(self, sync_frame_time=False):
        super().render(sync_frame_time)

        if self.viewer:
            self._draw_task()
        return

    def _draw_task(self):
        self._update_marker()

        cols = np.array([
            [0.0, 1.0, 0.0], # green
        ], dtype=np.float32)

        self.gym.clear_lines(self.viewer)

        if self._show_lines_flag:

            starts = self._humanoid_root_states[..., 0:3] # line from humanoid to marker
            ends = self._tar_pos[..., 0:3]

            verts = torch.cat([starts, ends], dim=-1).cpu().numpy()

            for i, env_ptr in enumerate(self.envs):
                curr_verts = verts[i]
                curr_verts = curr_verts.reshape([1, 6])
                self.gym.add_lines(self.viewer, env_ptr, curr_verts.shape[0], curr_verts, cols)

            # draw lines of the bbox
            cols = np.zeros((12, 3), dtype=np.float32) # 24 lines
            cols[:12] = [1.0, 0.0, 0.0] # red

            # transform bps from object local space to world space
            object_bps = self._every_env_object_bps.clone()
            object_pos = self._object_states[:, 0:3]
            object_rot = self._object_states[:, 3:7]
            object_pos_exp = torch.broadcast_to(object_pos.unsqueeze(-2), (object_pos.shape[0], object_bps.shape[1], object_pos.shape[1])) # (num_envs, 3) >> (num_envs, 8, 3)
            object_rot_exp = torch.broadcast_to(object_rot.unsqueeze(-2), (object_rot.shape[0], object_bps.shape[1], object_rot.shape[1])) # (num_envs, 4) >> (num_envs, 8, 4)
            object_bps_world_space = (quat_rotate(object_rot_exp.reshape(-1, 4), object_bps.reshape(-1, 3)) + object_pos_exp.reshape(-1, 3)).reshape(self.num_envs, 8, 3) # (num_envs, 8, 3)

            verts = torch.cat([
                object_bps_world_space[:, 0, :], object_bps_world_space[:, 1, :],
                object_bps_world_space[:, 1, :], object_bps_world_space[:, 2, :],
                object_bps_world_space[:, 2, :], object_bps_world_space[:, 3, :],
                object_bps_world_space[:, 3, :], object_bps_world_space[:, 0, :],

                object_bps_world_space[:, 4, :], object_bps_world_space[:, 5, :],
                object_bps_world_space[:, 5, :], object_bps_world_space[:, 6, :],
                object_bps_world_space[:, 6, :], object_bps_world_space[:, 7, :],
                object_bps_world_space[:, 7, :], object_bps_world_space[:, 4, :],

                object_bps_world_space[:, 0, :], object_bps_world_space[:, 4, :],
                object_bps_world_space[:, 1, :], object_bps_world_space[:, 5, :],
                object_bps_world_space[:, 2, :], object_bps_world_space[:, 6, :],
                object_bps_world_space[:, 3, :], object_bps_world_space[:, 7, :],
            ], dim=-1).cpu()

            for i, env_ptr in enumerate(self.envs):
                curr_verts = verts[i]
                curr_verts = curr_verts.reshape([12, 6])
                self.gym.add_lines(self.viewer, env_ptr, curr_verts.shape[0], curr_verts, cols)

            num_verts = 30
            axis = torch.tensor([[0.0, 0.0, 1.0]], device=self.device).reshape(1, 3).expand([num_verts, -1])
            ang = torch.linspace(0, 2 * np.pi, num_verts, device=self.device)
            quat = quat_from_angle_axis(ang, axis) # (num_verts, 4)

            axis = torch.tensor([[1.0, 0.0, 0.0]], device=self.device).reshape(1, 3).expand([num_verts, -1])
            axis = quat_rotate(quat, axis)
            pos  = axis * 0.5 # radius: 0.5m

            cols = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)

            for i, env_ptr in enumerate(self.envs):
                verts = pos.clone()
                verts += self._object_states[i, 0:3].unsqueeze(0)
                lines = torch.cat([verts[:-1], verts[1:]], dim=-1).cpu().numpy()
                curr_cols = np.broadcast_to(cols, [lines.shape[0], cols.shape[-1]])
                self.gym.add_lines(self.viewer, env_ptr, lines.shape[0], lines, curr_cols)
            
            axis = torch.tensor([[1.0, 0.0, 0.0]], device=self.device).reshape(1, 3).expand([num_verts, -1])
            axis = quat_rotate(quat, axis)
            pos  = axis * 1.5 # radius: 1.5m

            cols = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)

            for i, env_ptr in enumerate(self.envs):
                verts = pos.clone()
                verts += self._object_states[i, 0:3].unsqueeze(0)
                lines = torch.cat([verts[:-1], verts[1:]], dim=-1).cpu().numpy()
                curr_cols = np.broadcast_to(cols, [lines.shape[0], cols.shape[-1]])
                self.gym.add_lines(self.viewer, env_ptr, lines.shape[0], lines, curr_cols)

        return

    def post_physics_step(self):
        super().post_physics_step()
        
        self._update_hist_amp_obs()
        self._compute_amp_observations()

        amp_obs_flat = self._amp_obs_buf.view(-1, self.get_num_amp_obs())
        self.extras["amp_obs"] = amp_obs_flat
        self.extras["policy_obs"] = self.obs_buf.clone()

        if self._is_eval:
            self._compute_metrics_evaluation()
            self.extras["success"] = self._success_buf
            self.extras["precision"] = self._precision_buf

        return
    
    def _compute_metrics_evaluation(self):
        root_pos = self._humanoid_root_states[..., 0:3]
        root_rot = self._humanoid_root_states[..., 3:7]
        tar_pos_sit = self._tar_pos

        pos_diff = tar_pos_sit - root_pos
        pos_err = torch.norm(pos_diff, p=2, dim=-1)
        dist_mask = pos_err <= self._success_threshold
        self._success_buf[dist_mask] += 1

        self._precision_buf[dist_mask] = torch.where(pos_err[dist_mask] < self._precision_buf[dist_mask], pos_err[dist_mask], self._precision_buf[dist_mask])

        return

    def get_num_amp_obs(self):
        return self._num_amp_obs_steps * self._num_amp_obs_per_step

    def fetch_amp_obs_demo(self, num_samples):

        # random sample a skill
        sk_id = torch.multinomial(self._skill_disc_prob, num_samples=1, replacement=True)
        sk_name = self._skill[sk_id]
        curr_motion_lib = self._motion_lib[sk_name]

        if (self._amp_obs_demo_buf is None):
            self._build_amp_obs_demo_buf(num_samples)
        else:
            assert(self._amp_obs_demo_buf.shape[0] == num_samples)
        
        motion_ids = curr_motion_lib.sample_motions(num_samples)
        
        # since negative times are added to these values in build_amp_obs_demo,
        # we shift them into the range [0 + truncate_time, end of clip]
        truncate_time = self.dt * (self._num_amp_obs_steps - 1)
        motion_times0 = curr_motion_lib.sample_time(motion_ids, truncate_time=truncate_time)
        motion_times0 += truncate_time

        amp_obs_demo = self.build_amp_obs_demo(motion_ids, motion_times0, curr_motion_lib, sk_name)
        self._amp_obs_demo_buf[:] = amp_obs_demo.view(self._amp_obs_demo_buf.shape)
        amp_obs_demo_flat = self._amp_obs_demo_buf.view(-1, self.get_num_amp_obs())

        return amp_obs_demo_flat

    def build_amp_obs_demo(self, motion_ids, motion_times0, motion_lib, skill_name):
        dt = self.dt

        motion_ids = torch.tile(motion_ids.unsqueeze(-1), [1, self._num_amp_obs_steps])
        motion_times = motion_times0.unsqueeze(-1)
        time_steps = -dt * torch.arange(0, self._num_amp_obs_steps, device=self.device)
        motion_times = motion_times + time_steps

        motion_ids = motion_ids.view(-1)
        motion_times = motion_times.view(-1)
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
               = motion_lib.get_motion_state(motion_ids, motion_times)

        # randomly assign objects
        num_samples = len(motion_ids) // self._num_amp_obs_steps
        randomly_selected_env_ids = torch.multinomial(torch.ones(self.num_envs, device=self.device) * (1.0 / self.num_envs), num_samples=num_samples, replacement=True)

        if skill_name == "loco":
            object_root_pos_xy = torch.randn(num_samples, 2, device=self.device)
            object_root_pos_xy /= torch.linalg.norm(object_root_pos_xy, dim=-1, keepdim=True)
            object_root_pos_xy *= torch.rand(num_samples, 1, device=self.device) * 14.0 + 1.0 # randomize [1, 15]
            object_root_pos_xy += root_pos.reshape(num_samples, self._num_amp_obs_steps, -1)[:, 0, 0:2] # use the first frame

            object_root_pos_z = self._every_env_object_on_ground_trans[randomly_selected_env_ids]

            object_pos = torch.cat([object_root_pos_xy, object_root_pos_z.unsqueeze(-1)], dim=-1)

            axis = torch.tensor([[0.0, 0.0, 1.0]], device=self.device).reshape(1, 3).expand([num_samples, -1])
            ang = torch.rand((num_samples,), device=self.device) * 2 * np.pi
            object_rot = quat_from_angle_axis(ang, axis)
        
        else:
            object_pos, object_rot = motion_lib.get_obj_motion_state_single_frame(motion_ids.reshape(num_samples, self._num_amp_obs_steps)[:, 0])
 
            object_pos[:, -1] = self._every_env_object_on_ground_trans[randomly_selected_env_ids]

        object_pos = object_pos.repeat(1, self._num_amp_obs_steps).reshape(root_pos.shape[0], -1)
        object_rot = object_rot.repeat(1, self._num_amp_obs_steps).reshape(root_pos.shape[0], -1)

        object_bps = self._every_env_object_bps[randomly_selected_env_ids]
        object_facings = self._every_env_object_facings[randomly_selected_env_ids]
        object_bps = torch.broadcast_to(object_bps.unsqueeze(1), (object_bps.shape[0], self._num_amp_obs_steps, object_bps.shape[1], object_bps.shape[2])).reshape(root_pos.shape[0], 8, 3)
        object_facings = object_facings.repeat(1, self._num_amp_obs_steps).reshape(root_pos.shape[0], -1)

        amp_obs_demo = build_amp_observations(root_pos, root_rot, root_vel, root_ang_vel,
                                              dof_pos, dof_vel, key_pos,
                                              self._local_root_obs, self._root_height_obs,
                                              self._dof_obs_size, self._dof_offsets,
                                              object_pos, object_rot, object_bps, object_facings, self._enable_conditional_disc)

        return amp_obs_demo

    def _build_amp_obs_demo_buf(self, num_samples):
        self._amp_obs_demo_buf = torch.zeros((num_samples, self._num_amp_obs_steps, self._num_amp_obs_per_step), device=self.device, dtype=torch.float32)
        return
        
    def _setup_character_props(self, key_bodies):
        super()._setup_character_props(key_bodies)

        asset_file = self.cfg["env"]["asset"]["assetFileName"]
        num_key_bodies = len(key_bodies)

        if (asset_file == "mjcf/amp_humanoid.xml"):
            self._num_amp_obs_per_step = 13 + self._dof_obs_size + 28 + 3 * num_key_bodies # [root_h, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos]
        elif (asset_file == "mjcf/phys_humanoid.xml") or (asset_file == "mjcf/phys_humanoid_v2.xml") or (asset_file == "mjcf/phys_humanoid_v3.xml"):
            self._num_amp_obs_per_step = 13 + self._dof_obs_size + 28 + 2 * 2 + 3 * num_key_bodies # [root_h, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos]
        else:
            print("Unsupported character config file: {s}".format(asset_file))
            assert(False)

        if self._enable_conditional_disc:
            self._num_amp_obs_per_step += 3 + 6 + 8 * 3 + 2

        return

    def _load_motion(self, motion_file):
        assert(self._dof_offsets[-1] == self.num_dof)

        ext = os.path.splitext(motion_file)[1]
        if (ext == ".yaml"):

            with open(os.path.join(os.getcwd(), motion_file), 'r') as f:
                motion_config = yaml.load(f, Loader=yaml.SafeLoader)

            self._skill_categories = list(motion_config['motions'].keys()) # all skill names stored in the yaml file
            self._motion_lib = {}
            for skill in self._skill_categories:
                self._motion_lib[skill] = MotionLib(motion_file=motion_file,
                                                    skill=skill,
                                                    dof_body_ids=self._dof_body_ids,
                                                    dof_offsets=self._dof_offsets,
                                                    key_body_ids=self._key_body_ids.cpu().numpy(), 
                                                    device=self.device)
        else:
            raise NotImplementedError

        return
    
    def _reset_envs(self, env_ids):
        self._reset_default_env_ids = []
        self._reset_ref_env_ids = {}
        self._reset_ref_motion_ids = {}
        self._reset_ref_motion_times = {}
        if (len(env_ids) > 0):
            self._reset_actors(env_ids)
            self._reset_objects(env_ids)
            self._reset_task(env_ids)
            self._reset_env_tensors(env_ids)
            self._refresh_sim_tensors()
            self._compute_observations(env_ids)
        
        self._init_amp_obs(env_ids)

        return
    
    def _reset_objects(self, env_ids):

        # for skill is sit, the initial location of the object is from the reference object motion
        for sk_name in ["sit"]:
            if self._reset_ref_env_ids.get(sk_name) is not None:
                if (len(self._reset_ref_env_ids[sk_name]) > 0):
                    object_root_pos, object_root_rot = self._motion_lib[sk_name].get_obj_motion_state_single_frame(self._reset_ref_motion_ids[sk_name])
                    self._object_states[self._reset_ref_env_ids[sk_name], 0:2] = object_root_pos[..., 0:2]
                    self._object_states[self._reset_ref_env_ids[sk_name], 2] = self._every_env_object_on_ground_trans[self._reset_ref_env_ids[sk_name]]
                    self._object_states[self._reset_ref_env_ids[sk_name], 3:7] = object_root_rot

        # for skill is loco and reset default, we random generate an inital location of the object
        random_env_ids = []
        if len(self._reset_default_env_ids) > 0:
            random_env_ids.append(self._reset_default_env_ids)
        for sk_name in ["loco"]:
            if self._reset_ref_env_ids.get(sk_name) is not None:
                random_env_ids.append(self._reset_ref_env_ids[sk_name])

        if len(random_env_ids) > 0:
            ids = torch.cat(random_env_ids, dim=0)

            root_pos_xy = torch.randn(len(ids), 2, device=self.device)
            root_pos_xy /= torch.linalg.norm(root_pos_xy, dim=-1, keepdim=True)
            root_pos_xy *= torch.rand(len(ids), 1, device=self.device) * 4.0 + 1.0 # randomize
            root_pos_xy += self._humanoid_root_states[ids, :2] # get absolute pos, humanoid_root_state will be updated after set_env_state

            root_pos_z = torch.zeros((len(ids)), device=self.device, dtype=torch.float32)
            root_pos_z[:] = self._every_env_object_on_ground_trans[ids] # place the object on the ground

            axis = torch.tensor([[0.0, 0.0, 1.0]], device=self.device).reshape(1, 3).expand([ids.shape[0], -1])
            ang = torch.rand((len(ids),), device=self.device) * 2 * np.pi
            root_rot = quat_from_angle_axis(ang, axis)
            root_pos = torch.cat([root_pos_xy, root_pos_z.unsqueeze(-1)], dim=-1)

            self._object_states[ids, 0:3] = root_pos
            self._object_states[ids, 3:7] = root_rot
            self._object_states[ids, 7:10] = 0.0
            self._object_states[ids, 10:13] = 0.0

        return
    
    def _reset_env_tensors(self, env_ids):
        super()._reset_env_tensors(env_ids)
        
        if self._enable_IET:
            self._IET_step_buf[env_ids] = 0
            self._IET_triggered_buf[env_ids] = 0
        
        if self._is_eval:
            self._success_buf[env_ids] = 0
            self._precision_buf[env_ids] = float('Inf')

        env_ids_int32 = self._object_actor_ids[env_ids].view(-1)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        return

    def _reset_actors(self, env_ids):
        if (self._state_init == HumanoidSit.StateInit.Default):
            self._reset_default(env_ids)
        elif (self._state_init == HumanoidSit.StateInit.Start
              or self._state_init == HumanoidSit.StateInit.Random):
            self._reset_ref_state_init(env_ids)
        elif (self._state_init == HumanoidSit.StateInit.Hybrid):
            self._reset_hybrid_state_init(env_ids)
        else:
            assert(False), "Unsupported state initialization strategy: {:s}".format(str(self._state_init))
        return
    
    def _reset_default(self, env_ids):
        root_pos = self._initial_humanoid_root_states[env_ids, 0:3]
        # root_rot = self._initial_humanoid_root_states[env_ids, 3:7]
        root_vel = self._initial_humanoid_root_states[env_ids, 7:10]
        root_ang_vel = self._initial_humanoid_root_states[env_ids, 10:13]
        dof_pos = self._initial_dof_pos[env_ids]
        dof_vel = self._initial_dof_vel[env_ids]
        
        if len(env_ids) > 0:
            pos2d_dist = torch.distributions.uniform.Uniform(
                torch.tensor([-5.0, -5.0], device=self.device), 
                torch.tensor([5.0, 5.0], device=self.device))
            pos2d = pos2d_dist.sample((len(env_ids),))
            root_pos[:, 0:2] = pos2d

            axis = torch.tensor([[0.0, 0.0, 1.0]], device=self.device).reshape(1, 3).expand([env_ids.shape[0], -1])
            ang = torch.rand((len(env_ids),), device=self.device) * 2 * np.pi
            root_rot = quat_from_angle_axis(ang, axis)
        
        self._set_env_state(env_ids=env_ids, 
                            root_pos=root_pos, 
                            root_rot=root_rot, 
                            dof_pos=dof_pos, 
                            root_vel=root_vel, 
                            root_ang_vel=root_ang_vel, 
                            dof_vel=dof_vel)

        self._reset_default_env_ids = env_ids
        self._kinematic_humanoid_rigid_body_states[env_ids] = self._initial_humanoid_rigid_body_states[env_ids]
        self._every_env_init_dof_pos[env_ids] = self._initial_dof_pos[env_ids] # for "enableTrackInitState"

        return

    def _reset_ref_state_init(self, env_ids):
        sk_ids = torch.multinomial(self._skill_init_prob, num_samples=env_ids.shape[0], replacement=True)

        for uid, sk_name in enumerate(self._skill):
            curr_motion_lib = self._motion_lib[sk_name]
            curr_env_ids = env_ids[(sk_ids == uid).nonzero().squeeze(-1)] # be careful!!!

            if len(curr_env_ids) > 0:

                num_envs = curr_env_ids.shape[0]
                motion_ids = curr_motion_lib.sample_motions(num_envs)

                if (self._state_init == HumanoidSit.StateInit.Random
                    or self._state_init == HumanoidSit.StateInit.Hybrid):
                    motion_times = curr_motion_lib.sample_time_rsi(motion_ids) # avoid times with serious self-penetration
                elif (self._state_init == HumanoidSit.StateInit.Start):
                    motion_times = torch.zeros(num_envs, device=self.device)
                else:
                    assert(False), "Unsupported state initialization strategy: {:s}".format(str(self._state_init))

                root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
                    = curr_motion_lib.get_motion_state(motion_ids, motion_times)

                self._set_env_state(env_ids=curr_env_ids, 
                                    root_pos=root_pos, 
                                    root_rot=root_rot, 
                                    dof_pos=dof_pos, 
                                    root_vel=root_vel, 
                                    root_ang_vel=root_ang_vel, 
                                    dof_vel=dof_vel)

                self._reset_ref_env_ids[sk_name] = curr_env_ids
                self._reset_ref_motion_ids[sk_name] = motion_ids
                self._reset_ref_motion_times[sk_name] = motion_times

                # update buffer for kinematic humanoid state
                body_pos, body_rot, body_vel, body_ang_vel \
                    = curr_motion_lib.get_motion_state_max(motion_ids, motion_times)
                self._kinematic_humanoid_rigid_body_states[curr_env_ids] = torch.cat((body_pos, body_rot, body_vel, body_ang_vel), dim=-1)

                self._every_env_init_dof_pos[curr_env_ids] = dof_pos # for "enableTrackInitState"

        return

    def _reset_hybrid_state_init(self, env_ids):
        num_envs = env_ids.shape[0]
        ref_probs = to_torch(np.array([self._hybrid_init_prob] * num_envs), device=self.device)
        ref_init_mask = torch.bernoulli(ref_probs) == 1.0

        ref_reset_ids = env_ids[ref_init_mask]
        if (len(ref_reset_ids) > 0):
            self._reset_ref_state_init(ref_reset_ids)

        default_reset_ids = env_ids[torch.logical_not(ref_init_mask)]
        if (len(default_reset_ids) > 0):
            self._reset_default(default_reset_ids)

        return

    def _init_amp_obs(self, env_ids):
        self._compute_amp_observations(env_ids)

        if (len(self._reset_default_env_ids) > 0):
            self._init_amp_obs_default(self._reset_default_env_ids)

        for i, sk_name in enumerate(self._skill):
            if self._reset_ref_env_ids.get(sk_name) is not None:
                if (len(self._reset_ref_env_ids[sk_name]) > 0):
                    self._init_amp_obs_ref(self._reset_ref_env_ids[sk_name], self._reset_ref_motion_ids[sk_name],
                                           self._reset_ref_motion_times[sk_name], sk_name)

        return

    def _init_amp_obs_default(self, env_ids):
        curr_amp_obs = self._curr_amp_obs_buf[env_ids].unsqueeze(-2)
        self._hist_amp_obs_buf[env_ids] = curr_amp_obs
        return

    def _init_amp_obs_ref(self, env_ids, motion_ids, motion_times, skill_name):
        dt = self.dt
        motion_ids = torch.tile(motion_ids.unsqueeze(-1), [1, self._num_amp_obs_steps - 1])
        motion_times = motion_times.unsqueeze(-1)
        time_steps = -dt * (torch.arange(0, self._num_amp_obs_steps - 1, device=self.device) + 1)
        motion_times = motion_times + time_steps

        motion_ids = motion_ids.view(-1)
        motion_times = motion_times.view(-1)
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
               = self._motion_lib[skill_name].get_motion_state(motion_ids, motion_times)
        
        object_pos = self._object_states[env_ids, 0:3].repeat(1, self._num_amp_obs_steps - 1).reshape(root_pos.shape[0], -1)
        object_rot = self._object_states[env_ids, 3:7].repeat(1, self._num_amp_obs_steps - 1).reshape(root_pos.shape[0], -1)

        object_bps = self._every_env_object_bps[env_ids]
        object_facings = self._every_env_object_facings[env_ids]
        object_bps = torch.broadcast_to(object_bps.unsqueeze(1), (object_bps.shape[0], self._num_amp_obs_steps - 1, object_bps.shape[1], object_bps.shape[2])).reshape(root_pos.shape[0], 8, 3)
        object_facings = object_facings.repeat(1, self._num_amp_obs_steps - 1).reshape(root_pos.shape[0], -1)

        amp_obs_demo = build_amp_observations(root_pos, root_rot, root_vel, root_ang_vel, 
                                              dof_pos, dof_vel, key_pos, 
                                              self._local_root_obs, self._root_height_obs, 
                                              self._dof_obs_size, self._dof_offsets,
                                              object_pos, object_rot, object_bps, object_facings, self._enable_conditional_disc)

        self._hist_amp_obs_buf[env_ids] = amp_obs_demo.view(self._hist_amp_obs_buf[env_ids].shape)
        return
    
    def _set_env_state(self, env_ids, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel):
        self._humanoid_root_states[env_ids, 0:3] = root_pos
        self._humanoid_root_states[env_ids, 3:7] = root_rot
        self._humanoid_root_states[env_ids, 7:10] = root_vel
        self._humanoid_root_states[env_ids, 10:13] = root_ang_vel
        
        self._dof_pos[env_ids] = dof_pos
        self._dof_vel[env_ids] = dof_vel
        return

    def _update_hist_amp_obs(self, env_ids=None):
        if (env_ids is None):
            for i in reversed(range(self._amp_obs_buf.shape[1] - 1)):
                self._amp_obs_buf[:, i + 1] = self._amp_obs_buf[:, i]
        else:
            for i in reversed(range(self._amp_obs_buf.shape[1] - 1)):
                self._amp_obs_buf[env_ids, i + 1] = self._amp_obs_buf[env_ids, i]
        return
    
    def _compute_amp_observations(self, env_ids=None):
        if (env_ids is None):
            object_bps = self._every_env_object_bps
            object_facings = self._every_env_object_facings

            key_body_pos = self._rigid_body_pos[:, self._key_body_ids, :]
            self._curr_amp_obs_buf[:] = build_amp_observations(self._rigid_body_pos[:, 0, :],
                                                               self._rigid_body_rot[:, 0, :],
                                                               self._rigid_body_vel[:, 0, :],
                                                               self._rigid_body_ang_vel[:, 0, :],
                                                               self._dof_pos, self._dof_vel, key_body_pos,
                                                               self._local_root_obs, self._root_height_obs, 
                                                               self._dof_obs_size, self._dof_offsets,
                                                               self._object_states[:, 0:3], self._object_states[:, 3:7], object_bps, object_facings, self._enable_conditional_disc)
        else:
            object_bps = self._every_env_object_bps[env_ids]
            object_facings = self._every_env_object_facings[env_ids]

            kinematic_rigid_body_pos = self._kinematic_humanoid_rigid_body_states[:, :, 0:3]
            key_body_pos = kinematic_rigid_body_pos[:, self._key_body_ids, :]
            self._curr_amp_obs_buf[env_ids] = build_amp_observations(self._kinematic_humanoid_rigid_body_states[env_ids, 0, 0:3],
                                                                   self._kinematic_humanoid_rigid_body_states[env_ids, 0, 3:7],
                                                                   self._kinematic_humanoid_rigid_body_states[env_ids, 0, 7:10],
                                                                   self._kinematic_humanoid_rigid_body_states[env_ids, 0, 10:13],
                                                                   self._dof_pos[env_ids], self._dof_vel[env_ids], key_body_pos[env_ids],
                                                                   self._local_root_obs, self._root_height_obs, 
                                                                   self._dof_obs_size, self._dof_offsets,
                                                                   self._object_states[env_ids, 0:3], self._object_states[env_ids, 3:7], object_bps, object_facings, self._enable_conditional_disc)
        return


#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def compute_finish_state(root_pos, tar_pos, success_threshold):
    # type: (Tensor, Tensor, float) -> Tensor
    pos_diff = tar_pos - root_pos
    pos_err = torch.norm(pos_diff, p=2, dim=-1)
    dist_mask = pos_err <= success_threshold
    return dist_mask

@torch.jit.script
def compute_humanoid_reset_with_IET(reset_buf, progress_buf, contact_buf, contact_body_ids, rigid_body_pos,
                           max_episode_length, enable_early_termination, termination_heights,
                           max_IET_steps, IET_step_buff):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, bool, Tensor, int, Tensor) -> Tuple[Tensor, Tensor, Tensor]
    terminated = torch.zeros_like(reset_buf)
    IET_triggered = torch.zeros_like(reset_buf)

    if (enable_early_termination):

        # fall down
        body_height = rigid_body_pos[..., 2]
        fall_height = body_height < termination_heights
        fall_height[:, contact_body_ids] = False
        fall_height = torch.any(fall_height, dim=-1)
        has_fallen = torch.logical_and(torch.ones_like(fall_height), fall_height)

        terminated = torch.where(has_fallen, torch.ones_like(reset_buf), terminated)

    IET_triggered = torch.where(IET_step_buff >= max_IET_steps - 1, torch.ones_like(reset_buf), IET_triggered)
    reset = torch.logical_or(IET_triggered, terminated)
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset)

    return reset, terminated, IET_triggered

@torch.jit.script
def compute_humanoid_reset(reset_buf, progress_buf, contact_buf, contact_body_ids, rigid_body_pos,
                           max_episode_length, enable_early_termination, termination_heights):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, bool, Tensor) -> Tuple[Tensor, Tensor]
    terminated = torch.zeros_like(reset_buf)

    if (enable_early_termination):

        # fall down
        body_height = rigid_body_pos[..., 2]
        fall_height = body_height < termination_heights
        fall_height[:, contact_body_ids] = False
        fall_height = torch.any(fall_height, dim=-1)
        has_fallen = torch.logical_and(torch.ones_like(fall_height), fall_height)

        terminated = torch.where(has_fallen, torch.ones_like(reset_buf), terminated)
    
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated)

    return reset, terminated

@torch.jit.script
def build_amp_observations(root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos, 
                           local_root_obs, root_height_obs, dof_obs_size, dof_offsets,
                           object_pos, object_rot, object_bps, object_facings, enable_object_condition_disc):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, bool, int, List[int], Tensor, Tensor, Tensor, Tensor, bool) -> Tensor
    root_h = root_pos[:, 2:3]
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

    if (local_root_obs):
        root_rot_obs = quat_mul(heading_rot, root_rot)
    else:
        root_rot_obs = root_rot
    root_rot_obs = torch_utils.quat_to_tan_norm(root_rot_obs)
    
    if (not root_height_obs):
        root_h_obs = torch.zeros_like(root_h)
    else:
        root_h_obs = root_h
    
    local_root_vel = quat_rotate(heading_rot, root_vel)
    local_root_ang_vel = quat_rotate(heading_rot, root_ang_vel)

    root_pos_expand = root_pos.unsqueeze(-2)
    local_key_body_pos = key_body_pos - root_pos_expand
    
    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, local_key_body_pos.shape[1], 1))
    flat_end_pos = local_key_body_pos.view(local_key_body_pos.shape[0] * local_key_body_pos.shape[1], local_key_body_pos.shape[2])
    flat_heading_rot = heading_rot_expand.view(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], 
                                               heading_rot_expand.shape[2])
    local_end_pos = quat_rotate(flat_heading_rot, flat_end_pos)
    flat_local_key_pos = local_end_pos.view(local_key_body_pos.shape[0], local_key_body_pos.shape[1] * local_key_body_pos.shape[2])
    
    dof_obs = dof_to_obs(dof_pos, dof_obs_size, dof_offsets)
    obs = torch.cat((root_h_obs, root_rot_obs, local_root_vel, local_root_ang_vel, dof_obs, dof_vel, flat_local_key_pos), dim=-1)
    
    if enable_object_condition_disc:
        local_object_root_pos = quat_rotate(heading_rot, object_pos - root_pos)
        local_object_root_rot = quat_mul(heading_rot, object_rot)
        local_object_root_rot = torch_utils.quat_to_tan_norm(local_object_root_rot)

        face_vec_world_space = quat_rotate(object_rot, object_facings)
        face_vec_local_space = quat_rotate(heading_rot, face_vec_world_space)

        obj_root_pos_exp = torch.broadcast_to(object_pos.unsqueeze(1), (object_pos.shape[0], 8, object_pos.shape[1])).reshape(-1, 3) # [4096, 3] >> [4096, 8, 3] >> [4096*8, 3]
        obj_root_rot_exp = torch.broadcast_to(object_rot.unsqueeze(1), (object_rot.shape[0], 8, object_rot.shape[1])).reshape(-1, 4) # [4096, 4] >> [4096, 8, 4] >> [4096*8, 4]
        root_pos_exp = torch.broadcast_to(root_pos.unsqueeze(1), (root_pos.shape[0], 8, root_pos.shape[1])).reshape(-1, 3) # [4096, 3] >> [4096, 8, 3] >> [4096*8, 3]
        heading_rot_exp = torch.broadcast_to(heading_rot.unsqueeze(1), (heading_rot.shape[0], 8, heading_rot.shape[1])).reshape(-1, 4) # [4096, 4] >> [4096, 8, 4] >> [4096*8, 4]

        obj_bps_world_space = quat_rotate(obj_root_rot_exp, object_bps.reshape(-1, 3)) + obj_root_pos_exp
        obj_bps_local_space = quat_rotate(heading_rot_exp, obj_bps_world_space - root_pos_exp).reshape(-1, 24)

        obs = torch.cat([obs, obj_bps_local_space, face_vec_local_space[..., 0:2], local_object_root_pos, local_object_root_rot], dim=-1)

    return obs

@torch.jit.script
def compute_location_observations(root_states, tar_pos, object_states, object_bps, object_facings):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor
    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot) # (num_envs, 4)

    # compute observations for target
    local_tar_pos = quat_rotate(heading_rot, tar_pos - root_pos) # 3d xyz

    obj_root_pos = object_states[:, 0:3]
    obj_root_rot = object_states[:, 3:7]

    local_object_root_pos = quat_rotate(heading_rot, obj_root_pos - root_pos)
    local_object_root_rot = quat_mul(heading_rot, obj_root_rot)
    local_object_root_rot = torch_utils.quat_to_tan_norm(local_object_root_rot)

    obj_root_pos_exp = torch.broadcast_to(obj_root_pos.unsqueeze(1), (obj_root_pos.shape[0], 8, obj_root_pos.shape[1])).reshape(-1, 3) # [4096, 3] >> [4096, 8, 3] >> [4096*8, 3]
    obj_root_rot_exp = torch.broadcast_to(obj_root_rot.unsqueeze(1), (obj_root_rot.shape[0], 8, obj_root_rot.shape[1])).reshape(-1, 4) # [4096, 4] >> [4096, 8, 4] >> [4096*8, 4]
    root_pos_exp = torch.broadcast_to(root_pos.unsqueeze(1), (root_pos.shape[0], 8, root_pos.shape[1])).reshape(-1, 3) # [4096, 3] >> [4096, 8, 3] >> [4096*8, 3]
    heading_rot_exp = torch.broadcast_to(heading_rot.unsqueeze(1), (heading_rot.shape[0], 8, heading_rot.shape[1])).reshape(-1, 4) # [4096, 4] >> [4096, 8, 4] >> [4096*8, 4]

    obj_bps_world_space = quat_rotate(obj_root_rot_exp, object_bps.reshape(-1, 3)) + obj_root_pos_exp
    obj_bps_local_space = quat_rotate(heading_rot_exp, obj_bps_world_space - root_pos_exp).reshape(-1, 24)

    face_vec_world_space = quat_rotate(obj_root_rot, object_facings)
    face_vec_local_space = quat_rotate(heading_rot, face_vec_world_space)
    
    obs = torch.cat([local_tar_pos, obj_bps_local_space, face_vec_local_space[..., 0:2], local_object_root_pos, local_object_root_rot], dim=-1)

    return obs

@torch.jit.script
def compute_location_reward(root_pos, prev_root_pos, root_rot, prev_root_rot, object_root_pos, tar_pos, tar_speed, dt,
                            sit_vel_penalty, sit_vel_pen_coeff, sit_vel_penalty_thre, sit_ang_vel_pen_coeff, sit_ang_vel_penalty_thre):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, float, bool, float, float, float, float) -> Tensor

    d_obj2human_xy = torch.sum((object_root_pos[..., 0:2] - root_pos[..., 0:2]) ** 2, dim=-1)
    reward_far_pos = torch.exp(-0.5 * d_obj2human_xy)

    delta_root_pos = root_pos - prev_root_pos
    root_vel = delta_root_pos / dt
    tar_dir = object_root_pos[..., 0:2] - root_pos[..., 0:2] # d* is a horizontal unit vector pointing from the root to the object's location
    tar_dir = torch.nn.functional.normalize(tar_dir, dim=-1)
    tar_dir_speed = torch.sum(tar_dir * root_vel[..., :2], dim=-1)
    tar_vel_err = tar_speed - tar_dir_speed
    reward_far_vel = torch.exp(-2.0 * tar_vel_err * tar_vel_err)

    reward_far_final = 0.0 * reward_far_pos + 1.0 * reward_far_vel
    dist_mask = (d_obj2human_xy <= 0.5 ** 2)
    reward_far_final[dist_mask] = 1.0

    # when humanoid is close to the object
    reward_near = torch.exp(-10.0 * torch.sum((tar_pos - root_pos) ** 2, dim=-1))

    reward = 0.7 * reward_near + 0.3 * reward_far_final

    if sit_vel_penalty:
        min_speed_penalty = sit_vel_penalty_thre
        root_vel_norm = torch.norm(root_vel, p=2, dim=-1)
        root_vel_norm = torch.clamp_min(root_vel_norm, min_speed_penalty)
        root_vel_err = min_speed_penalty - root_vel_norm
        root_vel_penalty = -1 * sit_vel_pen_coeff * (1 - torch.exp(-2.0 * (root_vel_err * root_vel_err)))
        dist_mask = (d_obj2human_xy <= 1.5 ** 2)
        root_vel_penalty[~dist_mask] = 0.0
        reward += root_vel_penalty

        root_z_ang_vel = torch.abs((get_euler_xyz(root_rot)[2] - get_euler_xyz(prev_root_rot)[2]) / dt)
        root_z_ang_vel = torch.clamp_min(root_z_ang_vel, sit_ang_vel_penalty_thre)
        root_z_ang_vel_err = sit_ang_vel_penalty_thre - root_z_ang_vel
        root_z_ang_vel_penalty = -1 * sit_ang_vel_pen_coeff * (1 - torch.exp(-0.5 * (root_z_ang_vel_err ** 2)))
        root_z_ang_vel_penalty[~dist_mask] = 0.0
        reward += root_z_ang_vel_penalty

    return reward

@torch.jit.script
def compute_kinematic_rigid_body_states(root_pos, root_rot, initial_humanoid_rigid_body_states):
    # type: (Tensor, Tensor, Tensor) -> Tensor

    num_envs = initial_humanoid_rigid_body_states.shape[0]
    num_bodies = initial_humanoid_rigid_body_states.shape[1] # 15

    root_pos_exp = torch.broadcast_to(root_pos.unsqueeze(1), (root_pos.shape[0], num_bodies, root_pos.shape[1])).reshape(-1, 3) # (num_envs, 3) >> (num_envs, 15, 3) >> (num_envs*15, 3)
    root_rot_exp = torch.broadcast_to(root_rot.unsqueeze(1), (root_rot.shape[0], num_bodies, root_rot.shape[1])).reshape(-1, 4) # (num_envs, 4) >> (num_envs, 15, 4) >> (num_envs*15, 4)

    init_body_pos = initial_humanoid_rigid_body_states[..., 0:3] # (num_envs, 15, 3)
    init_body_rot = initial_humanoid_rigid_body_states[..., 3:7] # (num_envs, 15, 4)
    init_body_vel = initial_humanoid_rigid_body_states[..., 7:10] # (num_envs, 15, 3)
    init_body_ang_vel = initial_humanoid_rigid_body_states[..., 10:13] # (num_envs, 15, 3)

    init_root_pos = init_body_pos[:, 0:1, :] # (num_envs, 1, 3)
    init_body_pos_canonical = (init_body_pos - init_root_pos).reshape(-1, 3) # (num_envs, 15, 3) >> (num_envs*15, 3)
    init_body_rot = init_body_rot.reshape(-1, 4)
    init_body_vel = init_body_vel.reshape(-1, 3)
    init_body_ang_vel = init_body_ang_vel.reshape(-1, 3)
    
    curr_body_pos = (quat_rotate(root_rot_exp, init_body_pos_canonical) + root_pos_exp).reshape(-1, num_bodies, 3)
    curr_body_rot = (quat_mul(root_rot_exp, init_body_rot)).reshape(-1, num_bodies, 4)
    curr_body_vel = (quat_rotate(root_rot_exp, init_body_vel)).reshape(-1, num_bodies, 3)
    curr_body_ang_vel = (quat_rotate(root_rot_exp, init_body_ang_vel)).reshape(-1, num_bodies, 3)
    curr_humanoid_rigid_body_states = torch.cat((curr_body_pos, curr_body_rot, curr_body_vel, curr_body_ang_vel), dim=-1)
    
    return curr_humanoid_rigid_body_states
