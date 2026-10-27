# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
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
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin
"""基于 legged_gym 的网格地形、原点与高度采样，适配 Primuus 历史观测。"""

from isaacgym import gymapi
import torch

from quadruped_rl.utils.math import quat_apply_yaw
from .obstacles_terrain import PrimuusObstaclesTerrain
from .primuus_env import PrimuusEnv


class PrimuusObstaclesEnv(PrimuusEnv):
    def create_sim(self):
        self.up_axis_idx = 2
        self.sim = self.gym.create_sim(
            self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params,
        )
        self.terrain = PrimuusObstaclesTerrain(self.cfg.terrain)
        mesh = gymapi.TriangleMeshParams()
        mesh.nb_vertices = self.terrain.vertices.shape[0]
        mesh.nb_triangles = self.terrain.triangles.shape[0]
        mesh.transform.p.x = -self.cfg.terrain.border_size
        mesh.transform.p.y = -self.cfg.terrain.border_size
        mesh.static_friction = self.cfg.terrain.static_friction
        mesh.dynamic_friction = self.cfg.terrain.dynamic_friction
        mesh.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.ravel(), self.terrain.triangles.ravel(), mesh)
        # 不叠加全局plane，否则负高度地形会被平面遮住。
        self.height_samples = torch.tensor(self.terrain.height_field_raw, device=self.device)
        self._create_envs()

    def _get_env_origins(self):
        self.custom_origins = True
        cfg = self.cfg.terrain
        self.terrain_origins = torch.tensor(self.terrain.env_origins, device=self.device)
        self.terrain_types = torch.div(
            torch.arange(self.num_envs, device=self.device) * cfg.num_cols,
            self.num_envs, rounding_mode='floor',
        )
        self.terrain_levels = torch.randint(cfg.num_rows, (self.num_envs,), device=self.device)
        self.env_origins = self.terrain_origins[self.terrain_levels, self.terrain_types].clone()

    def _init_buffers(self):
        super()._init_buffers()
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device)
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device)
        xy = torch.cartesian_prod(x, y)
        self.num_height_points = xy.shape[0]
        self.height_points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device)
        self.height_points[:, :, :2] = xy
        self.measured_heights = torch.zeros(self.num_envs, self.num_height_points, device=self.device)
        self.privileged_height_scan_obs = torch.zeros_like(self.measured_heights)
        self._update_height_measurements()

    def _update_height_measurements(self, env_ids=None):
        selected = slice(None) if env_ids is None else env_ids
        points = quat_apply_yaw(
            self.base_quat[selected].repeat(1, self.num_height_points), self.height_points[selected],
        ) + self.root_states[selected, None, :3]
        indices = ((points[:, :, :2] + self.cfg.terrain.border_size) / self.cfg.terrain.horizontal_scale).long()
        px = indices[:, :, 0].clamp(0, self.height_samples.shape[0] - 2)
        py = indices[:, :, 1].clamp(0, self.height_samples.shape[1] - 2)
        height = torch.minimum(self.height_samples[px, py], self.height_samples[px + 1, py])
        height = torch.minimum(height, self.height_samples[px, py + 1]) * self.cfg.terrain.vertical_scale
        self.measured_heights[selected] = height
        if self.cfg.terrain.measure_heights:
            self.privileged_height_scan_obs[selected] = (
                self.root_states[selected, 2, None] - 0.5 - height
            ).clamp(-1.0, 1.0) * self.obs_scales.height_measurements

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        self._update_height_measurements()

    def _get_base_heights(self):
        # legged_gym _reward_base_height 的地面高度参考；沿用 Primuus 容差和权重。
        return self.root_states[:, 2] - self.measured_heights.mean(dim=1)

    def check_termination(self):
        super().check_termination()
        cfg = self.cfg.terrain
        half_size = self.root_states.new_tensor([cfg.terrain_length, cfg.terrain_width]) * 0.5
        outside = (torch.abs(self.root_states[:, :2] - self.env_origins[:, :2]) > half_size - cfg.edge_margin).any(dim=1)
        self.time_out_buf |= outside & ~self.reset_buf
        self.reset_buf |= outside

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return super().reset_idx(env_ids)
        # 难度行独立重采样，不按表现晋级；类型列在各环境间均衡分配。
        self.terrain_levels[env_ids] = torch.randint(self.cfg.terrain.num_rows, (len(env_ids),), device=self.device)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
        super().reset_idx(env_ids)
        self.turn_ground_height[env_ids] = self.env_origins[env_ids, 2, None]
        self._update_height_measurements(env_ids)
