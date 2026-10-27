"""Primuus 地形拼图；使用 Isaac Gym 标准地形生成器。"""

import numpy as np
from isaacgym import terrain_utils


class PrimuusObstaclesTerrain:
    def __init__(self, cfg):
        self.cfg = cfg
        self.length_pixels = round(cfg.terrain_length / cfg.horizontal_scale)
        self.width_pixels = round(cfg.terrain_width / cfg.horizontal_scale)
        self.border = round(cfg.border_size / cfg.horizontal_scale)
        shape = (cfg.num_rows * self.length_pixels + 2 * self.border,
                 cfg.num_cols * self.width_pixels + 2 * self.border)
        self.height_field_raw = np.zeros(shape, dtype=np.int16)
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3), dtype=np.float32)
        for row in range(cfg.num_rows):
            difficulty = row / max(cfg.num_rows - 1, 1)
            for col in range(cfg.num_cols):
                terrain = self._make_terrain(cfg.terrain_types[col % len(cfg.terrain_types)], difficulty)
                x = self.border + row * self.length_pixels
                y = self.border + col * self.width_pixels
                self.height_field_raw[x:x + self.length_pixels, y:y + self.width_pixels] = terrain.height_field_raw
                self.env_origins[row, col] = (
                    (row + 0.5) * cfg.terrain_length,
                    (col + 0.5) * cfg.terrain_width,
                    terrain.height_field_raw[self.length_pixels // 2, self.width_pixels // 2] * cfg.vertical_scale,
                )
        self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(
            self.height_field_raw, cfg.horizontal_scale, cfg.vertical_scale, cfg.slope_treshold,
        )

    def _make_terrain(self, kind, difficulty):
        cfg = self.cfg
        terrain = terrain_utils.SubTerrain(
            'primuus_' + kind, width=self.length_pixels, length=self.width_pixels,
            vertical_scale=cfg.vertical_scale, horizontal_scale=cfg.horizontal_scale,
        )
        if kind == 'blocks':
            height = np.interp(difficulty, [0.0, 1.0], cfg.block_height_range)
            terrain_utils.discrete_obstacles_terrain(
                terrain, height, *cfg.block_size_range, cfg.num_blocks, platform_size=cfg.platform_size,
            )
        elif kind in ('stairs_up', 'stairs_down'):
            height = np.interp(difficulty, [0.0, 1.0], cfg.step_height_range)
            # 标准生成器内部用int量化；先贴合网格，避免浮点误差少一格。
            height = np.nextafter(round(height / cfg.vertical_scale) * cfg.vertical_scale, np.inf)
            width = np.nextafter(round(cfg.step_width / cfg.horizontal_scale) * cfg.horizontal_scale, np.inf)
            # 从中央平台向外走：负金字塔上行，正金字塔下行。
            if kind == 'stairs_up':
                height = -height
            terrain_utils.pyramid_stairs_terrain(
                terrain, step_width=width, step_height=height, platform_size=cfg.platform_size,
            )
        elif kind in ('slope_up', 'slope_down'):
            slope = np.interp(difficulty, [0.0, 1.0], cfg.slope_range)
            if kind == 'slope_up':
                slope = -slope
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=cfg.platform_size)
        else:
            raise ValueError('Unknown Primuus terrain type: ' + kind)
        return terrain
