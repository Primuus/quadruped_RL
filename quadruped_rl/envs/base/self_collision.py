"""按名称过滤可参与自碰撞的形状；PhysX 相邻 link 过滤不能由此接口解除。"""

from pathlib import Path
import xml.etree.ElementTree as ET


def configure_urdf_self_collisions(gym, asset, asset_path, excluded_pairs):
    """使用预分解碰撞体；导入后形状数量或 link 不匹配时直接拒绝配置。"""
    path = Path(asset_path)
    if path.suffix.lower() != ".urdf":
        raise ValueError("Named self-collision exclusions require a URDF asset")
    if len(excluded_pairs) > 31:
        raise ValueError("At most 31 named self-collision exclusions are supported")

    links = {link.attrib["name"]: link for link in ET.parse(path).getroot().findall("link")}
    names = gym.get_asset_rigid_body_names(asset)
    ranges = gym.get_asset_rigid_body_shape_indices(asset)
    shape_indices = {}
    for body_name, shape_range in zip(names, ranges):
        if body_name not in links:
            raise ValueError("Imported body is absent from URDF: " + body_name)
        collisions = links[body_name].findall("collision")
        if shape_range.count != len(collisions):
            raise ValueError(
                "Named self-collision filtering needs one shape per URDF collision "
                "and unmerged links: " + body_name
            )
        for offset, collision in enumerate(collisions):
            name = collision.get("name")
            if not name or name in shape_indices:
                raise ValueError("URDF collisions must have unique, nonempty names")
            shape_indices[name] = shape_range.start + offset

    properties = gym.get_asset_rigid_shape_properties(asset)
    for shape in properties:
        shape.filter = 0
    # PhysX 中两个 shape 共享过滤位才互相排除；不同排除对使用不同位。
    for bit, pair in enumerate(excluded_pairs):
        if len(pair) != 2 or pair[0] == pair[1]:
            raise ValueError("A self-collision exclusion needs two distinct collision names")
        for name in pair:
            if name not in shape_indices:
                raise ValueError("Unknown self-collision shape: " + str(name))
            properties[shape_indices[name]].filter |= 1 << bit
    gym.set_asset_rigid_shape_properties(asset, properties)
