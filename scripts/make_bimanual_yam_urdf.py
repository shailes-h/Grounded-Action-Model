#!/usr/bin/env python3
"""Build a bimanual URDF from I2RT's single-arm YAM URDF (i2rt-robotics/i2rt,
i2rt/robot_models/arm/yam/v1/yam.urdf) by mounting two prefixed copies ("left_"/"right_")
onto a shared root link.

The left/right mounting offset below is a PLACEHOLDER (arms side-by-side, 0.5 m apart,
facing forward) — it is not calibrated to any real rig. It only affects where the arm
meshes render for the silhouette mask; get the real mounting geometry from your bimanual
base plate/frame before trusting silhouette overlays or training a deployable checkpoint.

Usage:
    python scripts/make_bimanual_yam_urdf.py \\
        --src-urdf /path/to/yam.urdf --src-assets /path/to/assets \\
        --out-dir assets/robots/bi_yam
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import xml.etree.ElementTree as ET


def _prefix_urdf(src_urdf: str, prefix: str) -> ET.Element:
    tree = ET.parse(src_urdf)
    root = tree.getroot()
    for link in root.findall("link"):
        link.set("name", prefix + link.get("name"))
        for mesh in link.findall(".//mesh"):
            fn = mesh.get("filename")
            mesh.set("filename", f"assets/{prefix}{os.path.basename(fn)}")
    for joint in root.findall("joint"):
        joint.set("name", prefix + joint.get("name"))
        joint.find("parent").set("link", prefix + joint.find("parent").get("link"))
        joint.find("child").set("link", prefix + joint.find("child").get("link"))
    return root


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src-urdf", required=True)
    ap.add_argument("--src-assets", required=True, help="directory of .stl meshes referenced by --src-urdf")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--arm-separation-m", type=float, default=0.5,
                     help="placeholder left/right mounting distance along Y (m)")
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out_dir, "assets"), exist_ok=True)

    left = _prefix_urdf(args.src_urdf, "left_")
    right = _prefix_urdf(args.src_urdf, "right_")

    out_root = ET.Element("robot", {"name": "bi_yam"})
    ET.SubElement(out_root, "link", {"name": "base_link"})

    for side_root, side, y in [(left, "left", args.arm_separation_m / 2),
                                (right, "right", -args.arm_separation_m / 2)]:
        for link in side_root.findall("link"):
            out_root.append(link)
        for joint in side_root.findall("joint"):
            out_root.append(joint)
        mount = ET.SubElement(out_root, "joint", {"name": f"{side}_mount", "type": "fixed"})
        ET.SubElement(mount, "origin", {"xyz": f"0 {y} 0", "rpy": "0 0 0"})
        ET.SubElement(mount, "parent", {"link": "base_link"})
        ET.SubElement(mount, "child", {"link": f"{side}_base"})

    for prefix in ("left_", "right_"):
        for fn in os.listdir(args.src_assets):
            if fn.lower().endswith(".stl"):
                shutil.copy(os.path.join(args.src_assets, fn),
                            os.path.join(args.out_dir, "assets", prefix + fn))

    out_path = os.path.join(args.out_dir, "bi_yam.urdf")
    ET.ElementTree(out_root).write(out_path, xml_declaration=True, encoding="utf-8")
    print(f"[bimanual-urdf] wrote {out_path}")
    print(f"[bimanual-urdf] WARNING: left/right mounting offset ({args.arm_separation_m} m apart, "
          "no rotation) is a PLACEHOLDER, not calibrated to any real rig.")

    joint_names = ([f"left_joint{i}" for i in range(1, 7)] + ["left_joint7"]
                   + [f"right_joint{i}" for i in range(1, 7)] + ["right_joint7"])
    print("\n[bimanual-urdf] suggested config.perception.robot block:\n")
    print(f"""  robot:
    urdf: {out_path}
    joint_names: {joint_names}
    T_base_root: [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    link_filter: [left_, right_]
    max_faces: 1500
    joint_affine: {{left_joint7: [-0.04695, 0.04695], right_joint7: [-0.04695, 0.04695]}}
    joint_mirror: {{left_joint8: left_joint7, right_joint8: right_joint7}}""")


if __name__ == "__main__":
    main()
