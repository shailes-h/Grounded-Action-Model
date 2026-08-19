# Vendored third-party packages

Minimal import closure of the RGB-D detector backbone used by `gam.perception.patch_tokens`:

| package | origin | role |
| --- | --- | --- |
| `wilddet3d/` | WildDet3D (open-vocabulary 3D detector: SAM3 detector + metric-depth backend) | click-prompted 2D/3D boxes and the 49x49 RGB-D feature map that becomes the patch tokens |
| `sam3/` | Segment Anything 3 (Meta) — model code + BPE vocab | detector/transformer used inside WildDet3D |
| `mdm/` | LingBot-Depth (DINOv2 RGB-D encoder / decoder) | metric depth backend (`robbyant/lingbot-depth-postrain-dc-vitl14`) |

Only the files needed at inference were copied; training / evaluation / visualisation code
was left out. Each package remains under its upstream license — add the corresponding LICENSE
files here before redistributing.
