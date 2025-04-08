# OwlV2SAM: Object Detection and Segmentation

This package combines OWLv2 (Open-World Localization via Vision and Language) with SAM (Segment Anything Model) for powerful object detection and segmentation capabilities.

## Features

- **Text-based object detection**: Use natural language to specify objects to detect
- **Point-based segmentation**: Segment objects by specifying foreground/background points
- **High-quality masks**: Generate precise segmentation masks for detected objects
- **Flexible visualization**: Visualize detection boxes, segmentation masks, and prompt points

# Usage

```bash
python det_seg.py --image <path_to_image> --mode <detect|point> --prompts <text_prompt1> <text_prompt2> --points <x1 y1 x2 y2 ...> --point_labels <1 1 0 0 ...> --multimask --mask_idx <index> --show_box --show_mask --show_points --visualize --save_path <output_path>
```

### Single foreground point

```bash
python det_seg.py --mode point --image path/to/image.jpg --points 500 375 --visualize
```

### Multiple points (foreground and background)

```bash
python det_seg.py --mode point --image path/to/image.jpg --points 500 375 700 450 --point_labels 1 0 --visualize
```

### Get multiple masks

```bash
python det_seg.py --mode point --image path/to/image.jpg --points 500 375 --multimask --visualize
```
