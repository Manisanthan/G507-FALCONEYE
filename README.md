# FollowAnything

FollowAnything is a zero-shot detection, tracking, and drone control system using PyTorch, DINO, SAM, and SiamMask. It supports interactive object selection, video/image-based detection, and real-time drone following.

## Features
- Zero-shot object detection (DINO, CLIP, SAM)
- Object tracking (AOT, SiamMask)
- Interactive selection (click/box)
- Video/image input and output
- Visualization and saving of results

## Folder Structure
- `DINO/` - DINO feature extraction and wrapper
- `VIDEO/` - Video utilities
- `Segment-and-Track-Anything/` - Segmentation and tracking modules
- `SiamMask/` - SiamMask tracker
- `frames/`, `img_logs/`, `result/`, `results/` - Output and logs
- `env/` - Python virtual environment (not tracked)

## Getting Started
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Prepare models:**
   - Download DINO and SiamMask checkpoints as needed.

## How to Run

To run the main script with interactive click detection, SAM, and visualization from your webcam, use:

```bash
python follow_anything.py --detect click --use_sam --path_to_video 0 --plot_visualizations
python follow_anything.py --detect image --path_to_video 0 --plot_visualizations --ref_image <image_path>
```

You can change the arguments as needed:
- `--detect click` for click-based detection (or `dino`, `box`, etc.)
- `--use_sam` to enable SAM segmentation
- `--path_to_video 0` to use the default webcam (or provide a video file path)
- `--plot_visualizations` to show visual output

See the script for more options and argument details.

## Excluding Files from Git
Add the following to your `.gitignore`:
```
env/
__pycache__/
*.pt
*.pth
*.avi
*.mp4
img_logs/
result/
results/
frames/
```


## Credits
- DINO: Facebook Research
- SAM: Meta AI
- SiamMask: Original authors

