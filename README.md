# FollowAnything

FollowAnything is a zero-shot detection, tracking, and drone control system using PyTorch, DINO, SAM, and SiamMask. It supports interactive object selection, video/image-based detection, and real-time drone following.

## Features
- Zero-shot object detection (DINO, CLIP, SAM)
- Object tracking (AOT, SiamMask)
- Interactive selection (click/box)
- Drone control via MAVSDK
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
3. **Run main script:**
   ```bash
   python follow_anything.py --detect click --tracker aot
   ```
   - See script arguments for options (e.g., `--detect dino`, `--tracker siammask`, etc.)

## Usage
- For interactive detection, use `--detect click` or `--detect box`.
- For video input, set `--path_to_video` to a file or directory.

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
- MAVSDK: Drone control

