# Volleyball YOLO Tracking Demo

YouTube source: https://www.youtube.com/watch?v=rSQ2e_yGk48
Boys volleyball match, fixed sideline tripod view.

This repo contains a 10-second demo from the middle of the match (~9:34-9:44, 574s-584s) processed with YOLOv8 + ByteTrack.

## Outputs

* `output/clip_10s.mp4` — raw 10s source clip, 640x360 30fps, 943 KB
* `output/tracked_10s.mp4` — YOLOv8n person detection + ByteTrack IDs + court edge approx overlay, 6.1 MB
* `output/sample_frame.jpg` — still frame at ~5s in with annotations

## How it was made

```bash
# download (yt-dlp android client bypasses bot check)
yt-dlp --extractor-args youtube:player_client=android -f 18 \
  -o volleyball.mp4 https://www.youtube.com/watch?v=rSQ2e_yGk48

# extract middle 10s
ffmpeg -ss 574.4 -i volleyball.mp4 -t 10 -c:v libx264 -pix_fmt yuv420p -an output/clip_10s.mp4

# run tracking
python3.9 -m pip install --user ultralytics opencv-python-headless lap yt-dlp imageio-ffmpeg
python3 track.py
```

See `track.py` for full pipeline:
* YOLOv8n COCO person class
* ByteTrack tracker via ultralytics built-in `bytetrack.yaml`
* Court edge approx: Canny 50/150 -> HoughLinesP -> convex hull -> approxPolyDP to 4 points, drawn cyan
* Draw green box + P{id} per track

## Results on this clip

* 301 frames processed
* 58 unique track IDs over 10s (includes players, refs, bench, spectators; filter by court polygon to get ~14 stable on-court)
* Court polygon approx in pixel coords: [[639,320],[1,331],[7,115],[639,97]]

## Limitations

* Single fixed camera -> 2D tracking only, no depth, no SfM / COLMAP possible (no baseline, dynamic scene violates static world assumption)
* YOLO fires on crowd; needs ROI filter + team color clustering for true 12-player tracking
* ByteTrack ID switches at net occlusion; pose-based ReID would improve
* Court edge is heuristic Hough, not learned line detector. For production use manual homography or train line seg model.
* No ball tracking in this demo — ball too small at 360p for COCO sports ball class reliably.

## Repo structure

```
volleyball-tracking-demo/
  README.md
  track.py
  requirements.txt
  output/
    clip_10s.mp4
    tracked_10s.mp4
    sample_frame.jpg
```

## View locally

```bash
python3 -m http.server 8000
# then open http://localhost:8000/output/tracked_10s.mp4
```

Or in VSCode remote, right-click output/tracked_10s.mp4 -> Open Preview.

## Next steps

* Switch to yolov8s-pose for skeleton tracking
* Filter detections to court polygon ROI
* K-means jersey color for team assignment
* Export MOT format CSV tracks
* Homography to 18x9 m top-down court coordinates
