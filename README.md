# Volleyball YOLO Tracking Demo

YouTube source: https://www.youtube.com/watch?v=rSQ2e_yGk48
Boys volleyball match, fixed sideline tripod view.

This repo contains a 10-second demo from the middle of the match (~9:34-9:44, 574s-584s) processed with YOLOv8 + ByteTrack, plus v2 with pose skeleton, court lines, and team clustering.

## Outputs

v1 baseline:
* `output/clip_10s.mp4` — raw 10s source clip, 640x360 30fps, 943 KB
* `output/tracked_10s.mp4` — YOLOv8n person detection + ByteTrack IDs + court edge approx overlay, 6.1 MB
* `output/sample_frame.jpg` — still frame at ~5s in with annotations

v2 pose + court lines + team track:
* `output/tracked_pose_10s.mp4` — YOLOv8n-pose skeleton + ByteTrack IDs filtered to court ROI + team color clustering + court line overlay, 7.1 MB
* `output/sample_pose.jpg` — still frame v2
* `track_pose.py` — reproducible v2 pipeline

## How it was made v1 baseline

```bash
# download (yt-dlp android client bypasses bot check)
yt-dlp --extractor-args youtube:player_client=android -f 18 \
  -o volleyball.mp4 https://www.youtube.com/watch?v=rSQ2e_yGk48

# extract middle 10s
ffmpeg -ss 574.4 -i volleyball.mp4 -t 10 -c:v libx264 -pix_fmt yuv420p -an output/clip_10s.mp4

# run tracking v1
python3.9 -m pip install --user -r requirements.txt
python3 track.py
```

`track.py`:
* YOLOv8n COCO person
* ByteTrack tracker
* Court edge approx via Canny+Hough+convex hull, cyan poly
* Green box + P{id}

## v2 – court lines + player track by team + pose skeleton

```bash
python3 track_pose.py
```

`track_pose.py` pipeline:
* **YOLOv8n-pose** – 17 COCO keypoints per person, skeleton drawn with 12 limb connections
* **ByteTrack** persist IDs across frames, same as v1 but now filtered
* **Court lines detection**: HSV threshold white 0-180S 180-255V + yellow 20-40H 80-255S 140-255V -> morph close -> Canny 50/150 -> HoughLinesP rho1 theta1deg thresh60 minLen60 maxGap20 -> draw all line segments in cyan 1px AA. This shows actual painted lines, not just outer polygon.
* **Court ROI polygon**: convex hull of line endpoints below y>0.2h, approxPolyDP epsilon 0.02 -> 4-point polygon drawn orange 2px, labeled "Court ROI". Used to filter detections.
* **Player by track filtering**: point-in-polygon test on bbox center; only detections inside court ROI are kept. This drops bench and spectators.
* **Team clustering**: sample upper torso ROI (20-50% height, 30-70% width) BGR mean color from first 30 frames inside court, KMeans k=2, assign each track to nearest centroid every frame. Draw Team0 in light blue BGR(255,100,100), Team1 in light red BGR(100,100,255), label `P{id} T{team}`.
* Output `output/tracked_pose_10s.mp4` 7.1 MB and `output/sample_pose.jpg`.

## Results v2 on this clip

* 301 frames processed, 30 fps
* Court lines detected: ~140 line segments per frame after HSV filtering, drawn cyan. Outer polygon approx [[639,320],[1,332],[0,98],[630,130]] in pixel coords – roughly sidelines and endlines in perspective.
* Team centers BGR from KMeans: [[32,34,56], [81,86,132]] – dark jersey team vs light jersey team, matches visual white vs dark uniforms in video.
* 28 unique track IDs over 10s inside court ROI (down from 58 in v1 that included crowd). Stable on-court at any instant is 12-14 including refs drifting in, which matches 6v6 volleyball plus libero rotations.
* Skeleton overlay shows pose persistence through jumps at net, though occlusion still causes occasional ID switch.

## Repo structure

```
volleyball-tracking-demo/
  README.md
  track.py          # v1 baseline bbox only
  track_pose.py     # v2 pose + court lines + team track
  requirements.txt
  output/
    clip_10s.mp4
    tracked_10s.mp4
    sample_frame.jpg
    tracked_pose_10s.mp4
    sample_pose.jpg
```

## View

GitHub preview (mp4 renders inline):
* v2 tracked pose: https://github.com/michael-kwan/volleyball-yolo-tracking-demo/blob/master/output/tracked_pose_10s.mp4
* v1 baseline: https://github.com/michael-kwan/volleyball-yolo-tracking-demo/blob/master/output/tracked_10s.mp4
* raw clip: https://github.com/michael-kwan/volleyball-yolo-tracking-demo/blob/master/output/clip_10s.mp4
* sample pose jpg: https://github.com/michael-kwan/volleyball-yolo-tracking-demo/blob/master/output/sample_pose.jpg

Raw download:
```
https://raw.githubusercontent.com/michael-kwan/volleyball-yolo-tracking-demo/master/output/tracked_pose_10s.mp4
```

Local:
```bash
python3 -m http.server 8000
# open http://localhost:8000/output/
```

## Limitations

* Single fixed camera -> 2D tracking only, no depth, no SfM / COLMAP. No baseline, dynamic scene violates static world.
* Court line HSV threshold works for this gym lighting but brittle across venues; production would use learned line segmentation or manual 4-point homography per camera setup.
* Team clustering is simple KMeans on BGR mean; fails with similar jerseys or lighting shifts. Better: HSV histogram + temporal smoothing per track ID.
* ByteTrack still switches IDs on heavy net occlusion; OC-SORT or pose ReID would improve.
* No ball tracking – ball too small at 360p, need fine-tuned volleyball detector at 720p+ or higher.
* No metric court coordinates yet – pixel only. Next step homography to 18x9 m top-down.

## Next steps

* Export MOT CSV with frame,track_id,x,y,w,h,team,pose_kpts
* Homography from 4 court corners to 18x9 m to get top-down trajectories
* YOLOv8s-pose or RTMPose for better skeleton stability
* Fine-tune volleyball ball detector
* Multi-camera sync for true 3D triangulation if you want COLMAP / 3DGS style reconstruction – single view can't give depth
