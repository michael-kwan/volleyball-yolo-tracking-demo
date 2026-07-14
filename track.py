"""
Volleyball YOLO tracking demo
Source: https://www.youtube.com/watch?v=rSQ2e_yGk48
Extracts middle 10s and runs YOLOv8n + ByteTrack + court edge approx.
"""

import cv2, os, numpy as np, subprocess, imageio_ffmpeg
from ultralytics import YOLO

VIDEO_IN = "volleyball.mp4"  # download with yt-dlp if not present
OUT_DIR = "output"
CLIP_OUT = os.path.join(OUT_DIR, "clip_10s.mp4")
TRACKED_OUT = os.path.join(OUT_DIR, "tracked_10s.mp4")
SAMPLE_OUT = os.path.join(OUT_DIR, "sample_frame.jpg")

os.makedirs(OUT_DIR, exist_ok=True)
ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()


def get_video_info(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, total, w, h


def extract_middle_clip(src, dst, duration=10):
    fps, total, w, h = get_video_info(src)
    dur = total / fps
    start = max(0, dur / 2 - duration / 2)
    print(f"Source {w}x{h} {fps}fps {total} frames {dur:.1f}s")
    print(f"Extracting {start:.1f}s to {start + duration:.1f}s -> {dst}")
    cmd = [
        ffmpeg,
        "-y",
        "-ss",
        str(start),
        "-i",
        src,
        "-t",
        str(duration),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-an",
        dst,
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def detect_court_poly(frame):
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 80, minLineLength=80, maxLineGap=15)
    if lines is None:
        return None
    pts = []
    for l in lines:
        if len(l) == 4:
            x1, y1, x2, y2 = l
        else:
            x1, y1, x2, y2 = l[0]
        pts.append([x1, y1])
        pts.append([x2, y2])
    pts = np.array(pts)
    mask = pts[:, 1] > h * 0.25
    pts_f = pts[mask]
    if len(pts_f) < 4:
        return None
    hull = cv2.convexHull(pts_f)
    epsilon = 0.02 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)
    return approx.reshape(-1, 2)


def main():
    # download note: run yt-dlp manually if volleyball.mp4 missing
    # yt-dlp --extractor-args youtube:player_client=android -f 18 -o volleyball.mp4 https://www.youtube.com/watch?v=rSQ2e_yGk48

    if not os.path.exists(CLIP_OUT):
        extract_middle_clip(VIDEO_IN, CLIP_OUT, 10)

    model = YOLO("yolov8n.pt")  # auto downloads
    cap = cv2.VideoCapture(CLIP_OUT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(TRACKED_OUT, fourcc, fps, (w, h))

    ret, first = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    court_poly = detect_court_poly(first)
    print("Court poly:", court_poly.tolist() if court_poly is not None else None)

    frame_idx = 0
    unique_ids = set()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model.track(
            frame, persist=True, tracker="bytetrack.yaml", classes=[0], verbose=False
        )
        annotated = frame.copy()
        if court_poly is not None:
            cv2.polylines(annotated, [court_poly], True, (0, 255, 255), 2)
            cv2.putText(
                annotated,
                "Court edge approx",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
        r = results[0]
        if r.boxes is not None and r.boxes.id is not None:
            boxes = r.boxes.xyxy.cpu().numpy()
            ids = r.boxes.id.cpu().numpy().astype(int)
            unique_ids.update(ids)
            for box, tid in zip(boxes, ids):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    annotated,
                    f"P{tid}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
        out.write(annotated)
        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"processed {frame_idx} frames")

    cap.release()
    out.release()
    # save sample frame at middle
    cap2 = cv2.VideoCapture(TRACKED_OUT)
    cap2.set(cv2.CAP_PROP_POS_FRAMES, 150)
    ret, sf = cap2.read()
    if ret:
        cv2.imwrite(SAMPLE_OUT, sf)
    cap2.release()
    print(f"Done. {frame_idx} frames, {len(unique_ids)} unique IDs")
    print(f"Output: {TRACKED_OUT}")


if __name__ == "__main__":
    main()
