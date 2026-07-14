"""
Volleyball YOLO Pose Tracking v2
Source: https://www.youtube.com/watch?v=rSQ2e_yGk48
- YOLOv8n-pose for skeleton
- ByteTrack for ID persistence
- Court line detection via HSV white/yellow threshold + HoughLinesP
- Court ROI polygon filter to keep only on-court players
- Team clustering via KMeans on jersey color (upper torso BGR mean)
"""

import cv2, os, numpy as np, subprocess, imageio_ffmpeg
from ultralytics import YOLO
from sklearn.cluster import KMeans

VIDEO_IN = "volleyball.mp4"
OUT_DIR = "output"
CLIP_OUT = os.path.join(OUT_DIR, "clip_10s.mp4")
TRACKED_OUT = os.path.join(OUT_DIR, "tracked_pose_10s.mp4")
SAMPLE_OUT = os.path.join(OUT_DIR, "sample_pose.jpg")

os.makedirs(OUT_DIR, exist_ok=True)
ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()


def get_info(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(3))
    h = int(cap.get(4))
    cap.release()
    return fps, total, w, h


def extract_middle(src, dst, duration=10):
    fps, total, w, h = get_info(src)
    start = max(0, total / fps / 2 - duration / 2)
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
    print(f"Extracted {start:.1f}s clip to {dst}")


def detect_court_lines_and_poly(frame):
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 60, 255])
    lower_yellow = np.array([20, 80, 140])
    upper_yellow = np.array([40, 255, 255])
    mask = cv2.bitwise_or(
        cv2.inRange(hsv, lower_white, upper_white),
        cv2.inRange(hsv, lower_yellow, upper_yellow),
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, 1)
    mask = cv2.dilate(mask, kernel, 1)
    edges = cv2.Canny(mask, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 60, minLineLength=60, maxLineGap=20)
    line_list = []
    if lines is not None:
        for l in lines:
            x1, y1, x2, y2 = l if len(l) == 4 else l[0]
            if np.hypot(x2 - x1, y2 - y1) > 40:
                line_list.append((x1, y1, x2, y2))
    pts = []
    for x1, y1, x2, y2 in line_list:
        pts.append([x1, y1])
        pts.append([x2, y2])
    poly = None
    if pts:
        pts = np.array(pts)
        pts_f = pts[pts[:, 1] > h * 0.2]
        if len(pts_f) >= 4:
            hull = cv2.convexHull(pts_f)
            approx = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True)
            poly = approx.reshape(-1, 2)
    return line_list, poly


def point_in_poly(x, y, poly):
    if poly is None:
        return True
    return cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0


def jersey_color(frame, box, w, h):
    x1, y1, x2, y2 = map(int, box)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w - 1, x2)
    y2 = min(h - 1, y2)
    hb = y2 - y1
    wb = x2 - x1
    if hb <= 0 or wb <= 0:
        return np.array([128, 128, 128], float)
    uy1 = y1 + int(0.2 * hb)
    uy2 = y1 + int(0.5 * hb)
    ux1 = x1 + int(0.3 * wb)
    ux2 = x1 + int(0.7 * wb)
    roi = frame[uy1:uy2, ux1:ux2]
    if roi.size == 0:
        return np.array([128, 128, 128], float)
    return roi.reshape(-1, 3).mean(axis=0)


def main():
    if not os.path.exists(CLIP_OUT):
        extract_middle(VIDEO_IN, CLIP_OUT, 10)
    model = YOLO("yolov8n-pose.pt")
    cap = cv2.VideoCapture(CLIP_OUT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(3))
    h = int(cap.get(4))
    out = cv2.VideoWriter(TRACKED_OUT, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    ret, first = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    court_lines, court_poly = detect_court_lines_and_poly(first)
    print(
        "Lines:",
        len(court_lines) if court_lines else 0,
        "Poly:",
        court_poly.tolist() if court_poly is not None else None,
    )
    # team clustering from first 30 frames
    colors = []
    tmp = cv2.VideoCapture(CLIP_OUT)
    for _ in range(30):
        ret, f = tmp.read()
        if not ret:
            break
        res = model.track(
            f, persist=True, tracker="bytetrack.yaml", classes=[0], verbose=False
        )[0]
        if res.boxes is None or res.boxes.id is None:
            continue
        for b in res.boxes.xyxy.cpu().numpy():
            cx = (b[0] + b[2]) / 2
            cy = (b[1] + b[3]) / 2
            if not point_in_poly(cx, cy, court_poly):
                continue
            colors.append(jersey_color(f, b, w, h))
    tmp.release()
    kmeans = KMeans(n_clusters=2, n_init=10, random_state=0).fit(np.array(colors))
    centers = kmeans.cluster_centers_
    print("Team centers BGR:", centers)

    def assign_team(col):
        return int(np.argmin([np.linalg.norm(col - c) for c in centers]))

    skeleton = [
        (5, 7),
        (7, 9),
        (6, 8),
        (8, 10),
        (5, 6),
        (5, 11),
        (6, 12),
        (11, 12),
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),
    ]
    unique = set()
    idx = 0
    cap = cv2.VideoCapture(CLIP_OUT)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        res = model.track(
            frame, persist=True, tracker="bytetrack.yaml", classes=[0], verbose=False
        )[0]
        ann = frame.copy()
        if court_lines:
            for x1, y1, x2, y2 in court_lines:
                cv2.line(ann, (x1, y1), (x2, y2), (0, 255, 255), 1, cv2.LINE_AA)
        if court_poly is not None:
            cv2.polylines(ann, [court_poly], True, (0, 200, 255), 2)
            cv2.putText(
                ann,
                "Court ROI",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 200, 255),
                2,
            )
        if res.boxes is not None and res.boxes.id is not None:
            boxes = res.boxes.xyxy.cpu().numpy()
            ids = res.boxes.id.cpu().numpy().astype(int)
            kps_all = (
                res.keypoints.xy.cpu().numpy() if res.keypoints is not None else None
            )
            for i, (box, tid) in enumerate(zip(boxes, ids)):
                x1, y1, x2, y2 = map(int, box)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                if not point_in_poly(cx, cy, court_poly):
                    continue
                unique.add(int(tid))
                team = assign_team(jersey_color(frame, box, w, h))
                color = (255, 100, 100) if team == 0 else (100, 100, 255)
                cv2.rectangle(ann, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    ann,
                    f"P{tid} T{team}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )
                if kps_all is not None and i < len(kps_all):
                    kps = kps_all[i]
                    for a, b in skeleton:
                        if a < len(kps) and b < len(kps):
                            xa, ya = kps[a]
                            xb, yb = kps[b]
                            if xa > 0 and ya > 0 and xb > 0 and yb > 0:
                                cv2.line(
                                    ann,
                                    (int(xa), int(ya)),
                                    (int(xb), int(yb)),
                                    color,
                                    2,
                                )
                    for x, y in kps:
                        if x > 0 and y > 0:
                            cv2.circle(ann, (int(x), int(y)), 3, color, -1)
        out.write(ann)
        idx += 1
        if idx % 30 == 0:
            print(idx)
    cap.release()
    out.release()
    cap2 = cv2.VideoCapture(TRACKED_OUT)
    cap2.set(cv2.CAP_PROP_POS_FRAMES, 150)
    ret, sf = cap2.read()
    cv2.imwrite(SAMPLE_OUT, sf)
    cap2.release()
    print(f"Done {idx} frames, {len(unique)} unique on-court IDs -> {TRACKED_OUT}")


if __name__ == "__main__":
    main()
