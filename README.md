PEOPLE DETECTION & TRACKING

This is a small demonstration project showing real-time people detection, tracking, and basic analysis using Ultralytics YOLO.

---

## Features:
- Detects people in a video or webcam stream
- Tracks each person with a unique ID using ByteTrack
- Saves cropped images of detected people
- Generates an annotated video with bounding boxes and IDs
- Each run saves results in a structured folder:
  results/run_1/
      objects/    # saved cropped person images
      video/      # annotated output video

---

## INSTALLATION

https://github.com/karibays/People-Detection-Tracking.git
cd people-tracking-demo
pip install -r requirements.txt

You need:
- Python 3.8+
- OpenCV
- Ultralytics

---

## USAGE

1) Process a video file

python process_video.py --input asserts/test2.webm

This will:
- Run YOLO tracking on test2.webm
- Save cropped person images under results/run_X/objects/
- Save the annotated video as results/run_X/video/test.mp4

2) Webcam live tracking

python live_webcam.py

Press "q" to quit.

---

## PROJECT STRUCTURE

people-tracking-demo/
- capture_object.py   - Capture class that saves person crops
- process_video.py    - Processes a video and saves annotated result
- live_webcam.py      - Optional live webcam demo
- results/            - Output (created automatically)
- README.md

---

## EXAMPLE RESULTS

Cropped person images:
    results/run_1/objects/object_3.jpg
    results/run_1/objects/object_7.jpg

Annotated video:
    results/run_1/video/test2.mp4

---

## FUTURE IMPROVEMENTS

- Pick the best frame for each person (sharpest/largest bbox)
- Add CSV/JSON export of tracked IDs
- Analyze model predictions
- Person sex classification
- Person age prediction
- Person nation prediction

---

## LICENSE

MIT – feel free to use and modify.
