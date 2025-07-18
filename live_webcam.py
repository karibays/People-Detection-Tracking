import cv2
import os
from ultralytics import YOLO
from capture_object import Capture


def process_video(model_path="model/best.pt", tracker_cfg="bytetrack.yaml"):
    model = YOLO(model_path)
    capture = Capture()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(f"❌ Cannot open webcam")
        return

    # Video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_file = 'webcam.mp4'
    output_path = os.path.join(os.path.dirname(capture.output_path), 'video', output_file)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # or 'XVID'
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"🎥 Processing webcam")
    print(f"📏 Resolution: {width}x{height}, FPS: {fps}, Frames: {total_frames}")

    frame_idx = 0
    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model.track(frame, persist=True, tracker=tracker_cfg, verbose=False)

        annotated_frame = results[0].plot()

        ids = results[0].boxes.id.int().tolist() if results[0].boxes.id is not None else []
        xyxy = results[0].boxes.xyxy.int().tolist()
        people_count = len(ids)

        capture.track_ids(frame, ids, xyxy)

        cv2.putText(
            annotated_frame,
            f"People: {people_count}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            2
        )

        cv2.imshow('frame', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        out.write(annotated_frame)

        frame_idx += 1
        if frame_idx % fps == 0:
            print(f"✅ Processed {frame_idx}/{total_frames} frames")

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"✅ Done! Saved output to: {output_path}")

if __name__ == "__main__":
    model_weights = "model/best.pt"

    process_video(model_weights)