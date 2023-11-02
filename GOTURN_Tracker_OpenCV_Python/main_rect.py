import cv2
import sys
import time
from tracker import *

if __name__ == "__main__":

    # Open video file
    capture = cv2.VideoCapture("../Test_Video_Files/cars.mp4")

    if not capture.isOpened():
        print("Cannot open video file")
        sys.exit()

    ok, frame = capture.read()
    if not ok:
        print("Cannot read video file")
        sys.exit()

    # Initialize calculating FPS
    start = time.time_ns()
    frame_count = 0
    fps = -1

    # Define an initial bounding box
    bounding_box = (10, 10, 100, 100)

    # Select a bounding box
    bounding_box = cv2.selectROI(frame, False)

    # Create a OpenCV tracker and Initialize tracker with first frame and bounding box
    tracker = OpenCVTracker()
    ok = tracker.init(frame, bounding_box)

    while True:
        # Read a new frame
        ok, frame = capture.read()
        if not ok:
            break

        # Increase frame count
        frame_count += 1

        # Update tracker
        ok, bounding_box = tracker.update(frame)

        # Draw bounding box
        if ok:
            # Tracking success
            p1 = (int(bounding_box[0]), int(bounding_box[1]))
            p2 = (int(bounding_box[0] + bounding_box[2]), int(bounding_box[1] + bounding_box[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        else:
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # Calculate Frames per second (FPS)
        if frame_count >= 30:
            end = time.time_ns()
            fps = 1000000000 * frame_count / (end - start)
            frame_count = 0
            start = time.time_ns()

        # Display tracker type on frame
        cv2.putText(frame, tracker.type(), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        # Display FPS on frame
        if fps > 0:
            fps_label = "FPS: %.2f" % fps
            cv2.putText(frame, fps_label, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        # Display result
        cv2.imshow("Tracking", frame)

        # Exit if ESC pressed
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
