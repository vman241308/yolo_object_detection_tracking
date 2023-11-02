import cv2
import sys
import time
from tracker import *

if __name__== "__main__":

    # Create tracker object
    tracker = EuclideanDistanceTracker()

    capture = cv2.VideoCapture("../Test_Video_Files/highway.mp4")

    if not capture.isOpened():
        print("Cannot open video file")
        sys.exit()

    ok, frame = capture.read()
    if not ok:
        print("Cannot read video file")
        sys.exit()

    # Object detection
    object_detector = cv2.createBackgroundSubtractorMOG2(history = 100, varThreshold = 16)
    # object_detector = cv2.createBackgroundSubtractorKNN()

    # Initialize calculating FPS
    start = time.time_ns()
    frame_count = 0
    fps = -1

    while True:
        # Read a new frame
        ok, frame = capture.read()
        if not ok:
            break

        # Increase frame count
        frame_count += 1
        # timer = cv2.getTickCount()

        # Extract region of interest
        roi = frame[340:720, 500:800]
        # roi = frame

        # 1. Object Detection
        mask = object_detector.apply(roi)
        _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for contour in contours:
            # Calculate area and remove small elements
            area = cv2.contourArea(contour)
            if area > 100:
                # cv2.drawContours(roi, [contour], -1, (0, 255, 0), 2)
                x, y, w, h = cv2.boundingRect(contour)

                detections.append([x, y, w, h])

        # 2. Object Tracking
        boxes_ids = tracker.update(detections)
        for box_id in boxes_ids:
            x, y, w, h, id = box_id
            cv2.putText(roi, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)

        # Calculate frames per second (FPS)
        # cv_fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        if frame_count >= 30:
            end = time.time_ns()
            fps = 1000000000 * frame_count / (end - start)
            frame_count = 0
            start = time.time_ns()

        # Display FPS on frame
        # cv2.putText(frame, "FPS: " + str(int(cv_fps)), (100, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
        if fps > 0:
            fps_label = "FPS: %.2f" % fps
            cv2.putText(frame, fps_label, (10, 25), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

        cv2.imshow("roi", roi)
        cv2.imshow("Frame", frame)
        cv2.imshow("Mask", mask)

        key = cv2.waitKey(30)
        if key == 27:
            break

    capture.release()
    cv2.destroyAllWindows()
