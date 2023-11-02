import cv2
import time
import sys
from SiamRPNpp_tracker import SiamRPNppTracker

sys.path.append("..")
from YOLO_Detector_OpenCV_Tracker_Python.YOLO_detector import *

# Main function
if __name__== "__main__":

    is_cuda = len(sys.argv) > 1 and sys.argv[1] == "cuda"

    yolo_detector = YoloDetector(YOLO_V8, is_cuda)

    tracker = None

    capture = cv2.VideoCapture("../Test_Video_Files/road.mp4")

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

    # Set mouse click callback function
    object_clicked = False
    object_selected = False
    ox, oy = -1, -1
    selected_object_bounding_box = None

    def mouse_click(event, x, y, flags, param):
        global ox, oy, object_clicked, object_selected
        if event == cv2.EVENT_LBUTTONDOWN:
            ox, oy = x, y
            object_selected = False
            object_clicked = True

    cv2.namedWindow("Frame")
    cv2.setMouseCallback("Frame", mouse_click)

    while True:
        # Read a new frame
        ok, frame = capture.read()
        if not ok:
            break

        if frame is None:
            break

        frame_count += 1

        if object_clicked or object_selected:
            if object_clicked and not object_selected:
                selected_object_bounding_box = None

                tracker = SiamRPNppTracker(is_cuda)

                class_ids, class_names, confidences, boxes = yolo_detector.apply(frame)

                for (class_id, class_name, confidence, box) in zip(class_ids, class_names, confidences, boxes):
                    x, y, w, h = box
                    if ox >= x and ox <= x + w and oy >= y and oy <= y + h:
                        tracker.init(frame, box)
                        selected_object_bounding_box = box
                        break

                object_selected = True
                object_clicked = False

            if object_selected:
                if not (selected_object_bounding_box is None):
                    output = tracker.track(frame)
                    selected_object_bounding_box = list(map(int, output['bbox']))

            if not (selected_object_bounding_box is None):
                color = (0, 0, 255)
                label = "%s (%d%%)" % (class_name, int(confidence * 100))

                box = selected_object_bounding_box
                cv2.rectangle(frame, box, color, 2)
                cv2.rectangle(frame, (box[0], box[1] - 20), (box[0] + box[2], box[1]), color, -1)
                cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0))

        if frame_count >= 30:
            end = time.time_ns()
            fps = 1000000000 * frame_count / (end - start)
            frame_count = 0
            start = time.time_ns()

        if fps > 0:
            fps_label = "FPS: %.2f" % fps
            cv2.putText(frame, fps_label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Frame", frame)

        key = cv2.waitKey(30)
        if key == 27:
            break

    capture.release()
    cv2.destroyAllWindows()
