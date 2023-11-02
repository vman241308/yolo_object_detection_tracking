import cv2

# OpenCV tracker types
CV_TRACKER_BOOSTING = "CV_TRACKER_BOOSTING"
CV_TRACKER_MIL = "CV_TRACKER_MIL"
CV_TRACKER_KCF = "CV_TRACKER_KCF"
CV_TRACKER_TLD = "CV_TRACKER_TLD"
CV_TRACKER_MEDIANFLOW = "CV_TRACKER_MEDIANFLOW"
CV_TRACKER_GOTURN = "CV_TRACKER_GOTURN"
CV_TRACKER_MOSSE = "CV_TRACKER_MOSSE"
CV_TRACKER_CSRT = "CV_TRACKER_CSRT"

def compare_opencv_version(major, minor, revision):
    (current_major, current_minor, current_revision) = cv2.__version__.split(".")

    current_major = int(current_major)
    current_minor = int(current_minor)
    current_revision = int(current_revision)

    if current_major > major:
        return 1
    elif current_major < major:
        return -1

    if current_minor > minor:
        return 1
    elif current_minor < minor:
        return -1

    if current_revision > revision:
        return 1
    elif current_revision < revision:
        return -1

    return 0


class OpenCVTracker:

    def __init__(self, tracker_type = CV_TRACKER_KCF):
        self._type = tracker_type

        # if int(minor_ver) < 3:
        if compare_opencv_version(4, 3, 0) < 0:
            self._tracker = cv2.Tracker_create(tracker_type)
        else:
            if tracker_type == CV_TRACKER_BOOSTING:
                if compare_opencv_version(4, 5, 1) <= 0:
                    self._tracker = cv2.TrackerBoosting_create()
                else:
                    self._tracker = cv2.legacy.upgradeTrackingAPI(cv2.legacy.TrackerBoosting_create())

            if tracker_type == CV_TRACKER_MIL:
                self._tracker = cv2.TrackerMIL_create()

            if tracker_type == CV_TRACKER_KCF:
                self._tracker = cv2.TrackerKCF_create()

            if tracker_type == CV_TRACKER_TLD:
                if compare_opencv_version(4, 5, 1) <= 0:
                    self._tracker = cv2.TrackerTLD_create()
                else:
                    self._tracker = cv2.legacy.upgradeTrackingAPI(cv2.legacy.TrackerTLD_create())

            if tracker_type == CV_TRACKER_MEDIANFLOW:
                if compare_opencv_version(4, 5, 1) <= 0:
                    self._tracker = cv2.TrackerMedianFlow_create()
                else:
                    self._tracker = cv2.legacy.upgradeTrackingAPI(cv2.legacy.TrackerMedianFlow_create())

            if tracker_type == CV_TRACKER_GOTURN:
                self._tracker = cv2.TrackerGOTURN_create()

            if tracker_type == CV_TRACKER_MOSSE:
                if compare_opencv_version(4, 5, 1) <= 0:
                    self._tracker = cv2.TrackerMOSSE_create()
                else:
                    self._tracker = cv2.legacy.upgradeTrackingAPI(cv2.legacy.TrackerMOSSE_create())

            if tracker_type == CV_TRACKER_CSRT:
                self._tracker = cv2.TrackerCSRT_create()

    def init(self, image, bounding_box):
        return self._tracker.init(image, bounding_box)

    def update(self, image):
        return self._tracker.update(image)

    def type(self):
        return self._type
