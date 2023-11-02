import math

class EuclideanDistanceTracker:

    def __init__(self):
        # Store the center positions of the objects
        self.center_points = {}

        # Keep the count of the IDs
        # Each time a new object is detected, the count will increase by one
        self.id_count = 0

    def update(self, object_boxes):
        # Object boxes and ids
        object_boxes_ids = []

        # Get center point of new object
        for object_box in object_boxes:
            x, y, w, h = object_box
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            # Find out if that object was detected already
            same_object_detected = False
            for id, center_point in self.center_points.items():
                distance = math.hypot(cx - center_point[0], cy - center_point[1])

                if distance < 25:
                    self.center_points[id] = (cx, cy)

                    # print(self.center_points)

                    object_boxes_ids.append([x, y, w, h, id])
                    same_object_detected = True
                    break

            # New object is detected, we assign the ID to that object
            if same_object_detected is False:
                self.center_points[self.id_count] = (cx, cy)
                object_boxes_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1

        # Clean the dictionary by center points to remove IDs not used anymore
        new_center_points = {}
        for object_box_id in object_boxes_ids:
            _, _, _, _, object_id = object_box_id
            center_point = self.center_points[object_id]
            new_center_points[object_id] = center_point

        # Update dictionary with IDs not used removed
        self.center_points = new_center_points.copy()

        return object_boxes_ids
