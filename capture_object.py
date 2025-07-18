# TODO: find a way to take the best picture
import cv2
import os


class Capture:
    def __init__(self, wait_frames: int=3):
        self.all_ids = []
        self.captured_ids = []
        self.id_count = {}
        self.wait = wait_frames

        self.output_path = self.make_dir()


    def __len__(self):
        return len(self.captured_ids)
    

    def make_dir(self, dir_base='results'):
        runs = 'run_' + str(len(os.listdir(dir_base)) + 1)
        output_path = os.path.join(dir_base, runs, 'objects')

        os.makedirs(output_path, exist_ok=True)

        return output_path
    

    def track_ids(self, frame, ids: list, xyxy: list):
        ids_copy = ids.copy()

        ids = [x for x in ids if x not in self.captured_ids]
        self.all_ids = list(set(self.all_ids + ids))
        
        dict_keys = self.id_count.keys()
        absent_ids = dict_keys - ids

        if absent_ids:
            for object_id in absent_ids:
                del self.id_count[object_id]
        
        for object_id in ids:
            self.id_count[object_id] = self.id_count.get(object_id, 0) + 1

            if self.id_count.get(object_id, 0) >= self.wait:
                self.captured_ids.append(object_id)
                bbox = xyxy[ids_copy.index(object_id)]

                del self.id_count[object_id]

                self.capture_object(frame, object_id, bbox, self.output_path)


    def capture_object(self, frame, object_id, bbox, output_path):
        x1, y1, x2, y2 = bbox
        cropped_frame = frame[y1: y2, x1: x2, :]

        cv2.imwrite(f'{output_path}/object_{object_id}.jpg', cropped_frame)