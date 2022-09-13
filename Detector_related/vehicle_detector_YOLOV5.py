import cv2
import logging
from multiprocessing import Process, Queue
import os
from threading import Timer
import time
import numpy as np

# Setup logging
log_formatter = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s:%(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# Console debug
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(log_formatter)
logger.addHandler(stream_handler)
# File logger
file_handler = logging.FileHandler(os.path.join("logs", "Vehicle_detector_YOLO5.log"))
file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.WARNING)
logger.addHandler(file_handler)


class VehicleDetectorYOLO5(Process):
    """
    VehicleDetector extends Process
    An object that can detect cars in pictures
    Uses Yolo5
    Done using this tutorial:
    https://learnopencv.com/object-detection-using-yolov5-and-opencv-dnn-in-c-and-python/
    On my Lenovo:
    Initialisation: 0.11s
    Analysis: 0.4s
    On my Dell:
    Initialisation: 0.07s
    Analysis: 1.8s
    """
    # Constants.
    INPUT_WIDTH = 640
    INPUT_HEIGHT = 640
    SCORE_THRESHOLD = 0.5
    NMS_THRESHOLD = 0.45
    CONFIDENCE_THRESHOLD = 0.25

    # Text parameters.
    FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.7
    THICKNESS = 1

    # Colors.
    BLACK = (0, 0, 0)
    BLUE = (255, 178, 50)
    YELLOW = (0, 255, 255)
    PICTURE_FOLDER = os.path.join(os.path.expanduser('~'), 'Documents', 'Motion_detection', 'cam_pictures',
                                  'detected_vehicles')

    def __init__(self, q_in, q_out, queue_check_interval_s, confidence_threshold):
        super(VehicleDetectorYOLO5, self).__init__()
        # How often to check queue for messages
        self.queue_check_interval_s = queue_check_interval_s
        # Delete received picture after analysing it
        self.delete_picture_after_analysis = False
        # If cars detected draw contour and save image
        self.draw_contours_and_save = True
        # Queue for sending back results to main thread
        self.q_out = q_out
        # Queue for receiving picture file_paths
        self.q_in = q_in
        # Number from 0.0 - 1.0 to set min confidence for car detection
        self.confidence_threshold = confidence_threshold

    def run(self):
        self.init_vehicle_detection()
        self.repeated_queue_check()

    def init_vehicle_detection(self):
        start_init_time = time.perf_counter()
        # Get class names
        classesFile = os.path.join("Detector_related", "detector_classes.names")
        self.classes = None
        with open(classesFile, 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')
        # Load Network
        modelWeights = os.path.join("Detector_related", "car_detect_model.onnx")
        self.net = cv2.dnn.readNet(modelWeights)
        end_init_time = time.perf_counter()
        logger.debug(f"Initialisation took {end_init_time - start_init_time}s")

    def repeated_queue_check(self):
        stop_flag = False
        if not self.q_in.empty():
            """
            stop_flag:: if True stop checking queue
            flag_return_data:: Indicates whether info about cars found in picture should be sent back via queue
            file_path:: location of image to be analysed
            """
            stop_flag, file_path, flag_return_data = self.q_in.get()
            if not stop_flag:
                # If stop flag not set start analysing picture
                self.detect_vehicles_from_file(file_path, flag_return_data)
        if not stop_flag:
            # Repeat function if not supposed to be stopped
            self.repeated_queue_check_thread = Timer(self.queue_check_interval_s, self.repeated_queue_check)
            self.repeated_queue_check_thread.start()

    def stop_queue_check(self):
        if self.repeated_queue_check_thread is not None:
            self.repeated_queue_check_thread.cancel()

    def pre_process(self, input_image):
        # Create a 4D blob from a frame.
        blob = cv2.dnn.blobFromImage(input_image, 1 / 255, (self.INPUT_WIDTH, self.INPUT_HEIGHT), [0, 0, 0], 1,
                                     crop=False)
        # Sets the input to the network.
        self.net.setInput(blob)
        # Run the forward pass to get output of the output layers.
        outputs = self.net.forward(self.net.getUnconnectedOutLayersNames())
        return outputs

    def draw_label(self, im, label, x, y):
        """Draw text onto image at location."""
        # Get text size.
        text_size = cv2.getTextSize(label, self.FONT_FACE, self.FONT_SCALE, self.THICKNESS)
        dim, baseline = text_size[0], text_size[1]
        # Use text size to create a BLACK rectangle.
        cv2.rectangle(im, (x, y), (x + dim[0], y + dim[1] + baseline), (0, 0, 0), cv2.FILLED)
        # Display text inside the rectangle.
        cv2.putText(im, label, (x, y + dim[1]), self.FONT_FACE, self.FONT_SCALE, self.YELLOW, self.THICKNESS,
                    cv2.LINE_AA)

    def post_process(self, input_image, outputs):
        # Highest confidence in detected objects
        highest_confidence = 0.0
        # Lists to hold respective values while unwrapping.
        class_ids = []
        confidences = []
        boxes = []
        # Rows.
        rows = outputs[0].shape[1]
        image_height, image_width = input_image.shape[:2]
        # Resizing factor.
        x_factor = image_width / self.INPUT_WIDTH
        y_factor = image_height / self.INPUT_HEIGHT
        # Iterate through detections.
        for r in range(rows):
            row = outputs[0][0][r]
            confidence = row[4]
            # Discard bad detections and continue.
            if confidence >= self.confidence_threshold:
                # Save highest confidence to return to user
                if confidence > highest_confidence:
                    highest_confidence = confidence
                classes_scores = row[5:]
                # Get the index of max class score.
                class_id = np.argmax(classes_scores)
                #  Continue if the class score is above threshold.
                if (classes_scores[class_id] > self.SCORE_THRESHOLD):
                    confidences.append(confidence)
                    class_ids.append(class_id)
                    cx, cy, w, h = row[0], row[1], row[2], row[3]
                    left = int((cx - w / 2) * x_factor)
                    top = int((cy - h / 2) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)
                    box = np.array([left, top, width, height])
                    boxes.append(box)
        # Perform non maximum suppression to eliminate redundant, overlapping boxes with lower confidences.
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.NMS_THRESHOLD)
        for i in indices:
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            # Draw bounding box.
            cv2.rectangle(input_image, (left, top), (left + width, top + height), self.BLUE, 3 * self.THICKNESS)
            # Class label.
            label = "{}:{:.2f}".format(self.classes[class_ids[i]], confidences[i])
            # Draw label.
            self.draw_label(input_image, label, left, top)
        return highest_confidence, input_image

    def detect_vehicles_from_file(self, file_location_name, flag_return_data):
        logger.debug(f"Detecting car in file {file_location_name}")
        start_detect_time = time.perf_counter()
        img = cv2.imread(file_location_name)
        # Process image.
        detections = self.pre_process(img)
        highest_confidence, img = self.post_process(img.copy(), detections)
        # If nothing detected, have the same file name
        new_file_name = file_location_name
        if highest_confidence > 0.0:
            # Something got detected
            if self.draw_contours_and_save:
                # file_name_old = file_location_name.rsplit('\\', 1)[1]
                file_name_old = os.path.split(file_location_name)[1]
                file_name_wo_ext = file_name_old.split('.')[0]
                logger.debug(f"file name old {file_name_old}")
                self.create_folder(self.PICTURE_FOLDER)
                new_file_name = os.path.join(self.PICTURE_FOLDER, f"{file_name_wo_ext}_cars.jpg")
                cv2.imwrite(new_file_name, img)
                logger.debug(f"Detected cars. Image file: {new_file_name}")
        if self.delete_picture_after_analysis:
            self.delete_file(file_location_name)
        if flag_return_data:
            logger.debug(f"flag_return_data is True returning: {new_file_name}")
            self.q_out.put((highest_confidence, file_location_name, new_file_name))
        end_detect_time = time.perf_counter()
        logger.debug(f"Detection took {end_detect_time - start_detect_time}s")
        logger.debug(f"Highest confidence object detected {highest_confidence} cars in picture {file_location_name}")

    def create_folder(self, path):
        if os.path.exists(path):
            logger.debug('Path exists, no need to create')
        else:
            logger.debug('Path does not exist, creating')
            os.makedirs(path)
            logger.debug('Path created')

    def delete_file(self, path):
        if os.path.exists(path):
            os.remove(path)
        else:
            logger.error("No file to delete")
