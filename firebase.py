from multiprocessing import Process, Queue
from firebase_admin import credentials, db
import firebase_admin
import my_secrets
from threading import Timer
import logging
from base64 import b64decode, b64encode
import os
import cv2
from datetime import datetime

# Setup logging
log_formatter = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s:%(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
# Console debug
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(log_formatter)
logger.addHandler(stream_handler)
# File logger
file_handler = logging.FileHandler(os.path.join("logs", "Google_firebase.log"))
file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.INFO)
logger.addHandler(file_handler)


class GFirebase(Process):
    """
    GFirebase extends Process
    Class connects to Google Firebase real time database
    Used to periodically uplaod a picture of parking space.
    """
    def __init__(self, q_in, queue_check_interval_s):
        super(GFirebase, self).__init__()
        # Queue for receiving picture file_paths
        self.q_in = q_in
        self.queue_check_interval_s = queue_check_interval_s

    def run(self):
        self.init_firebase()
        self.repeated_queue_check()

    def init_firebase(self):
        cred = credentials.Certificate(my_secrets.credentials_filename)
        # Initialise Firebas app
        firebase_admin.initialize_app(cred, {
            'databaseURL': my_secrets.google_firebase_link
        })
        # get reference to DB which will be used for writing
        self.fb_ref = db.reference("/")
        logger.info("Firebase initialised")

    def repeated_queue_check(self):
        stop_flag = False
        if not self.q_in.empty():
            logger.debug("Queue not empty")
            """
            stop_flag:: if True stop checking queue
            file_path:: location of image to be uploaded to Firebase
            """
            stop_flag, file_path, delete_after_upload = self.q_in.get()
            if not stop_flag:
                # If stop flag not set start analysing picture
                self.upload_file(file_path)
                if delete_after_upload:
                    logger.debug(f"Deleting after upload")
                    self.delete_file(file_path)
        if not stop_flag:
            # Repeat function if not supposed to be stopped
            self.repeated_queue_check_thread = Timer(self.queue_check_interval_s, self.repeated_queue_check)
            self.repeated_queue_check_thread.start()

    def upload_file(self, file_path):
        self.create_pic_w_time_text(file_path)
        # Get image as bytes
        img_data = open("temp_fb_pic.jpg", 'rb').read()
        # Encode data so it can be uploaded to DB
        im_b64 = b64encode(img_data).decode("utf8")
        # Upload to DB
        self.fb_ref.set({"pic": im_b64})
        self.delete_file("temp_fb_pic.jpg")
        logger.info(f"Uplaoded file to Firebase {file_path}")

    def create_pic_w_time_text(self, file_path):
        img = cv2.imread(file_path)
        self.draw_time(img, 640, 10)
        cv2.imwrite("temp_fb_pic.jpg", img)

    def draw_time(self, image, x, y):
        """
        :param image: image to draw the date on
        :param x: Space from top
        :param y: Start of text right side
        """
        date_string = self.get_time_string()
        text_size = cv2.getTextSize(date_string, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        dimensions, baseline = text_size[0], text_size[1]
        # Draw date
        cv2.putText(image, date_string, (x - dimensions[0], y + dimensions[1]), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 255), 2,
                    cv2.LINE_AA)

    def get_time_string(self):
        """
        For creation of file names
        :return: string containing time in format 2022_08_28_17_30_01
        """
        today = datetime.now()
        current_time = today.strftime("%H:%M:%S")
        return current_time

    def delete_file(self, path):
        if os.path.exists(path):
            os.remove(path)
            logger.debug(f"Deleted {path}")
        else:
            logger.error(f"No file to delete {path}")
