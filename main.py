import os
import queue
import time
import cv2
from video_file_manager import FileManager
from cam_recorder_process import MovementRecorder
from threading import Timer
from multiprocessing import Queue
from datetime import datetime
import smtplib, ssl
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import logging
from Detector_related.vehicle_detector_YOLOV5 import VehicleDetectorYOLO5
from tkinter import Tk, Label, Button, StringVar, Entry, Frame
import schedule
import my_secrets
from firebase import GFirebase

# Setup logging
log_formatter = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s:%(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)
# Console debug
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(log_formatter)
logger.addHandler(stream_handler)

# File logger
file_handler = logging.FileHandler(os.path.join("logs", "main.log"))
file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.INFO)
logger.addHandler(file_handler)


# TODO:
# send frames to webserver only if it is opened
# Firebeas has no file to delete err
# Handle low framerate when low loght - turn off auto light, stop recording or something
# CV window inside of tkinter?
# test and Train the model more if needed
# Add default sensitivity in UI
# Add car detection sensitivity setting to UI
# Install google drive for computer used
# Stream to web browser?

# Start program by craeting object of MainClass
def main():
    main = MainUIClass()


# Mainclass extends tkinter for creation of UI
class MainUIClass(Tk):
    CAM_TO_USE = 1
    # Folder to save recordings in
    PICTURE_FOLDER = os.path.join(os.path.expanduser('~'), 'Documents', 'Motion_detection', 'cam_pictures')
    # How often to check for a car. Takes some processing power, do not set too often.
    CHECK_FOR_CAR_INTERVAL_S = 10.0
    # How often to save the non analysed pictures
    SAVE_PARKING_IMAGE_INTERVAL_S = 20.0
    # How often to check folder to limit its side
    CHECK_FOLDER_SIZE_INTERVAL = 3600.0
    DELETE_IF_FOLDER_LARGER_THAN_GB = 10.00
    # How much files will be deleted in one go if folder exceed folder size limit
    DELETE_SIZE_GB = 1.0
    # Delete all files smaller than 1 MB. Most likely interupted recordings
    MIN_FILE_SIZE_GB = 0.001
    # UI constants
    BTN_WIDTH = 60
    # Size of element in a row with 3 elements
    LABEL_ENTRY_SET_SIZE = 20
    # How often should the status Queue be checked from the cam recorder process
    MAINLOOP_OTHER_INTERVAL_MS = 500
    # When to send daily email with picture and car detection
    TIME_OF_MAIL = "17:05"
    # Save images with rectangles drawn around cars
    SAVE_DETECTED_CAR_IMAGES = False
    # Images with no analysis. Save for training of model
    SAVE_PARKING_IMAGES = True

    def __init__(self):
        super().__init__()
        logger.info("Program started")
        self.list_cams()
        self.file_location_front = os.path.join(os.path.expanduser('~'), 'Documents', 'Motion_detection',
                                                'recordings_front')
        self.file_location_web = os.path.join(os.path.expanduser('~'), 'Documents', 'Motion_detection',
                                              'recordings_web')
        self.original_location = os.path.join(os.path.expanduser('~'), 'Documents', 'Motion_detection', 'recordings')
        self.folder_list = [self.file_location_front, self.file_location_web, self.original_location]
        # Que to send requests to front camera process
        self.q_front_send = Queue()
        # Que to send requests to web camera process
        self.q_web_send = Queue()
        # For getting pictures from cams to main theread
        self.q_return = Queue()
        # For getting status of webcams
        self.q_status = Queue()
        # Queue for receiving data from vehicle detection process
        self.q_veh_detect_receive = Queue()
        # Queue for sending data from vehicle detection process
        self.q_veh_detect_send = Queue()
        # Queue for sending pictures for upload to google firebase
        self.q_fb = Queue()
        # Object that will upload images to Google Firebase realtime Database
        self.fb_db = GFirebase(q_in=self.q_fb, queue_check_interval_s=1.0)
        self.fb_db.start()
        # Object that can detect cars
        self.vehicle_detector = VehicleDetectorYOLO5(q_in=self.q_veh_detect_send, q_out=self.q_veh_detect_receive,
                                                     queue_check_interval_s=0.5, confidence_threshold=0.3)
        self.vehicle_detector.start()
        # For my DELL laptop: camera 2 = web cam, camera 0 = front cam
        self.recorder_1 = MovementRecorder(q_in=self.q_front_send, q_out=self.q_return, q_status=self.q_status,
                                           cam_to_use_nr=self.CAM_TO_USE, fps=10,
                                           file_location=self.file_location_front,
                                           stream=True)
        # self.recorder_web = MovementRecorder(q_in=q_web_send, q_out=q_return, cam_to_use_nr=2, fps=10, file_location=file_location_web)
        self.folder_mngmnt_thread = None
        # Repeatedly check folder of saved videos for files too small and total folder size
        self.start_repeated_folder_mngmnt()
        # Indicates that the camera is starting up and car detection should not be executed yet
        self.startup = True
        # Thread for repeated car detection
        self.car_detection_thread = None
        # Check for cars in video in set time intervals
        self.start_repeated_car_detection()
        # Start recording with front camera
        self.recorder_1.start()
        # recorder_web.start()
        # Indicates that the picture that was saved should be analysed
        self.flag_check_saved_pic_for_car = False
        # Indicates that after car detection done a picture should be sent via email
        self.flag_send_car_detection_mail = False
        # Indicates that the mainprogram should check for returned picture from vehicle detector
        self.flag_check_veh_detector_queue = False
        self.time_of_last_parking_image_save = 0
        self.set_up_scheduled_picture_analysis(self.TIME_OF_MAIL)
        self.set_up_ui()

    def mainloop_user(self):
        """
        Periodically called function for doing tasks other than IO
        :return:
        """
        # For sending out of scheduled email
        schedule.run_pending()
        # Update UI with status of cam recorder
        self.check_status()
        # If started check queue of vehicle detection process for result of picture analysis
        if self.flag_check_veh_detector_queue:
            self.check_veh_detec_queue()
        # Start the loop again after delay
        self.after(self.MAINLOOP_OTHER_INTERVAL_MS, self.mainloop_user)

    def set_up_ui(self):
        self.protocol("WM_DELETE_WINDOW", self.save_and_finish)
        self.title('PARKING MONITOR')
        self.prepare_ui_elements()
        self.place_ui_elements()
        self.after(self.MAINLOOP_OTHER_INTERVAL_MS, self.mainloop_user)
        self.mainloop()

    def set_up_scheduled_picture_analysis(self, time_of_event):
        """
        Detect cars every day at certain time. At usual time before ariving home from work.
        :param time_of_event: Time of event each day
        :return:
        """
        schedule.every().day.at(time_of_event).do(self.detect_car_and_send_mail)

    def start_repeated_car_detection(self):
        """
        Periodically detect car in captured pictures
        :return:
        """
        if not self.startup:
            # Actually detect cars only on second and further calls
            self.request_detect_car()
        else:
            self.startup = False
        # Execute this same function in regular intervalse
        self.car_detection_thread = Timer(self.CHECK_FOR_CAR_INTERVAL_S, self.start_repeated_car_detection)
        self.car_detection_thread.start()

    def detect_car_and_send_mail(self):
        """
        Detect car in picture from cam recorder and send email with it
        :return:
        """
        logger.debug("detect_car_and_send_mail")
        self.flag_send_car_detection_mail = True
        self.request_detect_car()

    def request_detect_car(self):
        """
        Detect cars in cam recorder picture
        :return:
        """
        if not self.flag_check_saved_pic_for_car:
            # Set flag so when the picture is received the program knows to analyse it
            self.flag_check_saved_pic_for_car = True
            self.get_and_analyse_pic()
        else:
            logger.info("Already detecting car")

    def stop_vehicle_detector(self):
        """
        Stop vehicle detection process by setting the stop flag via Queue
        :return:
        """
        self.q_veh_detect_send.put((True, "", False))

    def check_status(self):
        """
        Get status from cam recorder process and display it in the UI
        :return:
        """
        try:
            status = self.q_status.get(True, 0)
            if status == MovementRecorder.STATUS_INIT:
                self.lbl_status.config(text="Initialising")
            elif status == MovementRecorder.STATUS_RECORDING:
                self.lbl_status.config(text="Recording")
            elif status == MovementRecorder.STATUS_WAITING_ON_MOVEMEMENT:
                self.lbl_status.config(text="Looking for movement")
            elif status == MovementRecorder.STATUS_ERROR:
                self.lbl_status.config(text="Error")
        except queue.Empty:
            pass

    def prepare_ui_elements(self):
        """
        Create UI elements
        :return:
        """
        self.lbl_status = Label(self, text='Initialising web cam')
        self.btn_take_pic = Button(self, text='TAKE PIC', command=self.get_and_save_pic, width=self.BTN_WIDTH)
        self.btn_mail_pic = Button(self, text='MAIL PIC', command=self.detect_car_and_send_mail, width=self.BTN_WIDTH)
        self.btn_toggle_video = Button(self, text='TOGGLE VIDEO', command=self.toggle_video, width=self.BTN_WIDTH)
        self.btn_test_detection = Button(self, text='CAR DETECTION', command=self.request_detect_car,
                                         width=self.BTN_WIDTH)
        self.frame_sens_input = Frame(self)
        self.lbl_sens = Label(self.frame_sens_input, text="Sensitivity: (5000)", width=self.LABEL_ENTRY_SET_SIZE)
        self.entry_sens_value = StringVar()
        self.entry_sens = Entry(self.frame_sens_input, width=self.LABEL_ENTRY_SET_SIZE,
                                textvariable=self.entry_sens_value)
        self.btn_set_sens = Button(self.frame_sens_input, text='SET', command=self.set_sensitivity,
                                   width=self.LABEL_ENTRY_SET_SIZE)

    def place_ui_elements(self):
        """
        Place created UI elements
        :return:
        """
        # Sens frame grid
        self.lbl_sens.grid(row=0, column=0)
        self.entry_sens.grid(row=0, column=1)
        self.btn_set_sens.grid(row=0, column=2)
        # Main grid
        self.lbl_status.grid(row=0, column=0)
        self.btn_take_pic.grid(row=1, column=0)
        self.btn_mail_pic.grid(row=2, column=0)
        self.btn_toggle_video.grid(row=3, column=0)
        self.btn_test_detection.grid(row=4, column=0)
        self.frame_sens_input.grid(row=5, column=0)

    def set_sensitivity(self):
        """
        Set sensitivity of movement detection
        :return:
        """
        self.q_front_send.put(f"sens_{self.entry_sens_value.get()}")

    def get_and_save_pic(self):
        picture = self.get_picture_from_cam_recorder_queue()
        self.save_pic_in_pic_folder(picture)
        self.time_of_last_parking_image_save = time.perf_counter()

    def get_and_analyse_pic(self):
        """
        Get picture from cam recorder and save it
        :return:
        """
        picture = self.get_picture_from_cam_recorder_queue()
        full_file_path = self.save_pic_in_pic_folder(picture)
        if full_file_path is not None:
            if self.flag_check_saved_pic_for_car:
                # If set the saved picture should be analysed to detect cars in it
                logger.debug(f"Starting detection of cars {full_file_path}")
                self.start_detection_of_cars(full_file_path)
                self.flag_check_saved_pic_for_car = False

    def save_pic_in_pic_folder(self, pic):
        """
        Save picture which should be in queue of cam_recorder
        pic = picture to save
        :return: path of picture saved None if there is nothing
        """
        if pic is not None:
            self.create_folder(self.PICTURE_FOLDER)
            file_name = f"{self.get_time_date_string()}.jpg"
            full_file_path = os.path.join(self.PICTURE_FOLDER, file_name)
            cv2.imwrite(full_file_path, pic)
            logger.debug(f"Successfully saved image {full_file_path}")
            return full_file_path
        return None

    def start_detection_of_cars(self, picture_path):
        """
        Put request data in queue for the vehicle detection object
        :param picture_path: Path of picture to analyse
        """
        #  stop_flag, file_path, flag_return_data
        self.q_veh_detect_send.put((False, picture_path, True))
        self.flag_check_veh_detector_queue = True

    def check_veh_detec_queue(self):
        """
        Check if cehicle detection theread has finished analysing picture
        :return:
        """
        logger.debug("Checking vehicle detector queue")
        try:
            self.flag_check_veh_detector_queue = False
            # If nothing detected analysed_pic_loc will be equal to original_loc
            highest_confidence, original_loc, analysed_pic_loc = self.q_veh_detect_receive.get(True, 0)
            car_detected = True if highest_confidence > 0.0 else False
            self.send_car_detection_mail(car_detected, analysed_pic_loc, highest_confidence)
            delete_after_upload = False
            if car_detected:
                # Car detected, 2 images - one with detected cars and the original
                if not self.SAVE_DETECTED_CAR_IMAGES:
                    # Delete image after uploading to FB depending on setting
                    delete_after_upload = True
                if not self.time_to_save_parking_picture():
                    # Delete original depending on setting and timing
                    logger.debug(f"Deleting original file because not yet time to save")
                    self.delete_file(original_loc)
            else:
                # Car not detected, only one image file, so if it should be deleted, then after upload to FB
                delete_after_upload = not self.time_to_save_parking_picture()
            # Upload picture to firebase
            self.q_fb.put((False, analysed_pic_loc, delete_after_upload))
            logger.debug(
                f"check_veh_detec_queue: highest_confidence {highest_confidence} picture_location{analysed_pic_loc}")
        except queue.Empty:
            logger.debug("Vehicle detector queue empty")

    def send_car_detection_mail(self, car_detected, pic_location, highest_confidence):
        if self.flag_send_car_detection_mail:
            # Reset flag
            self.flag_send_car_detection_mail = False
            if car_detected:
                self.send_pic_in_mail(pic_location, email_subject=f"Car detected, confidence "
                                                                      f"{highest_confidence}", email_text="blank")
            else:
                self.send_pic_in_mail(pic_location, email_subject=f"No cars detected", email_text="blank")

    def time_to_save_parking_picture(self):
        """
        :return: True if parking image should not be deleted
                False if it should be deleted
        """
        logger.debug(f"time_to_save_parking_picture")
        if self.SAVE_PARKING_IMAGES:
            current_time = time.perf_counter()
            time_passed_since_last_save = current_time - self.time_of_last_parking_image_save
            logger.debug(f"time_passed_since_last_save = {time_passed_since_last_save}")
            if time_passed_since_last_save >= self.SAVE_PARKING_IMAGE_INTERVAL_S:
                logger.debug("Do not delete original image")
                self.time_of_last_parking_image_save = current_time
                return True
            else:
                logger.debug("Deleting original image")
                return False
        else:
            return False

    def get_picture_from_cam_recorder_queue(self):
        # Request picture
        self.q_front_send.put("pic")
        try:
            pic = self.q_return.get(True, 1)
        except queue.Empty:
            logger.error("Unable to get temp picture from recorder")
            return None
        return pic

    def toggle_video(self):
        """
        Toggle displaying of video by sending a command to cam recorder process
        :return:
        """
        self.q_front_send.put("video")

    def save_and_finish(self):
        """
        User has closed the tkinter UI, shut down program
        :return:
        """
        self.stop_car_detection_thread()
        self.stop_repeated_folder_mngmnt()
        self.stop_vehicle_detector()
        # Stop recording, save file if recording currently in progress
        self.q_front_send.put("stop")
        logger.debug("Before join")
        # Wait for cam recorder process to stop
        self.recorder_1.join()
        logger.info("Exiting program by user press")
        # Close firebase process by setting stop flag
        self.q_fb.put((True, None, False))
        # Close tkinter UI
        self.destroy()

    def start_repeated_folder_mngmnt(self):
        """
        Repeatedly check folders which hold the saved files and check them for:
        Small junk files
        Empty folders
        Limit folder size
        :return:
        """
        mngr = FileManager()
        # Check folder where pictures are saved
        mngr.delete_empty_subfolders(self.PICTURE_FOLDER)
        mngr.limit_folder_size(self.PICTURE_FOLDER, self.DELETE_IF_FOLDER_LARGER_THAN_GB, self.DELETE_SIZE_GB,
                               delete_files_smaller_than_gb=0.000001)
        for folder in self.folder_list:
            mngr.delete_empty_subfolders(folder)
            mngr.limit_folder_size(folder, self.DELETE_IF_FOLDER_LARGER_THAN_GB, self.DELETE_SIZE_GB,
                                   self.MIN_FILE_SIZE_GB)
        # Repeat action in regular intervals
        self.folder_mngmnt_thread = Timer(self.CHECK_FOLDER_SIZE_INTERVAL, self.start_repeated_folder_mngmnt)
        self.folder_mngmnt_thread.start()

    def stop_repeated_folder_mngmnt(self):
        if self.folder_mngmnt_thread is not None:
            self.folder_mngmnt_thread.cancel()

    def stop_car_detection_thread(self):
        if self.car_detection_thread is not None:
            self.car_detection_thread.cancel()

    def send_pic_in_mail(self, pic_path, email_subject, email_text):
        """
        Send picture via email
        :param pic_path: Path of saved picture
        :param email_subject: Subject of email
        :param email_text: Body of email
        :return:
        """
        # Create a multipart message and set headers
        message = MIMEMultipart()
        message["From"] = my_secrets.sender_email
        message["To"] = 'briedisivars@gmail.com'
        message["Subject"] = email_subject
        # Add body to email
        message.attach(MIMEText(email_text, "plain"))
        img_data = open(pic_path, 'rb').read()
        part = MIMEBase("application", "octet-stream")
        part.set_payload(img_data)
        # Encode file in ASCII characters to send by email
        encoders.encode_base64(part)
        # Add header as key/value pair to attachment part
        part.add_header(
            "Content-Disposition",
            f"attachment; filename= pic.jpg",
        )
        # Add attachment to message and convert message to string
        message.attach(part)
        text = message.as_string()
        # Create a secure SSL context
        context = ssl.create_default_context()
        server = None
        # Try to log in to server and send email
        try:
            server = smtplib.SMTP(my_secrets.smtp_server, my_secrets.port)
            server.starttls(context=context)  # Secure the connection
            server.login(my_secrets.sender_email, my_secrets.password)
            server.sendmail(my_secrets.sender_email, 'briedisivars@gmail.com', text)
            logger.info("Successfully sent email with picture")
        except Exception as e:
            # Print any error messages to stdout
            logger.exception("Failed to send mail")
        finally:
            if server is not None:
                server.quit()

    def create_folder(self, path):
        """
        Creates folder if it already does not exist
        :param path: folder path
        """
        if os.path.exists(path):
            logger.debug(f"Path {path} exists, no need to create")
        else:
            logger.debug(f"Path {path} does not exist, creating")
            os.makedirs(path)

    def get_time_date_string(self):
        """
        For creation of file names
        :return: string containing time in format 2022_08_28_17_30_01
        """
        today = datetime.now()
        current_date = today.strftime("%Y_%m_%d")
        current_time = today.strftime("%H_%M_%S")
        string = f"{current_date}_{current_time}"
        return string

    def list_cams(self):
        """
        Test available cameras and print out results
        """
        non_working_ports = []
        dev_port = 0
        working_ports = []
        available_ports = []
        while len(non_working_ports) < 3:  # if there are more than 5 non working ports stop the testing.
            camera = cv2.VideoCapture(dev_port)
            if not camera.isOpened():
                non_working_ports.append(dev_port)
                logger.debug("Port %s is not working." % dev_port)
            else:
                is_reading, img = camera.read()
                w = camera.get(3)
                h = camera.get(4)
                if is_reading:
                    logger.debug("Port %s is working and reads images (%s x %s)" % (dev_port, h, w))
                    working_ports.append(dev_port)
                else:
                    logger.debug("Port %s for camera ( %s x %s) is present but does not read." % (dev_port, h, w))
                    available_ports.append(dev_port)
            dev_port += 1
        return available_ports, working_ports, non_working_ports

    def delete_file(self, path):
        if os.path.exists(path):
            os.remove(path)
            logger.debug(f"Deleted {path}")
        else:
            logger.error("No file to delete")


if __name__ == '__main__':
    main()
