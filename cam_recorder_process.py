import cv2
import time
from datetime import datetime
from multiprocessing import Process, Queue
import os
import logging
from webstream import SurveillanceWebStream

""" 
For decent camera speed. Now implemented in code
v4l2-ctl -d 2 -c gain_automatic=0
"""

# Setup logging
log_formatter = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s:%(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)
# Console debug
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(log_formatter)
logger.addHandler(stream_handler)
# File logger
file_handler = logging.FileHandler(os.path.join("logs", "cam_recorder.log"))
file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.INFO)
logger.addHandler(file_handler)


class MovementRecorder(Process):
    """
    Class MovementRecorder extends Process
    Class that reads frames from a webcam using opencv and detects movement. Saves video files if movement
    detected.
    """
    # How many seconds after no movement registered stop recording
    STOP_RECORDING_DLY_S = 60.0
    # After how many frames print out the average FPS
    FRAME_TIME_PRINTOUT_INTERVAL_FRAMES = 1000
    # Pront log statement of every frames FPS
    LOG_EVERY_PRINT_TIME = False
    # Statuses to return to main
    STATUS_ERROR = 0
    STATUS_INIT = 1
    STATUS_WAITING_ON_MOVEMEMENT = 2
    STATUS_RECORDING = 3

    def __init__(self, q_in, q_out, q_status, cam_to_use_nr, fps, file_location, disable_automatic_gain=False
                 , stream=True):
        """
        :param q_in: Queue for receiving requests from main
        :param q_out: Q for returning results to main
        :param q_status: Q for returning status to main
        :param cam_to_use_nr: Which camera to use
        :param fps: Target frames per second
        :param file_location: Where to save the recordings to
        :param disable_automatic_gain: If web cam cannot get good frames this can be enabled
        :param stream: If True video is streamed to web browser
        """
        super(MovementRecorder, self).__init__()
        self.disable_automatic_gain = disable_automatic_gain
        # Stop recording
        self.stream = stream
        self.stop_flag = False
        self.flag_return_pic = False
        self.flag_show_video = True
        self.flag_detect_cars = True
        self.cam_nr = cam_to_use_nr
        self.fps = fps
        self.file_location = file_location
        self.frame_time = 1.0 / fps
        # Queues for exchanging data with main
        self.q_in = q_in
        self.q_out = q_out
        self.q_status = q_status
        # Current and previous frame to detect movement
        self.current_frame = None
        self.previous_frame = None
        self.frame_to_display = None
        # For calculating FPS:
        self.current_frame_time = 0.0
        self.previous_frame_time = 0.0
        self.recording_active = False
        # To time when to stop recording
        self.last_movement_time = 0.0
        # video out stream
        self.v_out = None
        self.show_video_recording = True
        # Draw recording around movement
        self.draw_contours = True
        self.avg_frame_time = 0.0
        self.nr_of_frames_taken = 0
        self.current_file_name = "none"
        self.car_detected = False
        self.display_fps = True
        # Determines sensetivity of movement detection. Smaller values more sensitive. Default 5000
        self.frame_area_for_movement = 5000
        # Variables for streaming to webserver
        self.web_q_in: Queue
        self.web_q_out: Queue
        self.webserver: SurveillanceWebStream
        self.web_q_in = None
        self.web_q_out = None
        self.webserver = None
        # If True, a video frame must be sent to webserver thread
        self.streaming_active = False
        # Used to reset  the stream_active flag
        self.time_since_last_stream_frame_req = 0.0

    def run(self):
        # Inform main that camera initialising
        self.q_status.put(self.STATUS_INIT)
        camera_ok = False
        # might be needed so external web camera is not lagging
        if self.disable_automatic_gain:
            os.system(f"v4l2-ctl -d {self.cam_nr} -c gain_automatic=0")
        # Set camera on openCV object
        self.capture = cv2.VideoCapture(self.cam_nr)
        # Check if camera working
        if not self.capture.isOpened():
            logger.error(f"CAM{self.cam_nr}: Failed to init camera")
        else:
            logger.info(f"CAM{self.cam_nr}: Camera initialized successfully")
            camera_ok = True
        # Set FPS
        # self.capture.set(cv2.CAP_PROP_FPS, 30)
        self.capture.set(cv2.CAP_PROP_FPS, self.fps)
        # Get picture size
        self.frame_size = (int(self.capture.get(3)), int(self.capture.get(4)))
        logger.info(f"CAM{self.cam_nr}: Target FPS: {self.fps} Frame time: {self.frame_time}")
        logger.info(f"CAM{self.cam_nr}: Frame size: {self.frame_size}")
        if camera_ok:
            # Initialise streaming to webserver
            if self.stream:
                logging.debug("Stream is True initiating webserver")
                self.web_q_in = Queue()
                self.web_q_out = Queue()
                self.webserver = SurveillanceWebStream(self.web_q_in, self.web_q_out)
                self.webserver.start()
            # Camera initialised successfully, start recording
            self.record_on_movement()
            # self.display_video_only()
        else:
            # Did not manage to initialise camera, close OpenCV object and stop.
            self.q_status.put(self.STATUS_ERROR)
            # Unable to init camera
            self.capture.release()

    def check_queue(self):
        """
        Check Queue to get commands from main
        """
        if not self.q_in.empty():
            # Command present in Queue
            cmd = self.q_in.get()
            if isinstance(cmd, str):
                if cmd == "stop":
                    # Main stops video recording
                    logger.debug(f"CAM{self.cam_nr}Stop in queue")
                    # Set stop bit for web server thread
                    self.web_q_in.put((True, None))
                    self.stop_flag = True
                elif cmd == "pic":
                    # Main requests picture to be returned
                    logger.debug(f"CAM{self.cam_nr}Retunr pic in queue")
                    self.flag_return_pic = True
                elif cmd == "video":
                    # Main toggles video display
                    self.flag_show_video = not self.flag_show_video
                    if not self.flag_show_video:
                        cv2.destroyAllWindows()
                        logger.debug(f"CAM{self.cam_nr} Turning video off")
                    else:
                        logger.debug(f"CAM{self.cam_nr} Turning video on")
                elif cmd[:4] == "sens":
                    # Main changes sensitivity
                    logger.debug(f"CAM{self.cam_nr} Sense CMD {cmd}")
                    try:
                        sens_string = cmd.split("_")[1]
                        sens_int = int(sens_string)
                        logger.info(f"CAM{self.cam_nr} New sensitivity is {sens_int}")
                        self.frame_area_for_movement = sens_int
                    except:
                        logger.error(f"CAM{self.cam_nr} Received non int sensitivity")
                else:
                    logger.error(f"CAM{self.cam_nr} Unknown command in queue {cmd}")
            else:
                logger.error(f"CAM{self.cam_nr} Non string in queue.")

    def frame_rate_logging(self):
        """
        Calculate the time since last frame and calculate FPS.
        Print log statements
        :return: fps of last frame
        """
        # Calculate time for last frame and FPS
        time_dif = self.current_frame_time - self.previous_frame_time
        fps = 1 / time_dif
        if self.LOG_EVERY_PRINT_TIME:
            # If configured like this, print every frame's FPS
            logger.debug(f"CAM{self.cam_nr}: Frame time: {time_dif} FPS: {fps}")
        else:
            # Print out FPS in set frame intervals
            self.nr_of_frames_taken += 1
            self.avg_frame_time = (self.avg_frame_time * (
                    self.nr_of_frames_taken - 1) + time_dif) / self.nr_of_frames_taken
            if self.nr_of_frames_taken >= self.FRAME_TIME_PRINTOUT_INTERVAL_FRAMES:
                logger.info(
                    f"CAM{self.cam_nr}: Avg frame time is {self.avg_frame_time} in {self.nr_of_frames_taken} frames. Target is: {self.frame_time}")
                self.nr_of_frames_taken = 0
                self.avg_frame_time = 0.0
        return fps

    def record_on_movement(self):
        # Set initial status as waiting for movement
        self.q_status.put(self.STATUS_WAITING_ON_MOVEMEMENT)
        logger.info(f"CAM{self.cam_nr}: Recording on movement mode started")
        while not self.stop_flag:
            # Frame is skipped if one is not received from the camera
            skip_show_frame = False
            # Check Queue from main to see
            self.check_queue()
            # Get current time to calculate if next frame should be taken
            current_time = time.perf_counter()
            fps = 0.0
            time_passed_since_previous_frame = current_time - self.current_frame_time
            if time_passed_since_previous_frame >= self.frame_time:
                # Frame time passed, ask for next frame
                # Save previous frame and time of previous frame
                self.previous_frame = self.current_frame
                self.previous_frame_time = self.current_frame_time
                # Get new frame and save time of current frame
                self.current_frame_time = current_time
                ret, self.current_frame = self.capture.read()
                if not ret:
                    # Received invalid frame, skip showing of it
                    logger.error(f"CAM{self.cam_nr}: Invalid frame")
                    skip_show_frame = True
                else:
                    # Copy current frame so text can be added to it
                    self.frame_to_display = self.current_frame.copy()
                if self.flag_return_pic:
                    # If main has requested a frame, put it in the queue
                    self.put_frame_in_queue()
                    self.flag_return_pic = False
                if self.previous_frame is not None and not skip_show_frame:
                    # If the there is a previous frame and the current frame is not broken
                    # Check if there is movement between the previous and current frame
                    movement = self.check_for_movement()
                    if self.recording_active:
                        # If there already was a recording active
                        if not movement:
                            # If no movement, check if the recording should be stopped
                            time_since_last_movement = current_time - self.last_movement_time
                            if time_since_last_movement >= self.STOP_RECORDING_DLY_S:
                                # Enough time has passed since last movement to stop recording
                                self.recording_active = False
                        else:
                            # Update last movement time if movement detected
                            self.last_movement_time = current_time
                    else:
                        # No recording was active
                        if movement:
                            # Start recorkding on movement`
                            self.recording_active = True
                            self.last_movement_time = current_time
                    # FPS between last and current frame
                    fps = self.frame_rate_logging()
                self.record_file()
                self.stream_video(self.frame_to_display)
                if self.flag_show_video and not skip_show_frame:
                    # Show current frame in UI. Put FPS on picture
                    self.put_text_on_frame(fps)
                    cv2.imshow("Camera", self.frame_to_display)
                    # If this is not called, exceptiouns occur
                    cv2.waitKey(1)
        # Stop flag was set
        logger.debug(f"Exited main loop")
        self.stop_file_recording()
        self.capture.release()
        cv2.destroyAllWindows()
        logger.info(f"CAM{self.cam_nr} Stop flag was set, recording stopped")

    def stream_video(self, frame):
        """
        :param frame: to be displayed on the webserver
        """
        if self.webserver is not None:
            if not self.web_q_out.empty():
                # Somebody has opened the webstream
                while not self.web_q_out.empty():
                    # Clear the queue
                    self.web_q_out.get(False)
                self.time_since_last_stream_frame_req = time.perf_counter()
                self.streaming_active = True
            if self.streaming_active:
                while not self.web_q_in.empty():
                    # Emptyy previous items from Queue
                    self.web_q_in.get()
                self.web_q_in.put((False, frame))
            time_passed_since_last_stream_req = time.perf_counter() - self.time_since_last_stream_frame_req
            if time_passed_since_last_stream_req > 1.0:
                # Consider that nobody has opened the webstream if no frame has bee requested for a second
                self.streaming_active = False

    def put_text_on_frame(self, text):
        """
        Put text on top left corner of frame
        :param text:
        """
        cv2.putText(img=self.frame_to_display,
                    text=f'{text:.2f}',
                    org=(10, 30),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=(0, 255, 255),
                    thickness=2,
                    lineType=cv2.LINE_4)

    def record_file(self):
        """
        Method that writes video files
        """
        if self.recording_active and not self.stop_flag:
            # recording on
            if self.v_out is None:
                # No video writer active so it must be initialised
                # Updae status to main
                self.q_status.put(self.STATUS_RECORDING)
                # vout needs to be initialised
                # Create video file name using date and time
                today = datetime.now()
                current_date = today.strftime("%Y_%m_%d")
                current_time = today.strftime("%H_%M_%S")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.current_file_name = f"{current_date}_{current_time}.mp4"
                logger.info(f"CAM{self.cam_nr}: starting recording of video: {self.current_file_name}")
                # Create a folder for each day
                full_file_location = f"{self.file_location}/{current_date}"
                self.create_folder(full_file_location)
                # Initialise video writer
                self.v_out = cv2.VideoWriter(f"{full_file_location}/{self.current_file_name}", fourcc, self.fps,
                                             self.frame_size)
            # Write current frame
            self.v_out.write(self.current_frame)
        else:
            # recording should not be done
            self.stop_file_recording()

    def stop_file_recording(self):
        if self.v_out is not None:
            # If video writer is not not stop it
            logger.info(f"CAM{self.cam_nr}: Stopped recording of file {self.current_file_name}")
            self.v_out.release()
            self.v_out = None
            self.current_file_name = "none"

    def get_time_date_string(self):
        """
        :return: current date and time as string like this : 2022_02_31_19_30_05
        """
        today = datetime.now()
        current_date = today.strftime("%Y_%m_%d")
        current_time = today.strftime("%H_%M_%S")
        string = f"{current_date}_{current_time}"
        return string

    def create_folder(self, path):
        """
        Create folder
        :param path: folder to create
        """
        if os.path.exists(path):
            logger.debug(f"CAM{self.cam_nr}: Path {path} exists, no need to create")
        else:
            logger.info(f"CAM{self.cam_nr}: Path does not exist, creating {path}")
            os.makedirs(path)
            logger.debug(f"CAM{self.cam_nr}: Path created")

    def check_for_movement(self):
        """
        Checks differences between current and previous frame to determine if there was movement
        :return: Boolean True if movement detected, Fals if not
        """
        movement_detected = False
        # diference between current and previous frame
        dif = cv2.absdiff(self.current_frame, self.previous_frame)
        # Graysacale diference
        gray = cv2.cvtColor(dif, cv2.COLOR_RGB2GRAY)
        # Convert to blur image
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        # Get rid of noise
        _, thresh = cv2.threshold(src=blur, thresh=20, maxval=255, type=cv2.THRESH_BINARY)
        # Make the recorded differences bigger???
        dilated = cv2.dilate(thresh, None, iterations=3)
        # Draw contours of movement
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if cv2.contourArea(c) >= self.frame_area_for_movement:
                movement_detected = True
                if not self.draw_contours:
                    # Big enough movement detected do not process rest of contours
                    break
                else:
                    x, y, w, h = cv2.boundingRect(c)
                    cv2.rectangle(self.frame_to_display, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return movement_detected

    def change_settings(self, display_video=True, draw_contours=True):
        """
        Change setting of movement recorder
        :param display_video: set True to have a window showing current frame and constantly updating
        :param draw_contours: set True to have contours drawn around movement
        :return:
        """
        self.show_video_recording = display_video
        self.draw_contours = draw_contours

    def put_frame_in_queue(self):
        """
        Put current frame in Queue for main to receive
        """
        self.q_out.put(self.frame_to_display)

    def display_video_only(self):
        """
        Do not record but just display video.
        This mode might need tweeking to work, not testerd
        """
        while not self.stop_flag:
            skip_show_frame = False
            self.check_queue()
            current_time = time.perf_counter()
            time_passed_since_rpevious_frame = current_time - self.previous_frame_time
            if time_passed_since_rpevious_frame >= self.frame_time:
                self.previous_frame_time = self.current_frame_time
                self.current_frame_time = current_time
                self.previous_frame_time = self.current_frame_time
                ret, self.current_frame = self.capture.read()
                if not ret:
                    logger.error(f"CAM{self.cam_nr}: Invalid frame")
                    skip_show_frame = True
                if self.flag_return_pic:
                    self.put_frame_in_queue()
                    self.flag_return_pic = False
                if self.previous_frame is not None:
                    self.frame_rate_logging()
                if self.flag_show_video and not skip_show_frame:
                    cv2.imshow("Camera", self.current_frame)
                    cv2.waitKey(1)
        self.capture.release()
        cv2.destroyAllWindows()
        logger.info(f"CAM{self.cam_nr} Stop flag was set, recording stopped")
