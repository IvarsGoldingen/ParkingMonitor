import logging
import threading
import queue
from flask import Response
from flask import Flask
from multiprocessing import Queue
import cv2
import time
from flask import render_template
import os

# Setup logging
log_formatter = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s:%(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# Console debug
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(log_formatter)
logger.addHandler(stream_handler)
# File logger
file_handler = logging.FileHandler(os.path.join("logs", "webstream.log"))
file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.WARNING)
logger.addHandler(file_handler)


class SurveillanceWebStream(threading.Thread):
    """
    Class for streaming webcam to webpage
    Created using this tutorial:
    https://pyimagesearch.com/2019/09/02/opencv-stream-video-to-web-browser-html-page/
    """

    def __init__(self, queue):
        """
        :param queue: Queue for receiving frames to send to the webserver
        """
        super().__init__(daemon=True)
        self.queue = queue
        self._flask = Flask(__name__)

    def run(self):
        logger.debug(f"Starting flask")
        self._flask.add_url_rule('/', 'base_url', self.index)
        self._flask.add_url_rule('/videostream', 'video_stream', self.video_feed)
        self._flask.run()

    def index(self):
        # Called when  http://127.0.0.1:5000/ opened
        logger.debug(f"Index called")
        # Index Html will call video feed function
        return render_template("index.html")

    def video_feed(self):
        return Response(self.get_frame(),
                        mimetype="multipart/x-mixed-replace; boundary=frame")

    def get_frame(self):
        """
        Constantly check queue for frames or stop flag
        """
        while True:
            frame = None
            try:
                # while not self.queue.empty():
                stop_flag, frame = self.queue.get(block=True, timeout=0.1)
                if stop_flag:
                    logger.info(f"Stop flag received - stopping,")
                    break
            except queue.Empty:
                logger.debug(f"Queue empty")
            if frame is not None:
                # Encode frame for webserver
                (success, encodedImage) = cv2.imencode(".jpg", frame)
                if success:
                    yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
                           bytearray(encodedImage) + b'\r\n')


if __name__ == "__main__":
    # Example usage:
    flask_input_queue = Queue()
    web_video = SurveillanceWebStream(flask_input_queue)
    web_video.start()
    capture = cv2.VideoCapture(0)
    while True:
        ret, output_frame = capture.read()
        flask_input_queue.put((False, output_frame))
        time.sleep(0.1)
