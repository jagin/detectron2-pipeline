import cv2
import time

from threading import Thread
from queue import Queue


class WebcamVideoCapture:
    """Webcam capturing class utilizing threading and the queue to obtain FPS speedup."""

    def __init__(self, src=0, fourcc=None, width=None, height=None, fps=None,
                 transform=None, queue_size=128, name="WebcamVideoCapture"):
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video {src}")

        # Set capture properties
        if fourcc:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        if width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps:
            self.cap.set(cv2.CAP_PROP_FPS, fps)

        self.transform = transform

        # initialize the queue used to store frames read from the webcam
        self.queue = Queue(maxsize=queue_size)

        # initialize the variable used to indicate if the thread should be stopped
        self.stopped = False

        self.thread = Thread(target=self.update, args=(), name=name)
        self.thread.daemon = True

    def start(self):
        # start a thread to read frames from the file video stream
        self.thread.start()
        return self

    def get(self, cv2_prop):
        return self.cap.get(cv2_prop)

    def update(self):
        # keep looping infinitely until the thread is stopped
        while self.cap.isOpened():
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                break

            # drop the last frame if the queue is full
            if self.queue.full():
                self.queue.get()

            # otherwise, ensure the queue has room in it
            if not self.queue.full():
                # read the next frame from the file
                (grabbed, frame) = self.cap.read()

                # if the `grabbed` boolean is `False`, then we have reached the end of the video file
                if not grabbed:
                    self.stopped = True
                    break

                # if there are transforms to be done, might as well
                # do them on producer thread before handing back to
                # consumer thread. ie. Usually the producer is so far
                # ahead of consumer that we have time to spare.
                #
                # Python is not parallel but the transform operations
                # are usually OpenCV native so release the GIL.
                #
                # Really just trying to avoid spinning up additional
                # native threads and overheads of additional
                # producer/consumer queues since this one was generally
                # idle grabbing frames.
                if self.transform:
                    frame = self.transform(frame)

                # add the frame to the queue
                self.queue.put(frame)
            else:
                time.sleep(0.01)  # Rest for 1ms, we have a full queue

        self.cap.release()

    def read(self):
        # return next frame in the queue
        return self.queue.get()

    # Insufficient to have consumer use while(more()) which does
    # not take into account if the producer has reached end of
    # file stream.
    def running(self):
        return self.more() or not self.stopped

    def more(self):
        # return True if there are still frames in the queue. If stream is not stopped, try to wait a moment
        tries = 0
        while self.queue.qsize() == 0 and not self.stopped and tries < 5:
            time.sleep(0.1)
            tries += 1

        return self.queue.qsize() > 0

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
        self.thread.join()
