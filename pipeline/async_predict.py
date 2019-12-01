from collections import deque

from pipeline.pipeline import Pipeline
from pipeline.libs.async_predictor import AsyncPredictor


class AsyncPredict(Pipeline):
    """Pipeline task to perform prediction asynchronously (in separate processes)."""

    def __init__(self, cfg, num_gpus=1, num_cpus=1, queue_size=3, ordered=True):
        self.predictor = AsyncPredictor(cfg,
                                        num_gpus=num_gpus,
                                        num_cpus=num_cpus,
                                        queue_size=queue_size,
                                        ordered=ordered)
        self.ordered = ordered
        self.buffer_size = self.predictor.num_procs * queue_size

        super().__init__()

    def generator(self):
        if self.ordered:
            return self.serial_generator()
        else:
            return self.parallel_generator()

    def serial_generator(self):
        buffer = deque()
        stop = False
        buffer_cnt = 0
        while self.has_next() and not stop:
            try:
                data = next(self.source)
                buffer.append(data)
                self.predictor.put(data["image"])
                buffer_cnt += 1
            except StopIteration:
                stop = True

            if buffer_cnt >= self.buffer_size:
                predictions = self.predictor.get()
                data = buffer.popleft()
                data["predictions"] = predictions

                if self.filter(data):
                    yield self.map(data)

        while len(buffer):
            predictions = self.predictor.get()
            data = buffer.popleft()
            data["predictions"] = predictions

            if self.filter(data):
                yield self.map(data)

    def parallel_generator(self):
        buffer = {}
        stop = False
        buffer_cnt = 0
        while self.has_next() and not stop:
            try:
                data = next(self.source)
                buffer[data["image_id"]] = data
                self.predictor.put((data["image_id"], data["image"]))
                buffer_cnt += 1
            except StopIteration:
                stop = True

            if buffer_cnt >= self.buffer_size:
                image_id, predictions = self.predictor.get()
                data = buffer[image_id]
                data["predictions"] = predictions
                del buffer[image_id]

                if self.filter(data):
                    yield self.map(data)

        while len(buffer.keys()):
            image_id, predictions = self.predictor.get()
            data = buffer[image_id]
            data["predictions"] = predictions
            del buffer[image_id]

            if self.filter(data):
                yield self.map(data)

    def cleanup(self):
        self.predictor.shutdown()
