import torch
import multiprocessing as mp
import bisect

from detectron2.engine.defaults import DefaultPredictor


class AsyncPredictor:
    """ AsyncPredictor was ported from Detectron2 and extended with the possibility
    to run more workers on cpu which helps to speed up inference together with GPU.

    It is also possible to return results without preserving the order (in case of parallel
    processing of independent images).

    Credits: https://github.com/facebookresearch/detectron2
    """

    class _StopToken:
        pass

    class _PredictWorker(mp.Process):
        def __init__(self, cfg, task_queue, result_queue):
            self.cfg = cfg
            self.task_queue = task_queue
            self.result_queue = result_queue
            super().__init__()

        def run(self):
            predictor = DefaultPredictor(self.cfg)

            while True:
                task = self.task_queue.get()
                if isinstance(task, AsyncPredictor._StopToken):
                    break

                idx, data = task
                result = predictor(data)
                self.result_queue.put((idx, result))

    def __init__(self, cfg, num_gpus=1, num_cpus=1, queue_size=3, ordered=True):
        num_gpus = min(torch.cuda.device_count(), num_gpus)
        num_cpus = min(mp.cpu_count(), num_cpus)
        assert num_gpus > 0 or num_cpus > 0, "Number of gpus or cpus must be specified"
        num_procs = num_gpus + num_cpus

        self.ordered = ordered
        self.task_queue = mp.Queue(maxsize=num_procs * queue_size)
        self.result_queue = mp.Queue(maxsize=num_procs * queue_size)

        self.procs = []
        if num_gpus > 0:
            # Run GPU workers
            for gpuid in range(num_gpus):
                cfg = cfg.clone()
                cfg.defrost()
                cfg.MODEL.DEVICE = f"cuda:{gpuid}"
                self.procs.append(
                    AsyncPredictor._PredictWorker(cfg, self.task_queue, self.result_queue)
                )
        if num_cpus > 0:
            # Run CPU workers
            for _ in range(num_cpus):
                cfg = cfg.clone()
                cfg.defrost()
                cfg.MODEL.DEVICE = "cpu"
                self.procs.append(
                    AsyncPredictor._PredictWorker(cfg, self.task_queue, self.result_queue)
                )

        if self.ordered:
            self.put_idx = 0
            self.get_idx = 0
            self.result_rank = []
            self.result_data = []

        for p in self.procs:
            p.start()

    def put(self, image):
        if self.ordered:
            self.put_idx += 1
            self.task_queue.put((self.put_idx, image))
        else:
            image_idx, image = image
            self.task_queue.put((image_idx, image))

    def get(self):
        if self.ordered:
            self.get_idx += 1  # the index needed for this request

            if len(self.result_rank) and self.result_rank[0] == self.get_idx:
                # Result is already present in result_data buffer
                res = self.result_data[0]
                del self.result_data[0], self.result_rank[0]
                return res

            while True:
                # Make sure the results are returned in the correct order
                idx, res = self.result_queue.get()
                if idx == self.get_idx:
                    return res
                insert = bisect.bisect(self.result_rank, idx)
                self.result_rank.insert(insert, idx)
                self.result_data.insert(insert, res)
        else:
            return self.result_queue.get()

    def __call__(self, image):
        self.put(image)
        return self.get()

    def shutdown(self):
        for _ in self.procs:
            self.task_queue.put(AsyncPredictor._StopToken())

    @property
    def num_procs(self):
        return len(self.procs)
