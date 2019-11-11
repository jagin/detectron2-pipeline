import os
from tqdm import tqdm
import multiprocessing as mp

from pipeline.capture_images import CaptureImages
from pipeline.capture_image import CaptureImage
from pipeline.predict import Predict
from pipeline.async_predict import AsyncPredict
from pipeline.separate_background import SeparateBackground
from pipeline.annotate_image import AnnotateImage
from pipeline.save_image import SaveImage
from pipeline.utils import detectron


def parse_args():
    import argparse

    # Parse command line arguments
    ap = argparse.ArgumentParser(description="Detectron2 image processing pipeline")
    ap.add_argument("-i", "--input", required=True,
                    help="path to input image file or directory")
    ap.add_argument("-o", "--output", default="output",
                    help="path to output directory (default: output)")
    ap.add_argument("-p", "--progress", action="store_true",
                    help="display progress")
    ap.add_argument("-sb", "--separate-background", action="store_true",
                    help="separate background")

    # Detectron settings
    ap.add_argument("--config-file",
                    default="configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
                    help="path to config file (default: configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml)")
    ap.add_argument("--config-opts", default=[], nargs=argparse.REMAINDER,
                    help="modify model config options using the command-line")
    ap.add_argument("--weights-file", default=None,
                    help="path to model weights file")
    ap.add_argument("--confidence-threshold", type=float, default=0.5,
                    help="minimum score for instance predictions to be shown (default: 0.5)")

    # Mutliprocessing settings
    ap.add_argument("--gpus", type=int, default=1,
                    help="number of gpus (default: 1)")
    ap.add_argument("--workers", type=int, default=1,
                    help="number of workers for CPU (default: 1)")
    ap.add_argument("--single-process", action="store_true",
                    help="force the pipeline to run in a single process")

    return ap.parse_args()


def main(args):
    # Create output directory if needed
    os.makedirs(args.output, exist_ok=True)

    # Create pipeline steps
    capture_images = CaptureImages(args.input) \
        if os.path.isdir(args.input) else CaptureImage(args.input)

    cfg = detectron.setup_cfg(config_file=args.config_file,
                              weights_file=args.weights_file,
                              config_opts=args.config_opts,
                              confidence_threshold=args.confidence_threshold,
                              cpu=False if args.gpus > 0 else True)
    if not args.single_process:
        mp.set_start_method("spawn", force=True)
        predict = AsyncPredict(cfg,
                               num_gpus=args.gpus,
                               num_workers=args.workers,
                               ordered=False)
    else:
        predict = Predict(cfg)

    if args.separate_background:
        separate_background = SeparateBackground("vis_image")
        annotate_image = None
    else:
        separate_background = None
        metadata_name = cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        annotate_image = AnnotateImage("vis_image", metadata_name)

    save_image = SaveImage("vis_image", args.output)

    # Create image processing pipeline
    pipeline = (capture_images |
                predict |
                separate_background |
                annotate_image |
                save_image)

    # Iterate through pipeline
    try:
        for _ in tqdm(pipeline, disable=not args.progress):
            pass
    except StopIteration:
        return
    except KeyboardInterrupt:
        return
    finally:
        # Pipeline cleanup
        if isinstance(predict, AsyncPredict):
            predict.cleanup()


if __name__ == "__main__":
    args = parse_args()
    main(args)
