import os
from tqdm import tqdm
import multiprocessing as mp

from pipeline.capture_video import CaptureVideo
from pipeline.capture_frames import CaptureFrames
from pipeline.predict import Predict
from pipeline.async_predict import AsyncPredict
from pipeline.separate_background import SeparateBackground
from pipeline.annotate_video import AnnotateVideo
from pipeline.display_video import DisplayVideo
from pipeline.save_video import SaveVideo
from pipeline.utils import detectron
from pipeline.track_pose import TrackPose


def parse_args():
    import argparse

    # Parse command line arguments
    ap = argparse.ArgumentParser(description="Detectron2 video processing pipeline")
    ap.add_argument("-i", "--input", default="0",
                    help="path to input video file, image frames directory or camera identifier (default: camera 0)")
    ap.add_argument("-o", "--output", default="output",
                    help="path to output directory")
    ap.add_argument("-ov", "--out-video", default=None,
                    help="output video file name")
    ap.add_argument("--fps", type=int, default=None,
                    help="overwrite fps for output video or if it is unknown for image frames directory")
    ap.add_argument("-p", "--progress", action="store_true",
                    help="display progress")
    ap.add_argument("-d", "--display", action="store_true",
                    help="display video")
    ap.add_argument("-sb", "--separate-background", action="store_true",
                    help="separate background")
    ap.add_argument("-tp", "--track-pose", action="store_true",
                    help="track pose")

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

    # Pose tracking options
    ap.add_argument("--track-link-len", type=int, default=100,
                    help="look-ahead link_len frames to find tracked human bbox")
    ap.add_argument("--track-num", type=int, default=7,
                    help="pick high-score(top num) keypoints when computing pose_IOU")
    ap.add_argument("--track-mag", type=int, default=30,
                    help="box width/height around keypoint for computing pose IoU")
    ap.add_argument("--track-match", type=float, default=0.2,
                    help="match threshold in Hungarian Matching")
    ap.add_argument("--track-orb-features", type=int, default=1000,
                    help="ORB matching number of features")

    # Mutliprocessing settings
    ap.add_argument("--gpus", type=int, default=1,
                    help="number of GPUs (default: 1)")
    ap.add_argument("--cpus", type=int, default=0,
                    help="number of CPUs (default: 0)")
    ap.add_argument("--queue-size", type=int, default=3,
                    help="queue size per process (default: 3)")
    ap.add_argument("--single-process", action="store_true",
                    help="force the pipeline to run in a single process")

    return ap.parse_args()


def main(args):
    # Create output directory if needed
    os.makedirs(args.output, exist_ok=True)

    # Create pipeline steps
    if args.input.isdigit():
        capture_video = CaptureVideo(int(args.input))
    elif os.path.isfile(args.input):
        capture_video = CaptureVideo(args.input)
    elif os.path.isdir(args.input):
        capture_video = CaptureFrames(args.input)

    cfg = detectron.setup_cfg(config_file=args.config_file,
                              weights_file=args.weights_file,
                              config_opts=args.config_opts,
                              confidence_threshold=args.confidence_threshold,
                              cpu=False if args.gpus > 0 else True)
    if not args.single_process:
        mp.set_start_method("spawn", force=True)
        predict = AsyncPredict(cfg,
                               num_gpus=args.gpus,
                               num_cpus=args.cpus,
                               queue_size=args.queue_size,
                               ordered=True)
    else:
        predict = Predict(cfg)

    track_pose = TrackPose(link_len=args.track_link_len, num=args.track_num, mag=args.track_mag,
                           match=args.track_match, orb_features=args.track_orb_features) if args.track_pose else None

    if args.separate_background:
        separate_background = SeparateBackground("vis_image")
        annotate_video = None
    else:
        separate_background = None
        metadata_name = cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        annotate_video = AnnotateVideo("vis_image", metadata_name,
                                       predictions=track_pose is None,
                                       pose_flows=track_pose is not None)

    display_video = DisplayVideo("vis_image") \
        if args.display else None

    save_video = SaveVideo("vis_image", os.path.join(args.output, args.out_video),
                           capture_video.fps if args.fps is None else args.fps) \
        if args.out_video else None

    # Create image processing pipeline
    pipeline = (capture_video |
                predict |
                track_pose |
                separate_background |
                annotate_video |
                display_video |
                save_video)

    # Iterate through pipeline
    try:
        for _ in tqdm(pipeline,
                      total=capture_video.frame_count if capture_video.frame_count > 0 else None,
                      disable=not args.progress):
            pass
    except StopIteration:
        return
    except KeyboardInterrupt:
        return
    finally:
        # Pipeline cleanup
        if isinstance(predict, CaptureVideo):
            capture_video.cleanup()
        if isinstance(predict, AsyncPredict):
            predict.cleanup()
        if display_video:
            display_video.cleanup()
        if save_video:
            save_video.cleanup()


if __name__ == "__main__":
    args = parse_args()
    main(args)
