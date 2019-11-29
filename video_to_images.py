import os
from tqdm import tqdm

from pipeline.capture_video import CaptureVideo
from pipeline.display_video import DisplayVideo
from pipeline.save_image import SaveImage


def parse_args():
    import argparse

    # Parse command line arguments
    ap = argparse.ArgumentParser(description="Convert video to set of frame images pipeline")
    ap.add_argument("-i", "--input", default="0",
                    help="path to input video file or camera identifier")
    ap.add_argument("-o", "--output", default="output",
                    help="path to output directory")
    ap.add_argument("-ie", "--image-ext", default="jpg", choices=["jpg", "png"],
                    help="image extension (output format)")
    ap.add_argument("-d", "--display", action="store_true",
                    help="display video")
    ap.add_argument("-p", "--progress", action="store_true",
                    help="display progress")

    return ap.parse_args()


def main(args):
    # Create output directory if needed
    os.makedirs(args.output, exist_ok=True)

    # Create pipeline steps
    capture_video = CaptureVideo(int(args.input) if args.input.isdigit() else args.input)

    display_video = DisplayVideo("image") \
        if args.display else None

    save_image = SaveImage("image", args.output, image_ext=args.image_ext)

    # Create image processing pipeline
    pipeline = (capture_video |
                display_video |
                save_image)

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
        capture_video.cleanup()


if __name__ == "__main__":
    args = parse_args()
    main(args)
