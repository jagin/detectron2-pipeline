from pipeline.pipeline import Pipeline

import numpy as np
import cv2


class SeparateBackground(Pipeline):

    def __init__(self, dst, me_kernel=(7, 7), bg_kernel=(21, 21), desaturate=True):
        self.dst = dst
        self.me_kernel = me_kernel  # mask edges gaussian blur kernel
        self.bg_kernel = bg_kernel  # background gaussian blur kernel
        self.desaturate = desaturate  # convert background to grayscale

        super().__init__()

    def map(self, data):
        self.separate_background(data)

        return data

    def separate_background(self, data):
        if "predictions" not in data:
            return

        predictions = data["predictions"]
        if "instances" not in predictions:
            return

        instances = predictions["instances"]
        if not instances.has("pred_masks"):
            return

        # Sum up all the instance masks
        mask = instances.pred_masks.cpu().sum(0) >= 1
        mask = mask.numpy().astype("uint8")*255
        # Create 3-channels mask
        mask = np.stack([mask, mask, mask], axis=2)

        # Apply a slight blur to the mask to soften edges
        mask = cv2.GaussianBlur(mask, self.me_kernel, 0)

        # Take the foreground input image
        foreground = data["image"]

        # Create a Gaussian blur for the background image
        background = cv2.GaussianBlur(foreground, self.bg_kernel, 0)

        if self.desaturate:
            # Convert background into grayscale
            background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

            # convert single channel grayscale image to 3-channel grayscale image
            background = cv2.cvtColor(background, cv2.COLOR_GRAY2RGB)

        # Convert uint8 to float
        foreground = foreground.astype(float)
        background = background.astype(float)

        # Normalize the alpha mask to keep intensity between 0 and 1
        mask = mask.astype(float)/255.0

        # Multiply the foreground with the mask
        foreground = cv2.multiply(foreground, mask)

        # Multiply the background with ( 1 - mask )
        background = cv2.multiply(background, 1.0 - mask)

        # Add the masked foreground and background
        dst_image = cv2.add(foreground, background)

        # Return a normalized output image for display
        data[self.dst] = dst_image.astype("uint8")
