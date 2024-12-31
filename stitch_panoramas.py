import cv2
import numpy as np
import os

def stitch_images(image_paths):
    """Stitches a list of image paths into a panorama."""
    images = [cv2.imread(path) for path in image_paths]

    # Initialize the Stitcher class
    stitcher = cv2.Stitcher_create()

    # Stitch the images
    (status, stitched_image) = stitcher.stitch(images)

    if status == cv2.STITCHER_OK:
        return stitched_image
    else:
        print(f"Error during stitching: {status}")
        return None

# Example usage
dir_path = "inputs/360-1"
image_paths = [os.path.join(dir_path, img_name) for img_name in os.listdir("inputs/360-1")]
stitched_panorama = stitch_images(image_paths)

if stitched_panorama is not None:
    cv2.imwrite("panorama.jpg", stitched_panorama)
    cv2.imshow("Panorama", stitched_panorama)
    cv2.waitKey(0)
    cv2.destroyAllWindows()