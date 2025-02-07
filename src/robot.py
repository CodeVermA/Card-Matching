import os
import cv2 as cv
import numpy as np

from shapecomp import ShapeComp
from textcomp import TextComp
import shutil


def _display_images_side_by_side(images, window_name="Images"):
    # Ensure all images have the same height for proper concatenation
    height = min(img.shape[0] for img in images)  # Find the smallest height
    resized_images = [cv.resize(img,
                                (int(img.shape[1] * height / img.shape[0]), height)) for img in images]

    # Concatenate images horizontally
    concatenated_image = cv.hconcat(resized_images)

    # Display the image
    cv.imshow(window_name, concatenated_image)
    cv.waitKey(0)
    cv.destroyAllWindows()


class Robot:
    def __init__(self):
        self._capture = cv.VideoCapture(0)
        self.img = None
        self.cards_path = "resources"

        self.shape_comp = None
        self.text_comp = None

    def reset(self):
        shutil.rmtree(self.cards_path + "/cards")
        self.shape_comp = None
        self.text_comp = None

    def load_images(folder_path: str):
        images = []
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            img = cv.imread(img_path)  # Read the image
            if not (img is None):
                images.append(img)

        return images

    def setup_comp_type(self):
        self.reset()

        cards_names = self.extract_cards(self.cards_path)
        cards_imgs = Robot.load_images(self.cards_path + "/cards")

        if ShapeComp.has_shape(cards_imgs[0]):
            self.shape_comp = ShapeComp(cards_imgs)

        else:
            self.text_comp = TextComp(cards_imgs)

    def capture_image(self):
        """
        Capture the image from the camera and saves.
        """
        captured, img = self._capture.read()
        if captured:
            self.img = img
        else:
            raise RuntimeError("IMAGE NOT CAPTURED")

        self.setup_comp_type()

    def extract_cards(self, output_dir):
        output_dir = os.path.join(output_dir, "cards")
        # Load the image
        image = cv.imread(self.img)

        # Resize image for processing (optional, keeps aspect ratio)
        scale_percent = 50  # Resize to 50% of original
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized_image = cv.resize(image, dim, interpolation=cv.INTER_AREA)

        # Convert to grayscale
        gray = cv.cvtColor(resized_image, cv.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv.GaussianBlur(gray, (5, 5), 0)

        # Edge detection using Canny
        edges = cv.Canny(blurred, 50, 150)

        # Find contours
        contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # Sort contours by area (largest to smallest) and filter small ones
        min_area = 5000  # Minimum area threshold
        card_contours = [cnt for cnt in contours if cv.contourArea(cnt) > min_area]

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Process each detected card
        card_images = []
        for i, cnt in enumerate(card_contours):
            # Approximate contour to a polygon
            peri = cv.arcLength(cnt, True)
            approx = cv.approxPolyDP(cnt, 0.02 * peri, True)

            if len(approx) == 4:  # Ensure it's a quadrilateral (card shape)
                # Obtain top-down view of card using perspective transform
                pts = approx.reshape(4, 2)
                rect = np.zeros((4, 2), dtype="float32")

                # Order points: top-left, top-right, bottom-right, bottom-left
                s = pts.sum(axis=1)
                rect[0] = pts[np.argmin(s)]  # Top-left
                rect[2] = pts[np.argmax(s)]  # Bottom-right

                diff = np.diff(pts, axis=1)
                rect[1] = pts[np.argmin(diff)]  # Top-right
                rect[3] = pts[np.argmax(diff)]  # Bottom-left

                # Define dimensions of the new transformed card
                width, height = 267, 191
                dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")

                # Compute perspective transform matrix
                m = cv.getPerspectiveTransform(rect, dst)
                warped = cv.warpPerspective(resized_image, m, (width, height))

                # Save each card
                card_filename = os.path.join(output_dir, f"card_{i + 1}.jpg")
                cv.imwrite(card_filename, warped)
                card_images.append(card_filename)

        return card_images

    def compare(self):
        """
        Compare the recent image(Latest image added) with the rest of the images
        """
        if not (self.text_comp is None):
            self.text_comp.compare()

        elif not (self.shape_comp is None):
            self.shape_comp.compare()

        else:
            raise Exception("== NO COMPARISON TYPE SETUP ==\n\n")

    def destroy(self):
        self._capture.release()
        cv.destroyAllWindows()




