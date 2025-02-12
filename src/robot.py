import os
import cv2 as cv
import numpy as np

from shapecomp import ShapeComp
from textcomp import TextComp
import shutil

def display_images_in_grid(images, max_per_row=5, border_size=10, border_color=(255, 255, 255),
                           window_name="Image Grid"):
    if not images:
        print("No images to display.")
        return

    # Ensure all images have the same height for proper concatenation
    height = min(img.shape[0] for img in images)  # Find the smallest height
    resized_images = [cv.resize(img, (int(img.shape[1] * height / img.shape[0]), height)) for img in images]

    # Create a blank border image (same height, fixed width)
    border = np.full((height, border_size, 3), border_color, dtype=np.uint8)

    # Split images into rows of max_per_row images
    rows = [resized_images[i: i + max_per_row] for i in range(0, len(resized_images), max_per_row)]

    # Concatenate images row-wise with borders
    final_rows = []
    max_width = 0  # Track the widest row

    for row in rows:
        images_with_borders = []
        for i, img in enumerate(row):
            images_with_borders.append(img)
            if i < len(row) - 1:  # Add border between images, but not at the end of a row
                images_with_borders.append(border)

        # Concatenate images in the row
        row_concat = cv.hconcat(images_with_borders)
        final_rows.append(row_concat)
        max_width = max(max_width, row_concat.shape[1])  # Track widest row

    # Resize all rows to match the maximum width
    final_rows = [cv.copyMakeBorder(row, 0, 0, 0, max_width - row.shape[1], cv.BORDER_CONSTANT, value=border_color)
                  for row in final_rows]

    # Create vertical border with the same width as the rows
    v_border = np.full((border_size, max_width, 3), border_color, dtype=np.uint8)

    # Concatenate rows with vertical borders
    grid_image = final_rows[0]
    for i in range(1, len(final_rows)):
        grid_image = cv.vconcat([grid_image, v_border, final_rows[i]])

    # Display the final image grid
    cv.imshow(window_name, grid_image)
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
        if os.path.isdir(self.cards_path + "/cards"): shutil.rmtree(self.cards_path + "/cards")
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

        display_images_in_grid(cards_imgs)

        if len(cards_imgs) == 0:
            print("== NO CARDS FOUND ==")
            print("== NO COMPARISON TYPE SETUP. PLEASE CAPTURE THE IMAGE AGAIN ==\n")
            return

        if ShapeComp.has_shape(cards_imgs[0]):
            self.shape_comp = ShapeComp(cards_imgs)
            print("== SHAPE COMPARISON SETUP ==\n")
        else:
            self.text_comp = TextComp(cards_imgs)
            print("== TEXT COMPARISON SETUP ==\n")

    def capture_image(self):
        """
        Capture the image from the camera and saves.
        """
        captured = False
        while not captured:
            ret, frame = self._capture.read()
            if not ret:
                raise Exception("== FAILED TO CAPTURE IMAGE ==\n")

            cv.imshow("Press 'Space' to Capture, 'Esc' to Exit", frame)
            key = cv.waitKey(1) & 0xFF

            if key == ord(' '):  # Spacebar
                self.img = frame
                captured = True

            elif key == 27:  # Escape key
                self._capture.release()
                cv.destroyAllWindows()
                captured = True

        if captured: self.setup_comp_type()

    def load_board(self, image_path):
        """
        Capture the image from the camera and saves.
        """
        try:
            self.img = cv.imread(image_path)

        except Exception as e:
            print("== FAILED TO LOAD IMAGE ==")

        self.setup_comp_type()

    def extract_cards(self, output_dir):
        output_dir = os.path.join(output_dir, "cards")
        # Load the image

        # Resize image for processing (optional, keeps aspect ratio)
        scale_percent = 50  # Resize to 50% of original
        width = int(self.img.shape[1] * scale_percent / 100)
        height = int(self.img.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized_image = cv.resize(self.img, dim, interpolation=cv.INTER_AREA)

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




