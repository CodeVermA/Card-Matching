import os
import cv2 as cv
import numpy
import numpy as np
import pytesseract

# Global Variables
PATH = "resources"

# Static Function
def _load_images(folder_path: str):
    count = 0
    images = []

    for filename in os.listdir(folder_path):
        count += 1
        img_path = os.path.join(folder_path, filename)
        img = cv.imread(img_path)  # Read the image
        if not (img is None):
            images.append(img)

    return images



def display_images_side_by_side(images, window_name="Images"):
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

def get_text(image:numpy.ndarray) -> str:
    """Determines if a card contains text and returns it."""
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Apply thresholding to highlight text
    _, thresh = cv.threshold(gray, 150, 255, cv.THRESH_BINARY_INV)

    # Extract text using Tesseract OCR
    text = pytesseract.image_to_string(thresh, config='--psm 6')
    return text.strip()

def is_shape_card(image):
    """Determines if a card contains shapes using contour analysis."""
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Edge detection
    edges = cv.Canny(gray, 50, 150)

    # Find contours
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Filter out small noise
    shape_count = sum(1 for cnt in contours if cv.contourArea(cnt) > 500)

    return shape_count > 0  # If shapes are detected, it's a shape card


def extract_cards(image, output_dir):
    output_dir = os.path.join(output_dir, "cards")
    # Load the image
    image = cv.imread(image)

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


class Robot:
    def __init__(self):
        self._capture = cv.VideoCapture(0)
        self._orb = cv.ORB_create()

        self.images = []
        self._img_count = 0

        self._SIMILARITY_THRESHOLD = 0.70

    def capture_image(self, save_loc:str):
        """
        Capture the image from the camera and saves.

        @:param save_loc: Location to save the captured image.
        """
        save_loc = os.path.join(save_loc, "boardImage.jpg")
        captured, img = self._capture.read()
        if captured:
            cv.imwrite(save_loc, img)
        else:
            raise RuntimeError("IMAGE NOT CAPTURED")

        self.images = extract_cards(save_loc, PATH)

    def _feature_matching(self, img1, img2):
        """
        Compare Given images and return similarity(float)
        """
        img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
        img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

        # Find key points and descriptors
        kp1, des1 = self._orb.detectAndCompute(img1, None)
        kp2, des2 = self._orb.detectAndCompute(img2, None)

        # Use BFMatcher (Brute Force Matcher)
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)

        # Sort matches by distance (lower is better)
        matches = sorted(matches, key=lambda x: x.distance)

        # Compute similarity score
        similarity = len(matches) / max(len(kp1), len(kp2))

        return similarity, matches, img1, img2, kp1, kp2

    def _handle_shape_cards(self, img_to_compare):
        similar_images = [img_to_compare]

        for img in self.images[:-1]:
            similarity = self._feature_matching(img, img_to_compare)[0]
            if round(similarity, 2) >= self._SIMILARITY_THRESHOLD:
                similar_images.append(img)

        if len(similar_images) > 1 : display_images_side_by_side(similar_images)
        else: print("== NO SIMILAR IMAGES FOUND ==\n\n")

        #return similar_images

    def _handle_text_cards(self, text):
        pass

    def compare(self):
        img_to_compare = self.images[-1]

        if is_shape_card(img_to_compare):
            print("== SHAPE CARDS DETECTED == \n")
            self._handle_shape_cards(img_to_compare)

        text = get_text(img_to_compare)
        if text != '':
            print("== TEXT CARDS DETECTED == \n")
            self._handle_text_cards(text)

        else:
            raise RuntimeError("CARD NOT DETECTED AS SHAPE OR TEXT CARD")


    def destroy(self):
        self._capture.release()
        cv.destroyAllWindows()

