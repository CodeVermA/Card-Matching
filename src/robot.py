import os
import cv2 as cv

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


class Robot:
    def __init__(self):
        self._capture = cv.VideoCapture(0)
        self._orb = cv.ORB_create()

        self.images = _load_images(PATH)
        self._img_count = len(self.images)

        self._SIMILARITY_THRESHOLD = 0.70

    def capture_image(self) -> bool:
        """
        Capture the image from the camera and saves in return whether it was successful.
        """
        save_loc = f"images/image{self._img_count + 1}.jpg"
        captured, img = self._capture.read()

        if captured:
            cv.imwrite(save_loc, img)
            self.images.append(img)
            self._img_count += 1

        return captured

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

    def compare(self):
        img_to_compare = self.images[-1]
        similar_images = []

        for img in self.images:
            similarity = self._feature_matching(img, img_to_compare)[0]
            if round(similarity, 2) >= self._SIMILARITY_THRESHOLD:
                similar_images.append(img)

        display_images_side_by_side(similar_images)
        return similar_images

    def destroy(self):
        self._capture.release()
        cv.destroyAllWindows()

