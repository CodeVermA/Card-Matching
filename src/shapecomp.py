import os
import cv2 as cv
import numpy
from numpy.ma.core import alltrue


class ShapeComp:
    def __init__(self, images):
        if not all(map(lambda x: ShapeComp.has_shape(x), images)):
            raise ValueError("=== ALL IMAGES MUST HAVE SHAPE ===\n\n")

        self._images = images
        self._orb = cv.ORB_create()

        self._SIMILARITY_THRESHOLD = 0.7

    def add(self, image):
        if not ShapeComp.has_shape(image):
            raise ValueError("=== IMAGE MUST HAVE SHAPE ===\n\n")

        self._images.append(image)

    def has_shape(image:numpy.ndarray) -> bool:
        """Determines if a card contains shapes using contour analysis."""
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        # Edge detection
        edges = cv.Canny(gray, 50, 150)

        # Find contours
        contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # Filter out small noise
        shape_count = sum(1 for cnt in contours if cv.contourArea(cnt) > 500)

        return shape_count > 0  # If shapes are detected, it's a shape card

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

        return similarity, matches, kp1, kp2

    def compare(self):
        """
        Compare the recent image(Latest image added) with the rest of the images
        :return:
        """
        img_to_comp = self._images[-1]
        similar_images = [img_to_comp]

        for img in self._images[:-1]:
            similarity = self._feature_matching(img, img_to_comp)[0]
            if round(similarity, 2) >= self._SIMILARITY_THRESHOLD:
                similar_images.append(img)

        if len(similar_images) > 1:
            print(similar_images)
        else:
            print("== NO SIMILAR IMAGES FOUND ==\n\n")
