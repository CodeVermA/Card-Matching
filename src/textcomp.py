import cv2 as cv
import pytesseract

class TextComp:
    def __init__(self, images):
        self._texts = list(map(lambda x: TextComp._extract_text(x), images))

        if any(x.strip() == "" for x in self._texts):
            raise ValueError("== NO TEXT DETECTED IN ONE OF THE IMAGES ==\n\n")

    def add(self, image):
        text = TextComp._extract_text(image)

        if text == "":
            raise ValueError("== NO TEXT DETECTED ==\n\n")

        self._texts.append(text)

    def _extract_text(image):
        """
        Extract text from the given image using Tesseract OCR.
        :param image: The image file.
        :return: Extracted text.
        """
        if image is None:
            raise FileNotFoundError(f"=== IMAGE NOT FOUND ===\n\n")

        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        gray = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]

        extracted_text = pytesseract.image_to_string(gray, lang='eng', config='--psm 11')

        return extracted_text.strip()

    def compare(self):
        """
        Compare the recent image(Latest image added) with the rest of the images
        """
        similar_text = []
        text_to_comp = self._texts[-1]

        for text in self._texts[:-1]:
            if text == text_to_comp: similar_text.append(text)

        if len(similar_text) > 0: print(similar_text)
        else: print("=== NO SIMILAR TEXT FOUND ===\n\n")

