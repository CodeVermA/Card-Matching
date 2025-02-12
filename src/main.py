import cv2
from textcomp import TextComp

img = cv2.imread("resources/testimg1.png")

text = TextComp([img]).get_texts()
print(text)