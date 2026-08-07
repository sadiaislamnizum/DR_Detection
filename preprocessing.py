import cv2
import numpy as np

# ----------------------------------------------------------
# ROI Extraction
# ----------------------------------------------------------

def crop_retina(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)

    coords = cv2.findNonZero(thresh)

    if coords is not None:

        x, y, w, h = cv2.boundingRect(coords)

        image = image[y:y+h, x:x+w]

    return image


# ----------------------------------------------------------
# CLAHE Enhancement
# ----------------------------------------------------------

def apply_clahe(image):

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(

        clipLimit=2.0,

        tileGridSize=(8,8)

    )

    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))

    image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return image