import numpy as np
import cv2
import imutils
from tensorflow.keras.models import load_model
from imutils.contours import sort_contours
import io, base64
from PIL import Image

def find_contours(img):
    conts = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    conts = imutils.grab_contours(conts)
    conts = sort_contours(conts, method='left-to-right')[0]
    return conts


def extract_roi(img, y, h, x, w):
    roi = img[y:y + h, x:x + w]
    return roi


def thresholding(img):
    thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    return thresh


def resize_img(img, w, h):
    if w > h:
        resized = imutils.resize(img, width=28)
    else:
        resized = imutils.resize(img, height=28)

    (h, w) = resized.shape
    dX = int(max(0, 28 - w) / 2.0)
    dY = int(max(0, 28 - h) / 2.0)

    filled = cv2.copyMakeBorder(resized, top=dY, bottom=dY, right=dX, left=dX, borderType=cv2.BORDER_CONSTANT,
                                value=(0, 0, 0))
    filled = cv2.resize(filled, (28, 28))
    return filled


def normalization(img):
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=-1)
    return img


def process_box(gray, x, y, w, h, characters):
    roi = extract_roi(gray, y, h, x, w)
    thresh = thresholding(roi)
    (h, w) = thresh.shape
    resized = resize_img(thresh, w, h)
    normalized = normalization(resized)
    characters.append((normalized, (x, y, w, h)))


def base64_to_png(img_url):
    base64_str = img_url[22:len(img_url)]
    img = Image.open(io.BytesIO(base64.decodebytes(bytes(base64_str, "utf-8"))))
    img.save('my-image.png')

def get_chars (img_url):
    base64_to_png(img_url)
    network = load_model('network')
    network.summary()
    img = cv2.imread('../my-image.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    adaptive = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 9)
    invertion = 255 - adaptive
    dilation = cv2.dilate(invertion, np.ones((1, 1)))
    edges = cv2.Canny(dilation, 40, 150)
    dilation = cv2.dilate(edges, np.ones((1, 1)))
    conts = find_contours(dilation.copy())

    min_w, max_w = 4, 160
    min_h, max_h = 14, 140
    img_copy = img.copy()

    for c in conts:
        (x, y, w, h) = cv2.boundingRect(c)
        if (w >= min_w and w <= max_w) and (h >= min_h and h <= max_h):
            roi = gray[y:y + h, x:x + w]
            thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), (255, 100, 0), 2)

    (x, y, w, h) = cv2.boundingRect(conts[6])
    test_img = thresholding(gray[y:y + h, x:x + w])
    (h, w) = test_img.shape
    test_img2 = resize_img(test_img, w, h)

    characters = []
    for c in conts:
        (x, y, w, h) = cv2.boundingRect(c)
        if (w >= min_w and w <= max_w) and (h >= min_h and h <= max_h):
            process_box(gray, x, y, w, h, characters)

    boxes = [box[1] for box in characters]
    pixels = np.array([pixel[0] for pixel in characters], dtype='float32')
    digits = '0123456789'
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    characters_list = digits + letters
    characters_list = [l for l in characters_list]
    predictions = network.predict(pixels)

    text = ''
    for (prediction, (x, y, w, h)) in zip(predictions, boxes):
        i = np.argmax(prediction)
        character = characters_list[i]
        text = text + character

    return text
