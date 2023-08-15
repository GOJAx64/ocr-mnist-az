import base64
import io
import cv2
from PIL import Image
from easyocr import Reader


def base64_to_png(img_url):
    base64_str = img_url[22:len(img_url)]
    img = Image.open(io.BytesIO(base64.decodebytes(bytes(base64_str, "utf-8"))))
    img.save('my-image.png')


def get_chars(img_url):
    languages_list = ['en', 'es']
    text = ''
    base64_to_png(img_url)
    img = cv2.imread('my-image.png')
    reader = Reader(languages_list, False)
    results = reader.readtext(img)
    length = len(results)
    for i in range(length):
        item = results[i]
        text = text + item[1] + ' '
    return text
