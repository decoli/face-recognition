import argparse

import face_recognition
import matplotlib.pyplot as plt
from IPython.display import display
from PIL import Image, ImageDraw


def draw_faces(img, locs):
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img, mode="RGBA")

    for top, right, bottom, left in locs:
        draw.rectangle((left, top, right, bottom), outline="lime", width=2)

    plt.figure("demo")
    plt.imshow(img)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-F', '--face_path')
    args = parser.parse_args()

    img = face_recognition.load_image_file(args.face_path)
    face_locs = face_recognition.face_locations(img, model="cnn", number_of_times_to_upsample=2)
    draw_faces(img, face_locs)
