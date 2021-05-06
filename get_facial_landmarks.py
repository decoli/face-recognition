import face_recognition
import matplotlib.pyplot as plt
import numpy as np
import argparse

def get_facial_landmarks(args):
    # load image
    img = face_recognition.load_image_file(args.face_path)

    # get the face loc
    face_locs = face_recognition.face_locations(img, model="cnn")

    # get facial landmarks
    facial_landmarks = face_recognition.face_landmarks(img, face_locs)

    label = {
        'nose_bridge': 'nose_bridge',
        'nose_tip': 'nose_tip',
        'top_lip': 'top_lip',
        'bottom_lip': 'bottom_lip',
        'left_eye': 'left_eye',
        'right_eye': 'right_eye',
        'left_eyebrow': 'left_eyebrow',
        'right_eyebrow': 'right_eyebrow',
        'chin': 'chin'
        }

    # visualize
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(img)
    ax.set_axis_off()
    for face in facial_landmarks:
        for name, points in face.items():
            points = np.array(points)
            ax.plot(points[:, 0], points[:, 1], 'o-', ms=3, label=label[name])
    ax.legend(fontsize=14)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-F', '--face_path')
    args = parser.parse_args()

    get_facial_landmarks(args)
