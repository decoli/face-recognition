import argparse
import glob
import os

import face_recognition
import matplotlib.pyplot as plt


def check_face(args):

    # load the known face
    known_face_path = glob.glob(os.path.join(args.known_face_dir, '*'))
    known_face_imgs = []
    for each_path in known_face_path:
        img = face_recognition.load_image_file(each_path)
        known_face_imgs.append(img)

    # get the known face location
    known_face_locs = []
    for img in known_face_imgs:
        loc = face_recognition.face_locations(img, model="cnn")

        # no face or more than one.
        # assert len(loc) == 1

        known_face_locs.append(loc)

    # load the to-check face
    face_img_to_check = face_recognition.load_image_file(args.check_face_path)
    # get the to-check face loc
    face_loc_to_check = face_recognition.face_locations(face_img_to_check, model="cnn")

    # get face feature
    known_face_encodings = []
    for img, loc in zip(known_face_imgs, known_face_locs):
        (encoding,) = face_recognition.face_encodings(img, loc)
        known_face_encodings.append(encoding)

    (face_encoding_to_check,) = face_recognition.face_encodings(
        face_img_to_check, face_loc_to_check
    )

    # get matche result
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding_to_check)
    dists = face_recognition.face_distance(known_face_encodings, face_encoding_to_check)

    # print result
    for path, match, dist in zip(known_face_path, matches, dists):
        if match:
            print(dist, path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-K', '--known_face_dir', default='./known_face')
    parser.add_argument('-C', '--check_face_path')
    args = parser.parse_args()
    check_face(args)
