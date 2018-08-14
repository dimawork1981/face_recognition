"""
system of access accounting by face recognition
"""

import csv
import datetime
import os
import time

import cv2
import face_recognition

faces_folder = 'faces'
unknown_faces_folder = 'unknown faces'

# Checking for folders
if not os.path.exists(faces_folder):
    os.mkdir(faces_folder)
if not os.path.exists(unknown_faces_folder):
    os.mkdir(unknown_faces_folder)

# Get a list of names of known people
known_faces = [folder for folder in os.listdir(faces_folder) if os.path.isdir(os.path.join(faces_folder, folder))]

# Creation a dictionary of known faces encodings and their names
known_faces_encodings = {}
for face in known_faces:
    known_faces_encodings[face] = [
        face_recognition.face_encodings(face_recognition.load_image_file(os.path.join(faces_folder, face, foto)))[0]
        for foto in os.listdir(os.path.join(faces_folder, face)) if foto.lower().endswith('.jpg')]


def face_detection(face_encodings):
    names = []
    for face_encoding in face_encodings:
        for face in known_faces_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_faces_encodings[face], face_encoding)
            if True in matches:
                names.append(face)
    if not names:
        names = ['Unknown']
    return names


def write_name_to_base(dtime, users):
    with open('base.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        for user in users:
            writer.writerow((dtime, user))


def main():
    # Get a reference to webcam #0
    vc = cv2.VideoCapture(0)

    # check webcam
    if vc.isOpened():
        is_capturing = vc.read()[0]
    else:
        is_capturing = False

    user_counter = 0
    while is_capturing:
        # Grab a single frame of video
        is_capturing, frame = vc.read()

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_frame = frame[:, :, ::-1]

        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Detection after 1 second after the appearance of the face
        if face_encodings:
            user_counter += 1
        if user_counter > 2:
            dtime = datetime.datetime.strftime(datetime.datetime.now(), "%Y.%m.%d %H-%M-%S")
            print(dtime, 'face detected')
            users = face_detection(face_encodings)
            print(dtime, 'users: ', users)
            write_name_to_base(dtime, users)
            if 'Unknown' in users:
                print(dtime, 'Write unknown face to JPEG file')
                # save frame as JPEG file
                cv2.imwrite(os.path.join(unknown_faces_folder, "Unknown face at %s.jpg" % dtime), frame)
            user_counter = 0

        time.sleep(0.5)


if __name__ == '__main__':
    main()
