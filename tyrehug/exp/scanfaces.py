import cv2
import os
import argparse

def get_faces(filename):
    face_cascade = cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_eye.xml')
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    face_images = []

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 1:
            face_images.append(roi_color)

    return face_images


def scan_images(root_dir, output_dir):
    image_extensions = ["jpg", "png"]
    num_faces = 0
    num_images = 0

    print("Images directory {}".format(root_dir))
    print("Output directory {}".format(output_dir))
    print("-" * 20)

    for dir_name, subdir_list, file_list in os.walk(root_dir):
        print('Scanning directory: %s' % dir_name)
        for filename in file_list:
            extension = os.path.splitext(filename)[1][1:]
            if extension in image_extensions:
                faces = get_faces(os.path.join(dir_name, filename))
                num_images += 1

                for face in faces:
                    face_filename = os.path.join(output_dir, "face{}.png".format(num_faces))
                    cv2.imwrite(face_filename, face)
                    print("\tWrote {} extracted from {}".format(face_filename, filename))
                    num_faces += 1

    print("-" * 20)
    print("Total number of images: {}".format(num_images))
    print("Total number of faces: {}".format(num_faces))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract faces from photos.')
    parser.add_argument('imagesdir', type=str, help='Input directory of images')
    parser.add_argument('outputdir', type=str, help='Output directory for faces')
    args = parser.parse_args()

    scan_images(args.imagesdir, args.outputdir)
