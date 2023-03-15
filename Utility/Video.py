import sys

import cv2


# Inspired by: https://stackoverflow.com/questions/33311153/python-extracting-and-saving-video-frames
def split_video(src, destination):
    print(src)
    print(destination)
    vidcap = cv2.VideoCapture(str(src))
    success, image = vidcap.read()
    count = 100000
    if not success:
        print("Something went wrong when reading vidoe from " + str(src))
    while success:
        cv2.imwrite(str(destination) + "frame%d.jpg" % count, image)  # save frame as JPEG file
        success, image = vidcap.read()
        count += 1
    print("Created " + str(count - 100000) + " frames and saved them under " + str(destination))


def main():
    split_video(sys.argv[1], sys.argv[2])


if __name__ == '__main__':
    main()
