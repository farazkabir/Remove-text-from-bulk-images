"""Remove text from bulk images

This script allows the user to remove texts from multiple images automatically
or without human assistance using keras ocr and opencv inpainting

It accepts 3 optional arguments - images directory, output directory and batch size.
Default image directory: Folder named 'images' in the same directory as the script
Default output directory: Output folder inside same directory
Default batch size: 1

This script requires the libraires in the requirements.txt to be installed within the Python
environment you are running this script in.

"""


import time
import keras_ocr
import cv2
import math
import numpy as np
import os
import argparse


# Default parameters for inpainting
radius = 3
method = cv2.INPAINT_NS  # inpainting method


def midpoint(x1, y1, x2, y2):
    """Get's the middle coordinate of 2 points

    Parameters
    ----------
    x1: int
        x coordinate of point 1
    y1: int
        y coordinate of point 1
    x2: int
        x coordinate of point 2
    y2: int
        y coordinate of point 2

    Returns
    -------
    int, int
        x,y coordinates of the middle point
    """
    x_mid = int((x1 + x2)/2)
    y_mid = int((y1 + y2)/2)
    return (x_mid, y_mid)


pipeline = keras_ocr.pipeline.Pipeline()


def inpaint_text(img_paths, pipeline):
    """Detects text using keras ocr and then inpaints the area using opencv inpainting

    Parameters
    ----------
    image_paths: list
        list of the path of the images
    pipeline: object
        Keras OCR pipeline

    Returns
    -------
    array
        inpainted image
    """

    # read images
    images = [
        keras_ocr.tools.read(url) for url in img_paths
    ]

    # generate (word, box) tuples
    prediction_groups = pipeline.recognize(images)
    cnt = 0
    for i in range(0, len(images)):

        mask = np.zeros(images[i].shape[:2], dtype="uint8")
        for box in prediction_groups[i]:
            x0, y0 = box[1][0]
            x1, y1 = box[1][1]
            x2, y2 = box[1][2]
            x3, y3 = box[1][3]

            x_mid0, y_mid0 = midpoint(x1, y1, x2, y2)
            x_mid1, y_mi1 = midpoint(x0, y0, x3, y3)

            thickness = int(math.sqrt((x2 - x1)**2 + (y2 - y1)**2))

            cv2.line(mask, (x_mid0, y_mid0), (x_mid1, y_mi1), 255,
                     thickness)
            images[i] = cv2.inpaint(images[i], mask, radius, method)

    return(images)


cwd = os.getcwd()  # Get current directory

parser = argparse.ArgumentParser()
parser.add_argument('-i',  type=str, default=os.path.join(cwd,
                    'images'), help="Input Folder Path")
parser.add_argument('-o', type=str, default=os.path.join(cwd,
                    'output'), help="Output Folder Path")
parser.add_argument('-b', type=int, default=1, help="Batch Size")


args = parser.parse_args()


folder_path = args.i
output = args.o


files = os.listdir(folder_path)
files.sort()


if not os.path.isdir(output):
    os.makedirs(output)

batch_start = 0
batch_end = args.b


err = []
cnt = 0
t1 = time.time()

while True:

    paths = [os.path.join(folder_path, f) for f in files]

    try:
        imgs = inpaint_text(paths[batch_start:batch_end], pipeline)
        for img in imgs:

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            output_img = os.path.join(output, os.path.basename(paths[cnt]))
            cv2.imwrite(output_img, img_rgb)
            cnt += 1

        batch_start = batch_end

        if batch_end + batch_end <= len(paths):
            batch_end += batch_end
        else:
            batch_end = len(paths)

    except:
        err.append(paths[cnt])
        cnt += 1

        pass

    if(cnt == len(paths)):
        break


t2 = time.time()
print(f"time: {str(t2 - t1)}, total files: {str(len(os.listdir(output)))}")


print(err)

# Save the error list
with open(output + '/error.txt', 'w') as fp:
    fp.write("\n".join(str(item) for item in err))
