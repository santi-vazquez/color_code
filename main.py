from typing import *
from PIL import Image
import numpy as np
import cv2 as cv
import numpy.typing as npt

image = Image.open("clean.png")


def floodfill_helper(
    ids: npt.NDArray,
    pixels: npt.NDArray,
    width: int,
    height: int,
    pixel: int,
    value: int,
):
    pixels_queue = [pixel]
    while pixels_queue:
        pixel = pixels_queue.pop(0)
        if ids[pixel]:
            continue
        ids[pixel] = value

        row, col = pixel // width, pixel % width

        dp = [(-1, 0), (0, -1), (1, 0), (0, 1)]

        for dr, dc in dp:
            nr, nc = row + dr, col + dc
            if nr < 0 or nc < 0 or nr >= height or nc >= width:
                continue
            np = nr * width + nc
            if ids[np] == 0 and (pixels[np] == pixels[pixel]).all():
                pixels_queue.append(np)


def floodfill(pixels: npt.NDArray, width: int, height: int):
    ids = np.zeros((pixels.shape[0],))
    current_id = 0
    while np.min(ids) == 0:
        current_id += 1
        idx = np.argmin(ids)
        floodfill_helper(ids, pixels, width, height, idx, current_id)
        Image.fromarray(
            np.uint8((ids.reshape((width, height)) == current_id).astype(int) * 255)
        ).save('output/image' + str(current_id) + '.png')

    return (ids, current_id)

image_data = np.array(list(image.getdata()))

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 2
ret, label, center = cv.kmeans(np.float32(image_data), K, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((image.width * image.height, 4))

ids = floodfill(res2, image.width, image.height)

print(ids)