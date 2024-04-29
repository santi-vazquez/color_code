from typing import *
from PIL import Image
import numpy as np
import cv2 as cv
import numpy.typing as npt

image = Image.open("images/clean.png")


def get_adjacent(pixel: int, width: int, height: int) -> list[int]:
    row, col = pixel // width, pixel % width
    dp = [(-1, 0), (0, -1), (1, 0), (0, 1)]
    res = []
    for dr, dc in dp:
        nr, nc = row + dr, col + dc
        if nr < 0 or nc < 0 or nr >= height or nc >= width:
            continue
        res.append(nr * width + nc)
    return res


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
        for np in get_adjacent(pixel, width, height):
            if ids[np] == 0 and (pixels[np] == pixels[pixel]).all():
                pixels_queue.append(np)


def floodfill(pixels: npt.NDArray, width: int, height: int):
    ids = np.zeros((pixels.shape[0],), dtype=int)
    current_id = 0
    while np.min(ids) == 0:
        current_id += 1
        idx = np.argmin(ids)
        floodfill_helper(ids, pixels, width, height, idx, current_id)
        # if current_id < 4:
        #     Image.fromarray((255 * (ids == current_id)).reshape((image.width, image.height)).astype(np.uint8), mode='L').show()
    return (ids, current_id)


image_data = np.array(list(image.getdata()))


def segment_image_data(image_data: npt.NDArray, width: int, height: int, channels: int):
    brightness_mask = np.sum(image_data, axis=-1) > 256 + 128
    # Image.fromarray(
    #     (255 * brightness_mask).reshape((height, width)).astype(np.uint8),
    #     mode="L",
    # ).show()
    return brightness_mask


def is_contained_in(
    ids: npt.NDArray, id1: int, id2: int, width: int, height: int
) -> bool:
    pixel_idx = [i for i, p in enumerate(ids) if p == id1]
    if len(pixel_idx) < 10:
        return False
    pixel_idx_neighbors = sum([get_adjacent(p, width, height) for p in pixel_idx], [])
    neighbor_ids = {ids[p] for p in pixel_idx_neighbors}
    return (len(neighbor_ids) == 2 and id1 in neighbor_ids and id2 in neighbor_ids) or (
        len(neighbor_ids) == 1 and id2 in neighbor_ids
    )


def is_border_helper(i: int, width: int, height: int) -> bool:
    return (
        i // width == 0
        or i // width == height - 1
        or i % width == 0
        or i % width == width - 1
    )


def is_border(ids: npt.NDArray, search: set[int], width: int, height: int) -> bool:
    return any(
        [is_border_helper(i, width, height) for i, p in enumerate(ids) if p in search]
    )


(ids, region_count) = floodfill(
    segment_image_data(image_data, image.width, image.height, 3),
    image.width,
    image.height,
)


def get_friend_mapping(
    ids: npt.NDArray, width: int, height: int
) -> tuple[dict[int, set[int]], dict[int, bool], dict[int, int]]:
    freq = {i: set() for i in range(1, region_count + 1)}
    is_border = {i: False for i in range(1, region_count + 1)}
    size = {i: 9 for i in range(1, region_count + 1)}
    for i, p in enumerate(ids):
        neighbor_ids = {ids[p] for p in get_adjacent(i, width, height)}
        freq[p] |= neighbor_ids
        is_border[p] |= is_border_helper(i, width, height)
        size[p] += 1
    return freq, is_border, size


friend_mapping, is_border_mapping, size = get_friend_mapping(
    ids, image.width, image.height
)


class Square:
    def __init__(
        self,
        id: int,
        parent_id: int,
        grandparent_id: int,
        idx: Sequence[int],
        image_width: int,
    ) -> None:
        self.id = id
        self.parent_id = parent_id
        self.grandparent_id = grandparent_id
        self.idx = idx
        self.image_width = image_width

    def center(self) -> tuple[int, int]:
        places = [(i // self.image_width, i % self.image_width) for i in self.idx]
        rows = sum([row for row, _ in places]) // len(places)
        cols = sum([col for _, col in places]) // len(places)
        return (rows, cols)

    def distance(self, other: Self) -> int:
        self_row, self_col = self.center()
        other_row, other_col = other.center()
        return abs(self_row - other_row) + abs(self_col - other_col)

    def furthest(self, other: int) -> int:
        largest_dist = 0
        index_of_largest_dist = self.idx[0]

        for i in self.idx:
            dist = abs((i // image.width) - (other // image.width)) + abs(
                (i % image.width) - (other % image.width)
            )
            if dist > largest_dist:
                largest_dist = dist
                index_of_largest_dist = i

        return index_of_largest_dist

    def show_image(self):
        Image.fromarray(
            (255 * np.isin(ids, [self.id, self.parent_id, self.grandparent_id]))
            .reshape((image.height, image.width))
            .astype(np.uint8),
            mode="L",
        ).show()

    def __eq__(self, other: Self) -> bool:
        return self.id == other.id


squares: list[Square] = []

for first in range(1, region_count + 1):
    parent = friend_mapping[first] - {first}

    if len(parent) > 1:
        continue

    grand_parent = set().union(*[friend_mapping[p] for p in parent]) - {first}.union(
        parent
    )
    if len(grand_parent) > 1:
        continue
    grand_grand_parent = set().union(*[friend_mapping[p] for p in grand_parent]) - {
        first
    }.union(parent, grand_parent)

    if len(grand_grand_parent) > 1:
        continue

    if any([is_border_mapping[p] for p in grand_parent]):
        continue

    ## TODO: MAKE SURE ELEMENT IS SQUARE-ISH

    all_elements_idx = [
        i for i, p in enumerate(ids) if p in grand_parent.union(parent).union({first})
    ]

    squares.append(
        Square(
            first,
            parent.pop(),
            grand_parent.pop(),
            all_elements_idx,
            image.width,
        )
    )

max_distance = 0
two_squares = (squares[0], squares[1])

for first in squares:
    for second in squares:
        dist = first.distance(second)
        if dist > max_distance:
            max_distance = dist
            two_squares = (first, second)

max_distance = 0
most_distance_other_square = squares[0]

for square in squares:
    dist = min(
        two_squares[0].distance(square),
        two_squares[1].distance(square),
    )
    if dist > max_distance or size[most_distance_other_square.id] * 2 < size[square.id]:
        max_distance = dist
        most_distance_other_square = square

# Assumption: Second Square is at the right
if two_squares[0].center()[1] < two_squares[1].center()[1]:
    two_squares[0], two_squares[1] = two_squares[1], two_squares[0]

image_center = [
    (a + b) // 2 for a, b in zip(two_squares[0].center(), two_squares[1].center())
]

image_center_idx = image_center[0] * image.width + image_center[1]

top_left = most_distance_other_square.furthest(image_center_idx)
top_right = two_squares[0].furthest(image_center_idx)
bottom_left = two_squares[1].furthest(image_center_idx)

top_left_row_col = top_left // image.width, top_left % image.width
top_right_row_col = top_right // image.width, top_right % image.width
bottom_left_row_col = bottom_left // image.width, bottom_left % image.width

## TODO: FIGURE BOTTOM RIGHT USING TRIANGULATION

bottom_right_row_col = (
    top_left // image.width + (image_center[0] - top_left // image.width) * 2,
    top_left % image.width + (image_center[1] - top_left % image.width) * 2,
)

print(top_left_row_col, top_right_row_col, bottom_left_row_col, bottom_right_row_col)
