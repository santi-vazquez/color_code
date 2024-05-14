from typing import *
from PIL import Image
import numpy as np
import cv2
import numpy.typing as npt

image_path = "images/small_code.png"

image = Image.open(image_path)


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
        return (((self_row - other_row) ** 2) + ((self_col - other_col) ** 2)) ** 0.5

    def furthest(self, other: int) -> int:
        largest_dist = 0
        index_of_largest_dist = self.idx[0]

        for i in self.idx:
            dist = (
                (((i // image.width) - (other // image.width)) ** 2)
                + (((i % image.width) - (other % image.width)) ** 2)
            ) ** 0.5
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


def is_squareish(points: List[tuple[int, int]]):
    minimum_row = min((row for row, _ in points))
    minmum_col = min((col for row, col in points if row == minimum_row))

    corner = (minimum_row, minmum_col)
    diameter = 0
    opposing_corner = corner
    for row, col in points:
        dist = abs(row - corner[0]) + abs(col - corner[1])
        if dist > diameter:
            diameter = dist
            opposing_corner = (row, col)

    radius_plus_diameter = 0
    third_corner = opposing_corner
    for row, col in points:
        dist = ((((row - corner[0]) ** 2) + ((col - corner[1]) ** 2)) ** 0.5) + (
            (((row - opposing_corner[0]) ** 2) + ((col - opposing_corner[1]) ** 2))
            ** 0.5
        )
        if dist > radius_plus_diameter:
            radius_plus_diameter = dist
            third_corner = (row, col)

    middle_point = (
        (corner[0] + opposing_corner[0]) / 2,
        (corner[1] + opposing_corner[1]) / 2,
    )

    opposing_to_third_corner = (
        middle_point[0] - (third_corner[0] - middle_point[0]),
        middle_point[1] - (third_corner[1] - middle_point[1]),
    )

    squares = (corner, third_corner, opposing_corner, opposing_to_third_corner)

    def subtract_tuple(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
        return (a[0] - b[0], a[1] - b[1])

    def cross_product(a: tuple[int, int], b: tuple[int, int]):
        return a[0] * b[1] - a[1] * b[0]

    def is_in_points(
        point: tuple[int, int],
        points: tuple[
            tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]
        ],
    ):
        square_area = abs(
            cross_product(
                subtract_tuple(squares[0], squares[1]),
                subtract_tuple(squares[2], squares[1]),
            )
        )

        possible_in_square_area = 0
        for point_a, point_b in [
            (points[0], points[1]),
            (points[1], points[2]),
            (points[2], points[3]),
            (points[3], points[0]),
        ]:
            possible_in_square_area += (
                abs(
                    cross_product(
                        subtract_tuple(point_a, point), subtract_tuple(point_b, point)
                    )
                )
                // 2
            )

        return abs(square_area - possible_in_square_area) * 100 < square_area

    true_positive = 0
    false_negative = 0
    for point in points:
        if is_in_points(point, squares):
            true_positive += 1
        else:
            false_negative += 1

    square_area = abs(
        cross_product(
            subtract_tuple(squares[0], squares[1]),
            subtract_tuple(squares[2], squares[1]),
        )
    )

    false_positive = square_area - true_positive

    return (true_positive - false_negative - false_positive) / true_positive > 0.9


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

    all_elements_idx = [
        i for i, p in enumerate(ids) if p in grand_parent.union(parent).union({first})
    ]

    if not is_squareish(
        [(i // image.width, i % image.width) for i in all_elements_idx]
    ):
        continue

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

supposedly_bottom_left_to_top_right = (
    two_squares[0].center()[0] - two_squares[1].center()[0],
    two_squares[0].center()[1] - two_squares[1].center()[1],
)

supposedly_bottom_left_to_top_left = (
    most_distance_other_square.center()[0] - two_squares[1].center()[0],
    most_distance_other_square.center()[1] - two_squares[1].center()[1],
)

cross_product = (
    supposedly_bottom_left_to_top_right[0] * supposedly_bottom_left_to_top_left[1]
    - supposedly_bottom_left_to_top_right[1] * supposedly_bottom_left_to_top_left[0]
)

if cross_product < 0:
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

bottom_right_of_top_right = two_squares[0].furthest(top_left)
bottom_right_of_bottom_left = two_squares[1].furthest(top_left)

right_line = (
    bottom_right_of_top_right // image.width - top_right_row_col[0],
    bottom_right_of_top_right % image.width - top_right_row_col[1],
)

bottom_line = (
    bottom_right_of_bottom_left // image.width - bottom_left_row_col[0],
    bottom_right_of_bottom_left % image.width - bottom_left_row_col[1],
)


def find_intersection(m1, c1, m2, c2):
    if c2 == None:
        return (m1 * m2 + c1), m2
    if c1 == None:
        return (m2 * m1 + c2), m1
    x_intersect = (c2 - c1) / (m1 - m2)
    y_intersect = m1 * x_intersect + c1
    return y_intersect, x_intersect


def calculate_slope_and_intercept(first, delta):
    dx = delta[1]
    dy = delta[0]
    if dx == 0:
        return first[1], None
    m = dy / dx
    c = first[0] - m * first[1]
    return m, c


mr, cr = calculate_slope_and_intercept(top_right_row_col, right_line)
mb, cb = calculate_slope_and_intercept(bottom_left_row_col, bottom_line)

bottom_right_row_col = find_intersection(mr, cr, mb, cb)


def get_warped(pixels, bounding_box):
    x_set = set()
    y_set = set()
    for coord in bounding_box:
        x_set.add(int(coord[0]))
        y_set.add(int(coord[1]))

    pts1 = np.float32(bounding_box)
    width = max(x_set) - min(x_set)
    height = max(y_set) - min(y_set)
    pts2 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

    M = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(pixels, M, (width, height))


def segment(pixels, K):
    pixels = np.float32(pixels)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    attempts = 10

    _, labels, centers = cv2.kmeans(
        pixels, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS
    )
    centers = np.uint8(centers)

    # Map each pixel to its respective cluster
    return centers[labels.flatten()]


def horizontal_lines(pixels, bounding_box):
    warped = get_warped(pixels, bounding_box)
    qr_image_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    pixels = qr_image_gray.reshape((-1,))
    segmented_image = segment(pixels, 2).reshape(qr_image_gray.shape)
    return np.array(
        [
            [
                255.0 if p > segmented_image[row_i - 1, col_i] else 0.0
                for col_i, p in enumerate(row)
            ]
            for row_i, row in enumerate(segmented_image)
        ]
    )


def vertical_lines(pixels, bounding_box):
    warped = get_warped(pixels, bounding_box)
    qr_image_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    pixels = qr_image_gray.reshape((-1,))
    segmented_image = segment(pixels, 2).reshape(qr_image_gray.shape)
    return np.array(
        [
            [
                255.0 if p > segmented_image[row_i, col_i - 1] else 0.0
                for col_i, p in enumerate(row)
            ]
            for row_i, row in enumerate(segmented_image)
        ]
    ).transpose()


def std_dv_horizontal(pixels, bounding_box):
    values = [sum(row) for row in horizontal_lines(pixels, bounding_box)]
    import statistics

    return statistics.stdev(values)


def std_dv_vertical(pixels, bounding_box):
    values = [sum(row) for row in vertical_lines(pixels, bounding_box)]
    import statistics

    return statistics.stdev(values)


image_data_cv = cv2.imread(image_path)


def std_dv_with_vertical(move_vertical_percentage):
    bbox = [
        top_left_row_col,
        top_right_row_col,
        (
            bottom_right_row_col[0]
            + move_vertical_percentage
            * (top_right_row_col[0] - bottom_right_row_col[0]),
            bottom_right_row_col[1]
            + move_vertical_percentage
            * (top_right_row_col[1] - bottom_right_row_col[1]),
        ),
        bottom_left_row_col,
    ]
    bbox = [(col, row) for row, col in bbox]
    return std_dv_horizontal(image_data_cv, bbox)


def std_dv_with_horizontal(move_horizontal):
    bbox = [
        top_left_row_col,
        top_right_row_col,
        (
            bottom_right_row_col[0]
            + move_horizontal * (bottom_left_row_col[0] - bottom_right_row_col[0]),
            bottom_right_row_col[1]
            + move_horizontal * (bottom_left_row_col[1] - bottom_right_row_col[1]),
        ),
        bottom_left_row_col,
    ]
    bbox = [(col, row) for row, col in bbox]
    return std_dv_vertical(image_data_cv, bbox)


print(bottom_right_row_col)

move_vertical = 0

while std_dv_with_vertical(move_vertical + 0.001) > std_dv_with_vertical(move_vertical):
    move_vertical += 0.001

while std_dv_with_vertical(move_vertical - 0.001) > std_dv_with_vertical(move_vertical):
    move_vertical -= 0.001

bottom_right_row_col = (
    bottom_right_row_col[0]
    + move_vertical * (top_right_row_col[0] - bottom_right_row_col[0]),
    bottom_right_row_col[1]
    + move_vertical * (top_right_row_col[1] - bottom_right_row_col[1]),
)

print(bottom_right_row_col)

move_horizontal = 0

while std_dv_with_horizontal(move_horizontal + 0.001) > std_dv_with_horizontal(
    move_horizontal
):
    move_horizontal += 0.001

while std_dv_with_horizontal(move_horizontal - 0.001) > std_dv_with_horizontal(
    move_horizontal
):
    move_horizontal -= 0.001

bottom_right_row_col = (
    bottom_right_row_col[0]
    + move_horizontal * (bottom_left_row_col[0] - bottom_right_row_col[0]),
    bottom_right_row_col[1]
    + move_horizontal * (bottom_left_row_col[1] - bottom_right_row_col[1]),
)

print(bottom_right_row_col)

bbox = [top_left_row_col, top_right_row_col, bottom_right_row_col, bottom_left_row_col]
bbox = [(col, row) for row, col in bbox]

print(bbox)

# horizontal_lines(cv2.imread(image_path), bbox)

qr_image = get_warped(cv2.imread(image_path), bbox)

qr_image_rgb = cv2.cvtColor(qr_image, cv2.COLOR_BGR2RGB)
pixels = qr_image_rgb.reshape((-1, 3))

# # Reshape the segmented image to the original shape
segmented_image = segment(pixels, 4).reshape(qr_image.shape)
segmented_image_bw = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2GRAY)
height, width, _ = segmented_image.shape

(ids, region_count) = floodfill(
    segment_image_data(segmented_image.reshape((-1, 3)), width, height, 3),
    width,
    height,
)

friend_mapping, is_border_mapping, size = get_friend_mapping(ids, width, height)

squares = []


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

    # if any([is_border_mapping[p] for p in grand_parent]):
    #     continue

    all_elements_idx = [
        i for i, p in enumerate(ids) if p in grand_parent.union(parent).union({first})
    ]

    if not is_squareish([(i // width, i % width) for i in all_elements_idx]):
        continue

    squares.append(
        Square(
            first,
            parent.pop(),
            grand_parent.pop(),
            all_elements_idx,
            width,
        )
    )

square_length = int(
    np.round(
        (
            (width * height)
            / (sum([len(square.idx) for square in squares]) / len(squares) / 49)
        )
        ** 0.5
    )
)

print(squares[0].center(), segmented_image_bw[squares[0].center()])
print(squares[1].center(), segmented_image_bw[squares[1].center()])
print(squares[2].center(), segmented_image_bw[squares[2].center()])

mapping = {
    segmented_image_bw[squares[0].center()]: 1,
    segmented_image_bw[squares[2].center()]: 2,
    segmented_image_bw[squares[1].center()]: 3,
}

pixel_values = [
    [
        mapping.get(
            segmented_image_bw[
                int(((row_i + 0.5) / square_length) * height),
                int(((col_i + 0.5) / square_length) * width),
            ],
            0,
        )
        for col_i in range(square_length)
    ]
    for row_i in range(square_length)
]

# print(pixel_values)

values = []

for row_i in range(square_length):
    for col_i in range(square_length):
        if row_i < 8 and col_i < 8:
            continue
        if row_i < 8 and square_length - col_i < 9:
            continue
        if square_length - row_i < 9 and col_i < 8:
            continue
        values.append(pixel_values[row_i][col_i])

chars = [
    chr(h_64 * 64 + h_16 * 16 + h_4 * 4 + h_1)
    for h_64, h_16, h_4, h_1 in zip(
        values[::4], values[1::4], values[2::4], values[3::4]
    )
]

output = ""

for c in chars:
    if c == "\x00":
        break
    output += c

print(output)


# # print("data", data)
# # cv2.imshow("Bounding Box", qr_image)
cv2.imshow("Segmented Image", cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR))

cv2.waitKey(0)
cv2.destroyAllWindows()
