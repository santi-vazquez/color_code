# import cv2

# # Load the image
# img = cv2.imread("images/images-2.jpg")

# # Create a QR code detector
# qr_detector = cv2.QRCodeDetector()

# # Detect and decode the QR code
# data, bbox, rectified_image = qr_detector.detectAndDecode(img)

# if data:
#     print("QR Code detected:")
#     print("Data:", data)
#     print("Bounding box:", bbox)
# else:
#     print("No QR Code found")

# cv2.imshow('image', rectified_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


import cv2
import numpy as np

# Load the image
img = cv2.imread("images/images-3.jpg")

# Create a QR code detector
qr_detector = cv2.QRCodeDetector()

# Detect the QR code and get the bounding box
data, bbox, _ = qr_detector.detectAndDecode(img)
print(bbox[0])
x_set = set()
y_set = set()
for coord in bbox[0]:
    x_set.add(int(coord[0]))
    y_set.add(int(coord[1]))

# Optional: Apply perspective transformation for a straight view
pts1 = np.float32([bbox[0][0], bbox[0][1], bbox[0][2], bbox[0][3]])
width = max(x_set) - min(x_set)
height = max(y_set) - min(y_set)
pts2 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

M = cv2.getPerspectiveTransform(pts1, pts2)
qr_image = cv2.warpPerspective(img, M, (width, height))

qr_image_rgb = cv2.cvtColor(qr_image, cv2.COLOR_BGR2RGB)
pixels = qr_image_rgb.reshape((-1, 3))
pixels = np.float32(pixels)

k = 3 # Number of clusters
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
attempts = 10

_, labels, centers = cv2.kmeans(pixels, k, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
centers = np.uint8(centers)

# Map each pixel to its respective cluster
segmented_image = centers[labels.flatten()]

# Reshape the segmented image to the original shape
segmented_image = segmented_image.reshape(qr_image.shape)

print('data', data)
# cv2.imshow("Bounding Box", qr_image)
cv2.imshow("Segmented Image", cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR))

cv2.waitKey(0)
cv2.destroyAllWindows()