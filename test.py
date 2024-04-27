import cv2

# Load the image
img = cv2.imread("images/clean.png")

# Create a QR code detector
qr_detector = cv2.QRCodeDetector()

# Detect and decode the QR code
data, bbox, rectified_image = qr_detector.detectAndDecode(img)

if data:
    print("QR Code detected:")
    print("Data:", data)
    print("Bounding box:", bbox)
else:
    print("No QR Code found")

# cv2.imshow('image', rectified_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()