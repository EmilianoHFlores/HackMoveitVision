import cv2
import numpy as np

# Function to find the largest contour with red color
def find_red_blob(contours):
    largest_area = 0
    largest_contour = None

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > largest_area:
            largest_area = area
            largest_contour = contour

    return largest_contour

# Open the video capture
cap = cv2.VideoCapture('testvideo1cut.mp4')

def find_largest_blob(frame, lower, upper):
    # Convert the frame to the HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Threshold the frame to extract red regions
    mask = cv2.inRange(hsv, lower, upper)

    # Find contours in the thresholded frame
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest red blob
    largest_contour = find_red_blob(contours)
    return largest_contour

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break
    
    # Define the lower and upper range for red color
    lower_red = np.array([0, 200, 50])
    upper_red = np.array([5, 255, 255])
    largest_contour = find_largest_blob(frame, lower_red, upper_red)

    if largest_contour is not None:
        # Draw a bounding box around the largest red blob
        x, y, w, h = cv2.boundingRect(largest_contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # downscale the frame to a percentage
    downscale = 0.25
    frame = cv2.resize(frame, None, fx=downscale, fy=downscale)
    #show 
    cv2.imshow('frame', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()