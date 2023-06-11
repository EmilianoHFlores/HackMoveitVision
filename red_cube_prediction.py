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

x_list = []
y_list = []
prev_x = None
prev_y = None
still_count = 0
start_time = 0
# Open the video capture
cap = cv2.VideoCapture('testvideo1cut.mp4')

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the frame to the HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the lower and upper range for red color
    lower_red = np.array([0, 200, 50])
    upper_red = np.array([5, 255, 255])

    # Threshold the frame to extract red regions
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # Find contours in the thresholded frame
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest red blob
    largest_contour = find_red_blob(contours)

    if largest_contour is not None:
        # Draw a bounding box around the largest red blob
        x, y, w, h = cv2.boundingRect(largest_contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # if the cube starts moving, store the center of the cube in x_list and y_list
    if prev_x is not None and prev_y is not None and x is not None and y is not None:
        if x != prev_x and y != prev_y:
            print("Detecting movement")
            #use the center of the image to store in x_list and y_list
            # store x inverted because the image is inverted
            x_list.append((x+w/2))
            y_list.append((y-h/2))
            #draw recorded points in white
            for i in range(0, len(x_list)):
                cv2.circle(frame, (int(x_list[i]), int(y_list[i])), 5, (255, 0, 255), 5)
            # make a prediction of the trajectory of the cube, using all the previous data and regression
                
            if len(x_list) > 4:
                model = np.poly1d(np.polyfit(x_list, y_list, 2))
                # draw the quadratic function on the frame
                # check if the model is valid
                for i in range(0, 1000):
                    cv2.circle(frame, (i, int(model(i))), 1, (0, 0, 255), 1)
        else:
            still_count += 1
            # if the cube is still for 10 frames, clear the x_list and y_list
            if still_count >= 60:
                still_count = 0
                x_list.clear()
                y_list.clear()
                print("Clearing data")
    # store the current x and y as previous x and y
    prev_x = x
    prev_y = y
    # downscale the frame 25%
    downscale = 1.0
    frame = cv2.resize(frame, None, fx=downscale, fy=downscale)
    #show 
    cv2.imshow('frame', frame)
    # delay to slow down 
    cv2.waitKey(100)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()



