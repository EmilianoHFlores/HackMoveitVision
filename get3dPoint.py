#! /usr/bin/env python3

import rospy 
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2 
import time
from vision_utils import find_largest_blob, get_depth, deproject_pixel_to_point
from sensor_msgs.msg import CompressedImage, Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Point, PointStamped
import numpy as np

class SimplePubSub():
    def __init__(self):
        super().__init__('simple_sub')

        topic_name= '/head_front_camera/rgb/image_raw'

        self.cap = cv2.VideoCapture(0)
        self.br = CvBridge()

        self.subscription = rospy.Subscriber(topic_name, Image, self.img_callback, 10)
        self.depthsubscription = rospy.Subscriber('/head_front_camera/depth_registered/image_raw', Image, self.depth_callback, 10)
        self.infosubscription = rospy.Subscriber('/head_front_camera/rgb/camera_info', CameraInfo, self.info_callback, 1)
        self.pointPublisher = rospy.Publisher('ballTopic', Point, 1)
        self.stampedPointPublisher = rospy.Publisher('ballTopicStamped', PointStamped, 10)
        self.subscription
        self.br = CvBridge()

        self.stallCount = 0
        self.x_list = []
        self.y_list = []
        self.z_list = []
        self.t_list = []
        self.curr_time = 0
        print("Waiting for image")
    
    def depth_callback(self, data):
        self.depthImage = self.br.imgmsg_to_cv2(data, "32FC1")
        
    def info_callback(self, data):
        self.cameraInfo = data
        #print(data)

    def img_callback(self, data):
        #self.get_logger().info('Receiving video frame')
        frame = self.br.imgmsg_to_cv2(data)   
        lower_red = np.array([0, 200, 50])
        upper_red = np.array([5, 255, 255])
        largest_blob = find_largest_blob(frame, lower_red, upper_red)
        x = None
        y = None
        w = None
        h = None
        if largest_blob is not None:
            #print("Found a red blob")
            # Draw a bounding box around the largest red blob
            x, y, w, h = cv2.boundingRect(largest_blob)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # deproject the center of the blob
        else:
            self.x_list = []
            self.y_list = []
            self.z_list = []
            self.t_list = []
        # get the depth image
        try:

            depth = get_depth(self.depthImage, [int(x+w/2), int(y+h/2)])
            #print("Depth: ", depth)
            # point3D_ = deproject_pixel_to_point(self.imageInfo, point2D, depth)
            point3D_ = deproject_pixel_to_point(self.cameraInfo, [int(x+w/2), int(y+h/2)], depth)
            point3D = Point()
            point3D.x = point3D_[0]
            point3D.y = point3D_[1]
            point3D.z = float(point3D_[2])
            #print(point3D)
            self.pointPublisher.publish(point3D)
            point3DStamped = PointStamped()
            point3DStamped.header.frame_id = "head_front_camera_rgb_optical_frame"
            point3DStamped.point = point3D
            #print(point3DStamped)
            self.stampedPointPublisher.publish(point3DStamped)

            # making the prediction model
            self.x_list.append(point3D.x)
            self.y_list.append(point3D.y)
            self.z_list.append(point3D.z)
            self.t_list.append(self.curr_time)

            if len(self.x_list) > 10:
                print("List size ", len(self.x_list))
                # model can be made
                model_tx = np.polyfit(self.t_list, self.x_list, 1)
                model_tz = np.polyfit(self.t_list, self.z_list, 1)
                print(model_tz)
                # y model is quadratic because of gravity
                model_ty = np.polyfit(self.t_list, self.y_list, 2)
                #catch_time = self.get_closest_point(model_tx, model_ty, model_tz)
                catch_time = self.get_at_z(1, model_tx, model_ty, model_tz)
                print("Catch time: ", catch_time)
                estimated_x = model_tx(catch_time)
                estimated_y = model_ty(catch_time)
                estimated_z = model_tz(catch_time)
                print(f"Estimated x: {estimated_x}, y: {estimated_y}, z: {estimated_z}")

        except Exception as e:
            print("No depth image")
            print(e)

        self.curr_time+=1
        # show self.currtime in frame
        cv2.putText(frame, str(self.curr_time), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("camera", frame)
        cv2.waitKey(1)

    def get_closest_point(self, model_x, model_y, model_z):
        # make a list of the distances from the reference frame using the model
        distances = []
        # find the index of the minimum distance
        min_index = distances.index(min(distances))
        # find the time at which the ball is closest to the reference frame
        min_time = self.t_list[min_index]
        return min_time
    
    def get_at_z(self, z, model_x, model_y, model_z):
        # find the time at which the ball is at z
        response_time = -(model_z[1] - z) / model_z[0]
        print("get at z answers: ", response_time)
        # get biggest real root
        response_time = max(response_time)
        return response_time
    
    def run(self):
        #print every 1 second
        prevtime = time.time()
        while True:
            if time.time() - prevtime >= 1:
                print("1 second has passed")
                prevtime = time.time()
        pass


def main(args=None):
    while not rospy.is_shutdown:
        SimplePubSub()
    print("Closing")

if __name__ == '__main__':
    main()