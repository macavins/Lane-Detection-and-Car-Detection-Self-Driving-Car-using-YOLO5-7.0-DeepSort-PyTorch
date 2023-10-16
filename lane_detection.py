import numpy as np
import cv2
import glob
import os
import sys
from math import atan, pi
# import yolov5
import torch
# from yolov5.models.experimental import attempt_load
# from yolov5.utils.general import non_max_suppression
# from torchvision.models.detection import fasterrcnn_resnet50_fpn

# # Import YOLO-related modules and functions
# import yolov5
#
# # ...
#
# # Initialize YOLO model
# yolo_model = yolov5.load("yolov5s.pt", map_location='cpu')  # Load your YOLO model file
# frame = cv2.VideoCapture('./yolov5-7.0/data/video/test_sample.mp4')
# # ...
#
# # Inside your main loop, after capturing a frame:
# # Perform object detection using YOLO on the captured frame
# with torch.no_grad():
#     yolo_results = yolo_model(frame)
#
# # Filter and annotate detected vehicles
# for det in yolo_results[0]:
#     if det is not None and len(det):
#         det[:, :4] = det[:, :4].clip(0, frame.shape[2])
#         for *xyxy, conf, cls in det:
#             label = f'{yolo_model.names[int(cls)]} {conf:.2f}'
#             xyxy = [int(i) for i in xyxy]
#             cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
#             cv2.putText(frame, label, (xyxy[0], xyxy[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
#

# Step 1: Camera Calibration

def distortion_factors():
    # pointers
    # from cali imgs, 9*6 corners are identified
    x = 11  # 9, 11
    y = 8  # 6, 8
    obj_points = []  # 3D coordinates matrices
    img_points = []
    # x, y are equal distances while z coordinates are 0

    obj_point = np.zeros((8 * 11, 3), np.float32)  # zero out
    obj_point[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)

    # list calibration images here
    os.listdir("camera_cal/")
    calib_img_list = os.listdir("camera_cal/")

    # Image points are the corresponding object points with their coordinates in the distorted image
    # They are found in the image using the Open CV 'findChessboardCorners' function

    for img_name in calib_img_list:
        import_from = 'camera_cal/' + img_name
        img = cv2.imread(import_from)

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find chess corners
        ret, corners = cv2.findChessboardCorners(gray, (x, y), None)

        # if corners found then draw corners
        if ret == True:
            # Draw and Display corners
            cv2.drawChessboardCorners(img, (x, y), corners, ret)
            img_points.append(corners)
            obj_points.append(obj_point)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

    # Check undistorted image
    # for img_name in cal_img_list:
    #     import_from = 'camera_cal/' + img_name
    #     img = cv2.imread(import_from)
    #     undist = cv2.undistort(img, mtx, dist, None, mtx)
    #     export_to = 'camera_cal_undistorted/' + img_name
    #     #save the image in the destination folder#
    #     plt.imsave(export_to, undist)

    return mtx, dist


# Step 2: Transform car Camera to Bird's Eye View
# img_width = 1280
# img_heigt = 720

def warp(img, mtx, dist):
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    img_size = (img.shape[1], img.shape[0])
    offset = 150

    # Source points taken from images with straight lane lines,
    # these are to become parallel after the warp transform
    # src = np.float32([
    #     (350, 1080), # bottom-left corner
    #     (845, 700), # top-left corner
    #     (1020, 700), # top-right corner
    #     (1560, 1080) # bottom-right corner
    # ])

    src_points = np.float32([
        (317, 720),  # bottom-left corner
        (559, 457),  # top-left corner
        (671, 457),  # top-right corner
        (1026, 720)  # bottom-right corner
    ])
    # src = np.float32([
    #     (int(img_size[0]*350/1920), int(img_size[1]*1080/1080)), # bottom-left corner
    #     (int(img_size[0]*845/1920), int(img_size[1]*700/1080)), # top-left corner
    #     (int(img_size[0]*1020/1920), int(img_size[1]*700/1080)), # top-right corner
    #     (int(img_size[0]*1560/1920), int(img_size[1]*1080/1080)) # bottom-right corner
    # ])
    # Destination points are to be parallel, taken into account the image size
    dst = np.float32([
        [offset, img_size[1]],  # bottom-left corner
        [offset, 0],  # top-left corner
        [img_size[0] - offset, 0],  # top-right corner
        [img_size[0] - offset, img_size[1]]  # bottom-right corner
    ])

    # Calculate the transformation matrix and it's inverse transformation
    tm = cv2.getPerspectiveTransform(src_points, dst)
    tm_inv = cv2.getPerspectiveTransform(dst, src_points)
    warped = cv2.warpPerspective(undist, tm, img_size)

    return warped, tm_inv, undist


def binary_threshold(img):
    # Transform img to gray scale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply derivative in x direction. To detect vertical lines
    dx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0)  # dx = sobelx
    abs_dx = np.absolute(dx)
    # Scale result to 0-255
    scaled_d = np.uint8(255 * abs_dx / np.max(abs_dx))
    dx_binary = np.zeros_like(scaled_d)
    # Keep only derivative values that are in the margin of interest
    dx_binary[(scaled_d >= 30) & (scaled_d <= 255)] = 1

    # Detect pixels that are white in the grayscale image
    white_binary = np.zeros_like(gray_img)
    white_binary[(gray_img > 200) & (gray_img <= 255)] = 1  # 200,255

    # Convert img to HLS
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    H = hls[:, :, 0]
    S = hls[:, :, 2]
    sat_binary = np.zeros_like(S)
    # Detect pixels that have a high saturation value
    sat_binary[(S > 200) & (S <= 255)] = 1  # 90 , 255

    hue_binary = np.zeros_like(H)
    # Detect pixels that are yellow using the hue component
    hue_binary[(H > 15) & (H <= 25)] = 1  # 10, 25

    # Combine all pixels detected above
    binary_1 = cv2.bitwise_or(dx_binary, white_binary)
    binary_2 = cv2.bitwise_or(hue_binary, sat_binary)
    binary = cv2.bitwise_or(binary_1, binary_2)

    return binary


# Step 4: Detection of Lane Lines Using Histogram

def find_lane_pixels_using_histogram(binary_warped):
    res_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    window_img = np.zeros_like(res_img)

    # Take a histogram of bottom half of shape of img
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)

    # Find the peak of the left and right halves of the histogram
    # These are start of left and right lines
    midpoint = int(histogram.shape[0] // 2)
    left_x_base = np.argmax(histogram[:midpoint])
    right_x_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = int(1280 * (100 / 1920))
    # set minimum number of pixels found to recenter window
    minpix = int(1280 * (50 / 1920))

    # Set height of windows - based on nwindows above and image shape
    window_height = int(binary_warped.shape[0] // nwindows)
    # Identify x and y pos of all nonzero pixels in ing
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current pos to be updated later for each window in nwindows
    left_x_cur = left_x_base
    right_x_cur = right_x_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_idx = []
    right_lane_idx = []

    # Step through the windows 1 by 1
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_x_left_low = left_x_cur - margin
        win_x_left_high = left_x_cur + margin
        win_x_right_low = right_x_cur - margin
        win_x_right_high = right_x_cur + margin

        # Identify the nonzero pixels in x and y within the window
        good_left_idx = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_x_left_low) & (nonzerox < win_x_left_high)).nonzero()[0]
        good_right_idx = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_x_right_low) & (nonzerox < win_x_right_high)).nonzero()[0]

        # Append these indices to list
        left_lane_idx.append(good_left_idx)
        right_lane_idx.append(good_right_idx)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_idx) > minpix:
            left_x_cur = int(np.mean(nonzerox[good_left_idx]))
        if len(good_right_idx) > minpix:
            right_x_cur = int(np.mean(nonzerox[good_right_idx]))

    # Concatenate the arrays of indices (prev. a list of lists of pixels)
    try:
        left_lane_idx = np.concatenate(left_lane_idx)
        right_lane_idx = np.concatenate(right_lane_idx)
    except ValueError:
        pass

    # Extract left and right line pixel pos
    left_x = nonzerox[left_lane_idx]
    left_y = nonzeroy[left_lane_idx]
    right_x = nonzerox[right_lane_idx]
    right_y = nonzeroy[right_lane_idx]

    return left_x, left_y, right_x, right_y


def fit_poly(binary_warped, left_x, left_y, right_x, right_y):
    # Fit a 2nd order polynomial to each line
    left_fit = np.polyfit(left_y, left_x, 2)
    right_fit = np.polyfit(right_y, right_x, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    try:
        left_fit_x = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fit_x = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    except TypeError:
        print('failed to fit. Oh well')
        left_fit_x = 1*ploty**2 + 1*ploty
        right_fit_x = 1*ploty**2 + 1*ploty

    return left_fit, right_fit, left_fit_x, right_fit_x, ploty


def draw_poly_lines(binary_warped, left_fit_x, right_fit_x, ploty):
    # Create an image to draw on and an image to show the selection window
    res_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(res_img)

    margin = int(1280 * (100 / 1920))
    # Create Polygon to illustrate the search window
    # Recast the x, y points into usable format for cv2
    left_line_window1 = np.array([np.transpose(np.vstack([left_fit_x - margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fit_x + margin, ploty])))])
    left_line_points = np.hstack((left_line_window1, left_line_window2))

    right_line_window1 = np.array([np.transpose(np.vstack([right_fit_x - margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fit_x + margin, ploty])))])
    right_line_points = np.hstack((right_line_window1, right_line_window2))

    # Center line
    center_line_points = (left_line_points + right_line_points) / 2

    # Draw the lane onto the warped blank img
    cv2.fillPoly(window_img, np.int32([left_line_points]), (100, 100, 0))
    cv2.fillPoly(window_img, np.int32([right_line_points]), (100, 100, 0))
    cv2.fillPoly(window_img, np.int32([center_line_points]), (200, 100, 0))

    result = cv2.addWeighted(res_img, 1, window_img, 0.9, 0)

    return result


# Step 5: Detection of Lane Lines Based on prev step

def find_lane_pixels_using_prev_poly(binary_warped):
    # Width of margin around prev polynomial to search
    margin = int(1280 * (100 / 1920))
    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Area of search from activated x-vals
    # Within =/- margin of polynomial function


    left_lane_idx = ((nonzerox > (prev_left_fit[0]*(nonzeroy**2) + prev_left_fit[1]*nonzeroy +
                    prev_left_fit[2] - margin)) & (nonzerox < (prev_left_fit[0]*(nonzeroy**2) +
                    prev_left_fit[1]*nonzeroy + prev_left_fit[2] + margin))).nonzero()[0]

    right_lane_idx = ((nonzerox > (prev_right_fit[0]*(nonzeroy**2) + prev_right_fit[1]*nonzeroy +
                    prev_right_fit[2] - margin)) & (nonzerox < (prev_right_fit[0]*(nonzeroy**2) +
                    prev_right_fit[1]*nonzeroy + prev_right_fit[2] + margin))).nonzero()[0]

    left_x = nonzerox[left_lane_idx]
    left_y = nonzeroy[left_lane_idx]
    right_x = nonzerox[right_lane_idx]
    right_y = nonzeroy[right_lane_idx]

    return left_x, left_y, right_x, right_y


# Step 6: Calculate car pos and curve radius

def measure_curvature_meters(binary_warped, left_fit_x, right_fit_x, ploty):
    # Define x and y from pixels to meters conversion
    m_per_pixy = 30 / 1080 * (720 / 1080)
    m_per_pixx = 3.7 / 1920 * (1280 / 1920)

    left_fit_cr = np.polyfit(ploty * m_per_pixy, left_fit_x * m_per_pixx, 2)
    right_fit_cr = np.polyfit(ploty * m_per_pixy, right_fit_x * m_per_pixx, 2)

    # y val radius of curve
    y_eval = np.max(ploty)

    # Radius of curvature
    left_curve = ((1+(2*left_fit_cr[0]*y_eval*m_per_pixy+left_fit_cr[1])**2)**1.5)/np.absolute(2*left_fit_cr[0])
    right_curve = ((1+(2*right_fit_cr[0]*y_eval*m_per_pixy+right_fit_cr[1])**2)**1.5)/np.absolute(2*right_fit_cr[0])

    return left_curve, right_curve


def measure_pos_meters(binary_warped, left_fit, right_fit):
    # Define x pixels to meters
    m_per_pixx = 3.7/1920 * (1280/1920)
    # Choose y val to the bottom of img
    y_max = binary_warped.shape[0]
    # Calc left and right line positons at the bottomm of the img
    left_x_pos = left_fit[0]*y_max**2 + left_fit[1]*y_max + left_fit[2]
    right_x_pos = right_fit[0]*y_max**2 + right_fit[1]*y_max + right_fit[2]
    # x pos of the center of the lane
    center_lanes_x_pos = (left_x_pos + right_x_pos)//2
    # Deviation between the center of lane and center of pic
    # Car is center of pic
    # if the deviation is negative, the car is on felt hand side of the center of the lane
    car_pos = ((binary_warped.shape[1]//2) - center_lanes_x_pos) * m_per_pixx
    return car_pos


# Step 7: Delimitations back on img plane and add lane info text

def project_lane_info(img, binary_warped, ploty, left_fit_x, right_fit_x, tm_inv, left_curve, right_curve, car_pos):
    # Create an img and draw the lines
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    warp_color = np.dstack((warp_zero, warp_zero, warp_zero))

    # Center line modified
    margin = 400 * (1280 / 1920)
    # x, y points usable format in cv2
    points_left = np.array([np.transpose(np.vstack([left_fit_x, ploty]))])
    points_right = np.array(([np.flipud(np.transpose(np.vstack([right_fit_x, ploty])))]))

    points_left_center = np.array([np.transpose(np.vstack([left_fit_x + margin + 150, ploty]))])
    points_right_center = np.array([np.flipud(np.transpose(np.vstack([right_fit_x - margin, ploty])))])
    points_center = np.hstack((points_left_center, points_right_center))

    points_left_i = np.array([np.transpose(np.vstack([left_fit_x + margin + 150, ploty]))])
    points_right_i = np.array([np.flipud(np.transpose(np.vstack([right_fit_x - margin - 150, ploty])))])
    points_i = np.hstack((points_left_i, points_right_i))

    # Draw the lane onto the warped blank image
    colorwarp_img = cv2.polylines(warp_color, np.int_([points_left]), False, (0, 0, 255), 50)
    colorwarp_img = cv2.polylines(warp_color, np.int_([points_right]), False, (0, 0, 255), 50)
    colorwarp_img = cv2.fillPoly(warp_color, np.int_([points_center]), (0, 255, 0))

    # Warp the blank bak to original img space using inverse perspective maxtrix
    new_warp = cv2.warpPerspective(warp_color, tm_inv, (img.shape[1], img.shape[0]))

    # Combine the result with original img
    res_img = cv2.addWeighted(img, 0.7, new_warp, 0.3, 0)

    cv2.putText(res_img, 'Curve Radius [m]: ' + str((left_curve + right_curve) / 2)[:7], (5, 80), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(res_img, 'Center Offset [m]: ' + str(car_pos)[:7], (5, 110), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (255, 255, 255), 2, cv2.LINE_AA)

    return res_img, colorwarp_img, new_warp


# Step 8: Lane Finding Pipeline on Vid
def lane_finding_pipeline(img, init, mts, dist):
    global left_fit_hist
    global right_fit_hist
    global prev_left_fit
    global prev_right_fit
    # right_fit_x = None  # Initialize right_fit_x to None

    # # Car detection
    # car_bounding_boxes = car_detection(img, car_model)

    if init:
        left_fit_hist = np.array([])
        right_fit_hist = np.array([])
        prev_left_fit = np.array([])
        prev_right_fit = np.array([])

    binary_thresh = binary_threshold(img)
    binary_warped, tm_inv, _ = warp(binary_thresh, mts, dist)

    # Check
    binary_thresh_s = np.dstack((binary_thresh, binary_thresh, binary_thresh)) * 255
    binary_warped_s = np.dstack((binary_warped, binary_warped, binary_warped)) * 255

    if (len(left_fit_hist) == 0):
        left_x, left_y, right_x, right_y = find_lane_pixels_using_histogram(binary_warped)
        left_fit, right_fit, left_fit_x, right_fit_x, ploty = fit_poly(binary_warped, left_x, left_y, right_x, right_y)

        # Store fit in history
        left_fit_hist = np.array(left_fit)
        new_left_fit = np.array(left_fit)
        left_fit_hist = np.vstack([left_fit_hist, new_left_fit])

        right_fit_hist = np.array(right_fit)
        new_right_fit = np.array(right_fit)
        right_fit_hist = np.vstack([right_fit_hist, new_right_fit])

    else:
        prev_left_fit = [np.mean(left_fit_hist[:, 0]), np.mean(left_fit_hist[:, 1]), np.mean(left_fit_hist[:, 2])]
        prev_right_fit = [np.mean(right_fit_hist[:, 0]), np.mean(right_fit_hist[:, 1]), np.mean(right_fit_hist[:, 2])]
        left_x, left_y, right_x, right_y = find_lane_pixels_using_prev_poly(binary_warped)
        if (len(left_y) == 0 or len(right_y) == 0):
            left_x, left_y, right_x, right_y = find_lane_pixels_using_histogram(binary_warped)
        left_fit, right_fit, left_fit_x, right_fit_x, ploty = fit_poly(binary_warped, left_x, left_y, right_x, right_y)

        # Add new vals to the history
        new_left_fit = np.array(left_fit)
        left_fit_hist = np.vstack([left_fit_hist, new_left_fit])
        new_right_fit = np.array(right_fit)
        right_fit_hist = np.vstack([right_fit_hist, new_right_fit])

        # Remove old vals from history
        if (len(left_fit_hist) > 5):
            left_fit_hist = np.delete(left_fit_hist, 0, 0)
            right_fit_hist = np.delete(right_fit_hist, 0, 0)

    # Check
    draw_poly_img = draw_poly_lines(binary_warped, left_fit_x, right_fit_x, ploty)
    left_curve, right_curve = measure_curvature_meters(binary_warped, left_fit_x, right_fit_x, ploty)
    car_pos = measure_pos_meters(binary_warped, left_fit, right_fit)
    res_img, colorwarp_img, new_warp = project_lane_info(img, binary_warped, ploty, left_fit_x, right_fit_x, tm_inv, left_curve, right_curve, car_pos)

    return res_img, car_pos, colorwarp_img, draw_poly_img # , car_bounding_boxes


def main():
    cap = cv2.VideoCapture('./yolov5-7.0/data/video/test_sample_winner.mp4')  # Put sample
    # cap = cv2.VideoCapture('./yolov5-7.0/data/video/test_sample.mp')
    if not cap.isOpened():
        print('File could not be opened oof')
        cap.release()
        sys.exit()

    # Video out
    width = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps)

    angle = 0
    img_steering = cv2.imread('steering_wheel_image.jpg')
    rows, cols, ext = img_steering.shape

    # Create the video writer object
    res = cv2.VideoWriter('./yolov5/data/output/result_output_lane.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps,
                          (width, height))

    init = True
    mtx, dist = distortion_factors()

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        res_out, angle, colorwarp, draw_poly_img = lane_finding_pipeline(frame, init, mtx, dist)

        if angle > 1.5 or angle < - 1.5:
            init = True
        else:
            init = False

        # Steering img
        angle = atan((180 / pi) * (angle / 5))
        tm = cv2.getRotationMatrix2D((cols / 2, rows / 2), -angle * 10, 1)
        dst = cv2.warpAffine(img_steering, tm, (cols, rows))

        height, width, channel = dst.shape
        height1, width1, channel1 = res_out.shape
        res_out[(height1-height):height1, int(width1/2-width/2):(int(width1/2-width/2)+width)] = dst


        # Videowirte
        res.write(res_out)

        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        cv2.imshow('frame', res_out)
        cv2.namedWindow('colorwarp', cv2.WINDOW_NORMAL)
        cv2.imshow('colorwarp', colorwarp)
        cv2.namedWindow('draw_poly', cv2.WINDOW_NORMAL)
        cv2.imshow('draw_poly', draw_poly_img)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    res.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
