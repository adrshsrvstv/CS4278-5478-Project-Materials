import cv2
import numpy as np
from numpy import linalg as la

red_lower = np.asarray([170, 75, 100])
red_upper = np.asarray([180, 255, 255])

red_lower_l = np.asarray([0, 75, 100])
red_upper_l = np.asarray([10, 255, 255])

yellow_lower = np.asarray([23, 25, 100])
yellow_upper = np.asarray([29, 255, 255])

white_lower = np.asarray([0, 0, 100])
white_upper = np.asarray([255, 40, 255])

DISTANCE_OFFSET = 15
DEGREES_OFFSET = 0

midpoint_lookdown = np.asarray([320, 150])
midpoint_lookahead = np.asarray([320, 80])
midpoint_initial = np.asarray([320, 330])

origin = np.asarray([0, 0])


def crop_looking_down(img):
    h, w, c = img.shape
    return img[150:h - 180, 0:w]


def crop_looking_ahead(img):
    h, w, c = img.shape
    return img[100:h - 300, 0:w]


def crop_for_initialization(img):
    h,w, c = img.shape
    return img[150:h, 0:w]


def get_yellow_mask(hsv_img):  # in HSV format
    mask_yellow = cv2.inRange(hsv_img, yellow_lower, yellow_upper)
    return mask_yellow


def get_white_mask(hsv_img):  # in HSV format
    mask_white = cv2.inRange(hsv_img, white_lower, white_upper)
    return mask_white


def get_red_mask(hsv_img):  # in HSV format
    mask_red = cv2.inRange(hsv_img, red_lower, red_upper)
    mask_red_l = cv2.inRange(hsv_img, red_lower_l, red_upper_l)
    red_mask = cv2.bitwise_or(mask_red, mask_red_l)
    return red_mask


def get_lane_mask(hsv_img):  # in HSV format
    mask_yellow = get_yellow_mask(hsv_img)
    mask_white = get_white_mask(hsv_img)
    lane_mask = cv2.bitwise_or(mask_white, mask_yellow)
    return lane_mask


def get_lane_and_stop_mask(hsv_img):  # in HSV format
    stop_mask = get_red_mask(hsv_img)
    lane_mask = get_lane_mask(hsv_img)
    lane_and_stop = cv2.bitwise_or(stop_mask, lane_mask)
    return lane_and_stop


def get_masked_image(img, hsv_img):  # returns parts of original RGB image that contain the lanes
    return cv2.bitwise_and(img, img, mask=get_lane_and_stop_mask(hsv_img))


def get_edges(hsv_img, color="yellow", blur=True):
    if color == "white":
        mask = get_white_mask(hsv_img)
    elif color == "yellow":
        mask = get_yellow_mask(hsv_img)
    elif color == "red":
        mask = get_red_mask(hsv_img)
    else:
        raise ValueError('Invalid color option for lane edges')

    if blur:
        mask = cv2.medianBlur(mask, 3)

    edges = cv2.Canny(mask, 200, 250)

    return edges


def preprocess(img):
    hsv_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    return crop_looking_down(hsv_image), crop_looking_ahead(hsv_image), crop_for_initialization(hsv_image)


def get_lane_lines(hsv_image, mode):
    yellow = get_closest_line(hsv_image, "yellow", mode)
    white = get_closest_line(hsv_image, "white", mode)
    red = get_closest_line(hsv_image, "red", mode)

    if (white is not None) and (yellow is not None) and (not is_yellow_to_left_of_white(yellow, white)):
        white = None

    return yellow, white, red


def get_lines_in_image(edges):
    lines = cv2.HoughLinesP(edges, rho=1, theta=1 * np.pi / 180, threshold=35, minLineLength=75, maxLineGap=50)
    if lines is None:
        lines = cv2.HoughLinesP(edges, rho=1, theta=1 * np.pi / 180, threshold=35, minLineLength=70, maxLineGap=50)
    if lines is None:
        lines = []
    return lines


def filter_outlier_lines(lines):
    slopes = [np.around(np.degrees(np.arctan(slope(line)))) for line in lines]
    mean = np.mean(slopes)
    std = np.std(slopes)

    filtered_lines = [line for line in lines if (abs(mean - np.around(np.degrees(np.arctan(slope(line))))) <= std)]
    return filtered_lines


def slope(line):
    x1, y1, x2, y2 = line[0]
    return (y2 - y1) / (x2 - x1 + 0.00000001)


def get_closest_line(hsv_img, color, mode):
    edges = get_edges(hsv_img, color=color, blur=True)
    lines = filter_outlier_lines(get_lines_in_image(edges))

    closest_line = None
    min_dist = 500
    for line in lines:
        dist = get_distance_from_line(line, mode)
        if dist < min_dist:
            closest_line = line
            min_dist = dist
    return closest_line


def get_distance_from_line(line, mode):
    x1, y1, x2, y2 = line[0]
    p1 = np.asarray([x1, y1])
    p2 = np.asarray([x2, y2])
    if mode == "lookahead":
        return np.around(np.abs(np.cross(p2 - p1, midpoint_lookahead - p1) / la.norm(p2 - p1)))
    elif mode == "lookdown":
        return np.around(np.abs(np.cross(p2 - p1, midpoint_lookdown - p1) / la.norm(p2 - p1)))
    elif mode == "initial":
        return np.around(np.abs(np.cross(p2 - p1, midpoint_initial - p1) / la.norm(p2 - p1)))
    else:
        raise ValueError("Invalid mode specified.")


# if both white and yellow are present, and if their slopes are same sign, yellow line needs to be closer to origin
# when initializing position, you might be in the other lane.

# so first anchor on yellow line, then see if there's a line on the other side of origin for this line

def is_yellow_to_left_of_white(yellow_line, white_line):
    o_x, o_y = origin
    x1, y1, x2, y2 = yellow_line[0]
    w_x1, w_y1, _, _ = white_line[0]
    cross_origin = (x2 - x1) * (o_y - y1) - (y2 - y1) * (o_x - x1)
    cross_white = (x2 - x1) * (w_y1 - y1) - (y2 - y1) * (w_x1 - x1)
    return cross_origin * cross_white < 0


def get_d_est(yellow, white, mode):
    if yellow is not None and white is not None:
        d_est = (get_distance_from_line(yellow, mode) - get_distance_from_line(white, mode))/2
    elif white is not None:
        d_est = one_side_distance_offset(mode)-get_distance_from_line(white, mode)
    elif yellow is not None:
        d_est = get_distance_from_line(yellow, mode) - one_side_distance_offset(mode)
    else:
        return 0
    return d_est + DISTANCE_OFFSET

def one_side_distance_offset(mode):
    if mode == "lookdown":
        return 200
    if mode == "initial":
        return 360
    if mode == "lookahead":
        return 95

def get_heading(yellow, white):
    if (yellow is not None) and (white is not None):
        heading = np.degrees(np.arctan((slope(white) + slope(yellow)) / 2))
    elif (yellow is not None) and slope(yellow) <= 0:
        heading = np.degrees(np.arctan(slope(yellow))) - 90 + DEGREES_OFFSET
    elif (yellow is not None) and slope(yellow) > 0:
        heading = np.degrees(np.arctan(slope(yellow))) + 90 - DEGREES_OFFSET
    elif (white is not None) and slope(white) <= 0:
        heading = 270 - np.degrees(np.arctan(slope(white))) + DEGREES_OFFSET
    elif (white is not None) and slope(white) > 0:
        heading = 90 - np.degrees(np.arctan(slope(white))) - DEGREES_OFFSET
    else:
        heading = 0  # if you can't see yellow or white, keep going straight
    return np.radians(np.around(heading))


def is_line_to_left_of_centre(yellow):
    x1, y1, x2, y2 = yellow[0]
    if y1 > y2:
        return x1 < 320
    else:
        return x2 < 320