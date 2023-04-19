import math
from enum import Enum
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


DISTANCE_OFFSET = 5  # analytically optimal

building_lower = np.array([5, 0, 5])
building_upper = np.array([255, 10, 255])

DEGREES_OFFSET = 0
DOUBLE_LINE_HEADING_OFFSET = 3

midpoint_lookdown = np.asarray([320, 150])
midpoint_lookahead = np.asarray([320, 80])
midpoint_initial = np.asarray([320, 330])

origin = np.asarray([0, 0])


class Mode(str, Enum):
    LOOKDOWN = "lookdown"
    LOOKAHEAD = "lookahead"
    INITIAL = "initial"


class State(Enum):
    INITIALIZING = 1
    IN_LANE_AND_FORWARD = 2
    IN_LANE_AND_WAITING_TO_TURN = 3
    IN_LANE_USING_RED = 4
    TURNING_WITHIN_LANE = 5
    TURNING_AT_INTERSECTION = 6
    CROSSING_INTERSECTION = 7


def get_mode_from_state(state):
    if state == State.INITIALIZING:
        return Mode.INITIAL
    elif state == State.IN_LANE_AND_FORWARD:
        return Mode.LOOKDOWN
    elif state == State.IN_LANE_AND_WAITING_TO_TURN:
        return Mode.LOOKDOWN
    elif state == State.TURNING_WITHIN_LANE:
        return Mode.LOOKDOWN
    elif state == State.IN_LANE_USING_RED:
        return Mode.INITIAL
    elif state == State.TURNING_AT_INTERSECTION:
        return Mode.LOOKDOWN
    elif state == State.CROSSING_INTERSECTION:
        return Mode.LOOKDOWN
    else:
        raise ValueError("Invalid state:", state)


def crop_looking_down(img):
    h, w, c = img.shape
    return img[150:h - 180, 0:w]


def crop_looking_ahead(img):
    h, w, c = img.shape
    return img[100:h - 300, 0:w]


def crop_for_initialization(img):
    h, w, c = img.shape
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

def get_building_mask(hsv_img): #in HSV format
    mask_building = cv2.inRange(hsv_img, building_lower, building_upper)
    return mask_building


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


def preprocess(img, state):
    mode = get_mode_from_state(state)
    hsv_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    if mode == Mode.LOOKDOWN:
        return crop_looking_down(hsv_image)
    elif mode == Mode.LOOKAHEAD:
        return crop_looking_ahead(hsv_image)
    elif mode == Mode.INITIAL:
        return crop_for_initialization(hsv_image)
    else:
        raise ValueError("Invalid Mode.")


def get_lane_lines(img, state):
    hsv_image = preprocess(img, state)
    red = get_closest_line(hsv_image, "red", state)
    yellow = get_closest_line(hsv_image, "yellow", state, red_line_for_reference=red)
    white = get_closest_line(hsv_image, "white", state, yellow_line_for_reference=yellow, red_line_for_reference=red)
    return yellow, white, red


def get_lines_in_image(edges):
    lines = cv2.HoughLinesP(edges, rho=1, theta=1 * np.pi / 180, threshold=35, minLineLength=75, maxLineGap=20)
    if lines is None:
        lines = cv2.HoughLinesP(edges, rho=1, theta=1 * np.pi / 180, threshold=35, minLineLength=65, maxLineGap=20)
    if lines is None:
        lines = []
    return lines


def filter_outlier_lines(lines):
    if (lines is None) or len(lines) == 0 or len(lines) == 1:
        filtered_lines = lines
    elif (len(lines) == 2) and abs(np.degrees(np.arctan(slope(lines[0]))) - np.degrees(np.arctan(slope(lines[1])))) > 30:
        filtered_lines = [lines[0]] if get_length(lines[0]) > get_length(lines[1]) else [lines[1]]
    else:
        slopes = [abs(np.around(angle_in_degrees(line))) for line in lines]
        mean = np.mean(slopes)
        std = np.std(slopes)
        filtered_lines = [line for line in lines if (abs(mean - abs(np.around(angle_in_degrees(line)))) <= min(2 * std, 15))]
    return filtered_lines


def get_length(line):
    x1, y1, x2, y2 = line[0]
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def slope(line):
    x1, y1, x2, y2 = line[0]
    return (y2 - y1) / (x2 - x1 + 0.00000001)


def angle_in_degrees(line):
    return np.degrees(np.arctan(slope(line)))


def filter_on_color_based_on_state(lines, color, state, yellow_line_for_reference=None, red_line_for_reference=None):
    if state == State.INITIALIZING:
        if color == 'red':
            pass
        if color == 'white':
            pass
        if color == 'yellow':
            if red_line_for_reference is not None:
                # Avoid yellow lines that are almost parallel with red
                lines = [line for line in lines if abs(angle_in_degrees(line)-angle_in_degrees(red_line_for_reference)) > 18]
            regular_yellow_lines = [line for line in lines if angle_in_degrees(line) < -12]
            if len(regular_yellow_lines) != 0:
                lines = regular_yellow_lines
    elif state == State.IN_LANE_AND_FORWARD:
        if color == 'red':
            # avoid detecting non-horizontal lines
            lines = [line for line in lines if abs(angle_in_degrees(line)) < 15]
            # avoid red lines that are maybe horizontal but are entirely to the left of the image - they belong to opposite lane ahead of intersection
            lines = [line for line in lines if (line[0][0] > 320 or line[0][2] > 320)]
        if color == 'white':
            # avoid detecting almost horizontal lines, such as of a perpendicular lane, but also lines from lane next to current pos
            lines = [line for line in lines if angle_in_degrees(line) > 15]
            # avoid detecting lines that are not to the right of yellow if yellow is present
            if yellow_line_for_reference is not None:
                lines = [line for line in lines if is_yellow_to_left_of_white(yellow_line_for_reference, line)]
        if color == 'yellow':
            lines = [line for line in lines if angle_in_degrees(line) < -12]
    elif state == State.IN_LANE_AND_WAITING_TO_TURN:
        if color == 'red':
            # avoid detecting non-horizontal lines
            lines = [line for line in lines if abs(angle_in_degrees(line)) < 15]
        if color == 'white':
            if red_line_for_reference is not None:
                # avoid detecting almost horizontal lines, such as of a perpendicular lane, but also lines from lane next to current pos
                lines = [line for line in lines if angle_in_degrees(line) > 15]
            if yellow_line_for_reference is not None:
                lines = [line for line in lines if is_yellow_to_left_of_white(yellow_line_for_reference, line)]
        if color == 'yellow':
            lines = [line for line in lines if (get_length(line) > 85 or angle_in_degrees(line) < -4)]  # reconsider
    elif state == State.TURNING_WITHIN_LANE:
        if color == 'red':
            pass
        if color == 'white':
            # avoid detecting lines that are not to the right of yellow if yellow is present
            if yellow_line_for_reference is not None:
                lines = [line for line in lines if is_yellow_to_left_of_white(yellow_line_for_reference, line)]
        if color == 'yellow':
            lines = [line for line in lines if abs(angle_in_degrees(line)) > 5]  # reconsider
    elif state == State.IN_LANE_USING_RED:
        if color == 'red':
            # avoid red lines that are not nearly horizontal - for example, fom a perpendicular lane
            lines = [line for line in lines if abs(angle_in_degrees(line)) < 10]
            # avoid red lines that are maybe horizontal but are entirely to the left of the image - they belong to opposite lane ahead of intersection
            lines = [line for line in lines if (line[0][0] > 320 or line[0][2] > 320)]
        if color == 'white':
            # avoid the ghost white line on edge of red lane
            if red_line_for_reference is not None:
                lines = [line for line in lines if abs(angle_in_degrees(line) - np.degrees(np.arctan(slope(red_line_for_reference)))) > 15]
        if color == 'yellow':
            # Avoid yellow lines that are almost parallel to red line
            if red_line_for_reference is not None:
                lines = [line for line in lines if abs(angle_in_degrees(line) - np.degrees(np.arctan(slope(red_line_for_reference)))) > 15]
    elif state == State.CROSSING_INTERSECTION:
        if color == 'red':
            # avoid red lines that are not nearly horizontal - for example, fom a perpendicular lane
            # lines = [line for line in lines if abs(angle_in_degrees(line)) < 10]
            # avoid red lines that are maybe horizontal but are entirely to the left of the image - they belong to opposite lane ahead of intersection
            lines = [line for line in lines if (line[0][0] > 320 or line[0][2] > 320)]
        if color == 'white':
            # avoid detecting almost horizontal lines, such as of a perpendicular lane, but also lines from lane next to current pos
            lines = [line for line in lines if angle_in_degrees(line) > 15]
            # avoid detecting lines that are not to the right of yellow if yellow is present
            if yellow_line_for_reference is not None:
                lines = [line for line in lines if is_yellow_to_left_of_white(yellow_line_for_reference, line)]
        if color == 'yellow':
            # avoid detecting almost horizontal lines, such as of a perpendicular lane
            lines = [line for line in lines if angle_in_degrees(line) < -7]
    elif state == State.TURNING_AT_INTERSECTION:
        if color == 'red':
            pass
        if color == 'white':
            # avoid detecting almost horizontal lines
            lines = [line for line in lines if angle_in_degrees(line) > 15]
        if color == 'yellow':
            pass
    else:
        raise ValueError("Invalid State:", state)

    return np.asarray(lines)


def get_closest_line(hsv_img, color, state, yellow_line_for_reference=None, red_line_for_reference=None):
    edges = get_edges(hsv_img, color=color, blur=True)
    lines = get_lines_in_image(edges)
    lines = filter_on_color_based_on_state(lines, color, state, yellow_line_for_reference, red_line_for_reference)
    lines = filter_outlier_lines(lines)

    closest_line = None
    min_dist = 5000000
    for line in lines:
        dist = get_distance_from_line(line, get_mode_from_state(state))
        if dist < min_dist:
            closest_line = line
            min_dist = dist
    return closest_line


def get_distance_from_line(line, mode):
    x1, y1, x2, y2 = line[0]
    p1 = np.asarray([x1, y1])
    p2 = np.asarray([x2, y2])
    if mode == Mode.LOOKAHEAD:
        return np.around(np.abs(np.cross(p2 - p1, midpoint_lookahead - p1) / la.norm(p2 - p1)))
    elif mode == Mode.LOOKDOWN:
        return np.around(np.abs(np.cross(p2 - p1, midpoint_lookdown - p1) / la.norm(p2 - p1)))
    elif mode == Mode.INITIAL:
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
        d_est = (get_distance_from_line(yellow, mode) - get_distance_from_line(white, mode)) / 2
    elif white is not None:
        d_est = one_side_distance_offset(mode) - get_distance_from_line(white, mode)
    elif yellow is not None:
        d_est = get_distance_from_line(yellow, mode) - one_side_distance_offset(mode)
    else:
        return 0  # could be intersection
    return d_est + DISTANCE_OFFSET


def one_side_distance_offset(mode):
    if mode == Mode.LOOKDOWN:
        return 200
    if mode == Mode.INITIAL:
        return 350
    if mode == Mode.LOOKAHEAD:
        return 85


def get_heading(yellow, white):
    if (yellow is not None) and (white is not None):
        heading = (np.degrees(np.arctan(slope(yellow))) + np.degrees(np.arctan(slope(white)))) + DOUBLE_LINE_HEADING_OFFSET
    elif (yellow is not None) and slope(yellow) <= 0:
        heading = np.degrees(np.arctan(slope(yellow))) - 90 + DEGREES_OFFSET
    elif (yellow is not None) and slope(yellow) > 0:
        heading = np.degrees(np.arctan(slope(yellow))) + 90 - DEGREES_OFFSET
    elif (white is not None) and slope(white) <= 0:
        heading = 270 - np.degrees(np.arctan(slope(white))) + DEGREES_OFFSET
    elif (white is not None) and slope(white) > 0:
        heading = 90 - np.degrees(np.arctan(slope(white))) - DEGREES_OFFSET
    else:
        heading = 0  # if you can't see yellow or white, keep going straight cuz it is probably intersection
    return np.radians(np.around(heading))


def is_line_to_left_of_centre(yellow):
    x1, y1, x2, y2 = yellow[0]
    if y1 > y2:
        return x1 < 320
    else:
        return x2 < 320


def get_heading_from_red_line(red_line):
    target_slope = np.around(np.degrees(np.arctan((-1 / (slope(red_line) + 0.0000000001)))))
    if target_slope > 0:
        heading = np.radians(90 - target_slope)
    else:
        heading = np.radians(target_slope + 90)
    return heading


def get_best_possible_building_location(left_img, right_img, forward_img):
    """
    Take three images:
    left: angle = 60
    right: angle = 45
    forward: angle = 0
    Pass them to this function and it'll tell you which direction and position the building is most probably in.
    Output egs: ['left', 1], ['right', 2], ['stright', 3]
    Direction can be left, right, or straight.
    Position can be 1, 2 or 3 - 1 is the left part of the image, 2 is the middle part of the image and 3 is the right part of the image.
    Or you can think of position as the literal part of the image that the cut part comes from. The first cut part, second cut part or the third cut part.
    """
    lh, lw, _ = left_img.shape
    rh, rw, _ = right_img.shape
    sh, sw, _ = forward_img.shape
    h_cut_l1, h_cut_l2, h_cut_l3 = 3*lh//8, 1*lh//3, 2*lh//7
    h_cut_r1, h_cut_r2, h_cut_r3 = 4*rh//11, 2*rh//4, 5*rh//8
    h_cut_s1, h_cut_s2, h_cut_s3 = 2*sh//7, 2*sh//7, 2*sh//7
    img1, img2, img3 = left_img[:h_cut_l1, :lw//3, :], left_img[:h_cut_l2, lw//3:2*lw//3, :], left_img[:h_cut_l3, 2*lw//3:, :]
    img4, img5, img6 = right_img[:h_cut_r1, :rw//3, :], right_img[:h_cut_r2, rw//3:2*rw//3, :], right_img[:h_cut_r3, 2*rw//3:, :]
    img7, img8, img9 = forward_img[:h_cut_s1, :sw//3, :], forward_img[:h_cut_s2, sw//3:2*sw//3, :], forward_img[:h_cut_s3, 2*sw//3:, :]

    max_prob = -1
    direction = None
    position = None
    for i, img in enumerate([img1, img2, img3]):
        building_mask = get_building_mask(img)
        prob = len(np.where(building_mask!=0)[0])/building_mask.size()
        if prob > max_prob:
            direction = 'left'
            position = i
            max_prob = prob
    for i, img in enumerate([img4, img5, img6]):
        building_mask = get_building_mask(img)
        prob = len(np.where(building_mask!=0)[0])/building_mask.size()
        if prob > max_prob:
            direction = 'right'
            position = i
            max_prob = prob
    for i, img in enumerate([img7, img8, img9]):
        building_mask = get_building_mask(img)
        prob = len(np.where(building_mask!=0)[0])/building_mask.size()
        if prob > max_prob:
            direction = 'straight'
            position = i
            max_prob = prob
    return [direction, position]


"""
As soon as you reach the goal tile, stop the bot
Take three pictures for right(45 degrees), left(60 degrees) and forward(0 degrees)
Send them to get_best_possible_building_location()
Take actions as per the output, for example, [left, 1] and [right, 3] mean the hotel is probably right next to you already
[left, 2] and [right, 2] mean the hotel is a few steps ahead
[left, 3] and [right, 3] mean the hotel is near the end of the tile.
[straight, 1], [straight, 2] [straight, 3] mean the hotel is at the end of the tile
You might want to use lookdown and make sure you don't cross a white or yellow line while moving forward
"""