from Perception import *
from Plan import *
from Intentions import get_plan
import pprint

pp = pprint.PrettyPrinter(indent=4)

TURN_ANGLE_PER_STEP_FOR_INTERSECTION = 24
TURN_ANGLE_PER_STEP_IN_LANE = 25


def get_pid_params_for(state):
    if state == State.INITIALIZING:
        k_p, k_d = 0.075, 2.5
    elif state == State.IN_LANE_AND_FORWARD:
        k_p, k_d = 0.15, 12.5
    elif state == State.IN_LANE_AND_WAITING_TO_TURN:
        k_p, k_d = 0.15, 12.5
    elif state == State.TURNING_WITHIN_LANE:
        k_p, k_d = 0.125, 12.5
    elif state == State.IN_LANE_USING_RED:
        k_p, k_d = 0.15, 12.5
    elif state == State.TURNING_AT_INTERSECTION:
        k_p, k_d = 0.125, 12.5
    elif state == State.CROSSING_INTERSECTION:
        k_p, k_d = 0.075, 15
    else:
        raise ValueError("Invalid state")
    return k_p, k_d


def get_speed(steering, state):
    if state is State.TURNING_WITHIN_LANE:
        speed = 1
    else:
        if abs(steering) < 3:
            speed = 1
        else:
            speed = math.exp(-abs(steering) / 25)
    return round(speed, 2)


def get_turn_steps_for_intersection(intent):
    if intent == Intent.LEFT:
        return 10
    elif intent == Intent.RIGHT:
        return 10
    else:
        raise ValueError("Invalid Intent", intent)


def get_straight_steps_for_turn_at_intersection(intent):
    if intent == Intent.LEFT:
        return 14
    elif intent == Intent.RIGHT:
        return 5
    else:
        raise ValueError("Invalid Intent", intent)


def get_straight_steps_for_turn_in_lane(intent):
    if intent == Intent.LEFT:
        return 3
    elif intent == Intent.RIGHT:
        return 4
    else:
        raise ValueError("Invalid Intent", intent)


def get_turn_steps_for_in_lane(intent):
    if intent == Intent.LEFT:
        return 20
    elif intent == Intent.RIGHT:
        return 19
    else:
        raise ValueError("Invalid Intent", intent)


def get_turn_angle_for_in_lane_turning(intent):
    if intent == Intent.LEFT:
        return np.radians(26)
    elif intent == Intent.RIGHT:
        return np.radians(-30)
    else:
        raise ValueError("Invalid Intent", intent)


class Controller:
    def __init__(self, plan_file=None, map_image=None, start_tile=None, goal_tile=None):
        self.state = State.INITIALIZING
        self.straight_steps_before_turn = 0
        self.turn_steps_taken = 0
        self.turn_intent = None
        self.turn_goal = None
        self.done_turn_goals = []

        if (plan_file is not None) and (map_image is None):
            self.plan = Plan(filepath=plan_file)
        elif (map_image is not None) and (start_tile is not None) and (goal_tile is not None) and (plan_file is None):
            self.plan = Plan(list_plan=get_plan(map_image, start_tile, goal_tile))
        else:
            raise ValueError("Exactly one of plan_file or map_image (along with start and goal tiles) must be supplied.")
        pp.pprint(self.plan.plan)

    def change_state_to(self, state):
        if self.state != state:
            print("State changed to ", state)
        self.state = state

    def mark_turn_goal_done(self, goal):
        self.done_turn_goals.append(goal)

    def is_turn_goal_done(self, goal):
        return goal in self.done_turn_goals

    def check_plan(self, info):
        goal_tile, intent = self.plan.get_current_goal()
        if (goal_tile is not None) and (info['curr_pos'] == goal_tile):
            self.plan.mark_current_goal_done()

    def get_next_action(self, observation, info):

        self.check_plan(info)
        goal_tile, intent = self.plan.get_current_goal()
        next_goal, next_intent = self.plan.get_next_goal()

        yellow, white, red = get_lane_lines(observation, self.state)
        k_p, k_d = get_pid_params_for(self.state)
        d_est = get_d_est(yellow, white, get_mode_from_state(self.state))

        if self.state == State.INITIALIZING:
            heading = get_heading(yellow, white)
            # if red is not None:
            #     heading = get_heading_from_red_line(red)
            #     if abs(np.degrees(heading)) < 20:
            #         self.change_state_to(State.IN_LANE_USING_RED)
            # if (red is not None) and (white is not None) and abs(angle_in_degrees(red)) < 5:
            #     self.change_state_to(State.IN_LANE_USING_RED)
            if abs(np.degrees(heading)) < 10:  # and abs(d_est) < 25:
                self.change_state_to(State.IN_LANE_AND_FORWARD)

        elif self.state == State.IN_LANE_AND_FORWARD:
            if red is not None:
                heading = get_heading_from_red_line(red)
                self.change_state_to(State.IN_LANE_USING_RED)
            elif ((intent is not None) and (intent != Intent.FORWARD) and (not self.is_turn_goal_done(goal_tile))) or (next_intent is not None and next_intent != Intent.FORWARD and (not self.is_turn_goal_done(next_goal))):
                heading = get_heading(yellow, white)
                if (intent is not None) and (intent != Intent.FORWARD) and (not self.is_turn_goal_done(goal_tile)):
                    self.turn_intent = intent
                    self.turn_goal = goal_tile
                if (next_intent is not None) and (next_intent != Intent.FORWARD) and (not self.is_turn_goal_done(next_goal)):
                    self.turn_intent = next_intent
                    self.turn_goal = next_goal
                print("\tSetting turn intent to ", self.turn_intent)
                self.change_state_to(State.IN_LANE_AND_WAITING_TO_TURN)
            elif white is not None and angle_in_degrees(white) > 20:
                heading = np.radians(angle_in_degrees(white) - 28)
            elif (yellow is None and white is not None) or (white is None and yellow is not None):
                self.change_state_to(State.INITIALIZING)
                return self.get_next_action(observation, info)
            else:
                heading = 0

        elif self.state == State.IN_LANE_AND_WAITING_TO_TURN:
            closer_line, farther_line = (yellow, white) if self.turn_intent == Intent.LEFT else (white, yellow)
            if red is not None:
                heading = get_heading_from_red_line(red)
                self.change_state_to(State.IN_LANE_USING_RED)
            elif (white is not None) and (yellow is not None) and (abs(angle_in_degrees(white)) > 15) and (abs(angle_in_degrees(yellow)) > 15):
                heading = get_heading(yellow, white)
            elif (farther_line is not None) and (abs(angle_in_degrees(farther_line)) < 10) and (get_distance_from_line(farther_line, get_mode_from_state(self.state)) < 140):
                self.change_state_to(State.TURNING_WITHIN_LANE)
                heading = 0
                d_est = 0
                self.straight_steps_before_turn = 1
            else:
                heading = 0  # get_heading(yellow, white)
                d_est = 0

        elif self.state == State.IN_LANE_USING_RED:
            if red is not None:
                heading = get_heading_from_red_line(red)
            else:
                d_est = 0
                heading = 0
                if intent == Intent.FORWARD:
                    self.change_state_to(State.CROSSING_INTERSECTION)
                else:
                    self.change_state_to(State.TURNING_AT_INTERSECTION)
                    self.turn_intent = intent

        elif self.state == State.TURNING_WITHIN_LANE:
            #print("Current intent: ", intent, "turn_intent: ", self.turn_intent, " Current goal: ", goal_tile)
            closer_line, farther_line = (yellow, white) if self.turn_intent == Intent.LEFT else (white, yellow)
            if self.straight_steps_before_turn < get_straight_steps_for_turn_in_lane(self.turn_intent):
                d_est = 0
                heading = 0
                self.straight_steps_before_turn += 1
            elif self.turn_steps_taken < get_turn_steps_for_in_lane(self.turn_intent):
                heading = get_turn_angle_for_in_lane_turning(self.turn_intent)
                d_est = 0
                self.turn_steps_taken += 1
            elif intent != self.turn_intent:
                #print("turn_intent: ", self.turn_intent, " curr_intent: ", intent, " next_intent: ", next_intent)
                heading = 0
                self.change_state_to(State.IN_LANE_AND_FORWARD if intent == Intent.FORWARD else State.IN_LANE_AND_WAITING_TO_TURN)
                self.turn_intent = None if intent == Intent.FORWARD else intent
                print("\tSetting turn intent to ", self.turn_intent)
                self.turn_steps_taken = 0
                self.straight_steps_before_turn = 0
                self.mark_turn_goal_done(self.turn_goal)
                self.turn_goal = None
            elif ((yellow is not None) and (white is None)) or ((yellow is None) and (white is not None)):
                self.turn_steps_taken = 0
                self.straight_steps_before_turn = 0
                self.turn_intent = None
                print("\tSetting turn intent to ", self.turn_intent)
                self.change_state_to(State.INITIALIZING)
                self.mark_turn_goal_done(self.turn_goal)
                self.turn_goal = None
                return self.get_next_action(observation, info)
            else:
                heading = 0
                d_est = 0

        elif self.state == State.CROSSING_INTERSECTION:
            # print(self.straight_steps_before_turn, self.turn_steps_taken)
            self.straight_steps_before_turn += 1
            if intent == Intent.FORWARD:
                if (yellow is not None) and (white is not None):
                    heading = get_heading(yellow, white)
                    self.change_state_to(State.IN_LANE_AND_FORWARD)
                    self.straight_steps_before_turn = 0
                elif (yellow is None) and (white is not None):  # left only intersection
                    heading = np.radians(angle_in_degrees(white) - 28)
                elif (red is not None) and abs(angle_in_degrees(red)) > 20 and (red[0][0] > 320 and red[0][2] > 320):
                    heading = np.radians(angle_in_degrees(red) - 28)
                    d_est = 188 - get_distance_from_line(red, get_mode_from_state(self.state))
                else:  # no lines visible, go straight
                    heading = 0
                    d_est = 0
            else:
                self.change_state_to(State.TURNING_AT_INTERSECTION)
                self.turn_intent = intent
                print("Current intent: ", intent, " Current goal: ", goal_tile)
                return self.get_next_action(observation, info)

        elif self.state == State.TURNING_AT_INTERSECTION:
            # print(self.straight_steps_before_turn, self.turn_steps_taken)
            if self.straight_steps_before_turn < get_straight_steps_for_turn_at_intersection(self.turn_intent):
                heading = np.radians(angle_in_degrees(white) - 28) if (white is not None) else 0
                d_est = 0
                self.straight_steps_before_turn += 1
            elif self.turn_steps_taken < get_turn_steps_for_intersection(self.turn_intent):
                heading = np.radians(TURN_ANGLE_PER_STEP_FOR_INTERSECTION) if self.turn_intent == Intent.LEFT else np.radians(-TURN_ANGLE_PER_STEP_FOR_INTERSECTION)
                d_est = 0
                self.turn_steps_taken += 1
            elif (yellow is not None) and (white is not None):
                heading = get_heading(yellow, white)
                if intent == Intent.FORWARD:  # you would've reached goal tile by now
                    self.change_state_to(State.IN_LANE_AND_FORWARD)
                    self.straight_steps_before_turn = 0
                    self.turn_steps_taken = 0
                    self.turn_intent = None
            else:
                heading = 0
                d_est = 0
        else:
            raise ValueError("Invalid state.")

        steering = k_p * d_est + k_d * heading
        speed = get_speed(steering, self.state)
        return speed, steering
