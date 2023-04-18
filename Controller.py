from Perception import *
from Plan import *

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
    if intent == Intent.RIGHT:
        return 10
    else:
        raise ValueError("Invalid Intent", intent)


def get_straight_steps_for_turn_at_intersection(intent):
    if intent == Intent.LEFT:
        return 14
    if intent == Intent.RIGHT:
        return 7
    else:
        raise ValueError("Invalid Intent", intent)


class Controller:
    def __init__(self, plan_file):
        self.state = State.INITIALIZING
        self.plan = Plan(plan_file)
        self.straight_steps_before_turn = 0
        self.turn_steps_taken = 0
        self.turn_intent = None

    def change_state_to(self, state):
        if self.state != state:
            print("State changed to ", state)
        self.state = state

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
            if abs(np.degrees(heading)) < 10:  # and abs(d_est) < 25:
                self.change_state_to(State.IN_LANE_AND_FORWARD)

        elif self.state == State.IN_LANE_AND_FORWARD:
            if red is not None:
                heading = get_heading_from_red_line(red)
                self.change_state_to(State.IN_LANE_USING_RED)
            elif (intent != Intent.FORWARD) or (next_intent != Intent.FORWARD):
                heading = get_heading(yellow, white)
                self.turn_intent = intent if intent != Intent.FORWARD else next_intent
                self.change_state_to(State.IN_LANE_AND_WAITING_TO_TURN)
            elif (yellow is None and white is not None) or (white is None and yellow is not None):
                self.change_state_to(State.INITIALIZING)
                return self.get_next_action(observation, info)
            else:
                heading = get_heading(yellow, white)

        elif self.state == State.IN_LANE_AND_WAITING_TO_TURN:
            closer_line, farther_line = (yellow, white) if self.turn_intent == Intent.LEFT else (white, yellow)
            if red is not None:
                heading = get_heading_from_red_line(red)
                self.change_state_to(State.IN_LANE_USING_RED)
            elif (white is not None) and (yellow is not None) and (abs(angle_in_degrees(white)) > 15) and (abs(angle_in_degrees(yellow)) > 15):
                heading = get_heading(yellow, white)
            elif (farther_line is not None) and (abs(angle_in_degrees(farther_line)) < 15):
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
            closer_line, farther_line = (yellow, white) if self.turn_intent == Intent.LEFT else (white, yellow)
            if self.straight_steps_before_turn < 5:
                d_est = 0
                heading = 0
                self.straight_steps_before_turn += 1
            elif self.turn_steps_taken < 20:
                heading = np.radians(TURN_ANGLE_PER_STEP_IN_LANE) if self.turn_intent == Intent.LEFT else np.radians(-TURN_ANGLE_PER_STEP_IN_LANE)
                d_est = 0
                self.turn_steps_taken += 1
            elif (yellow is not None) and (white is not None):
                heading = get_heading(yellow, white)
                if intent == Intent.FORWARD:
                    self.change_state_to(State.IN_LANE_AND_FORWARD)
                    self.turn_steps_taken = 0
                    self.straight_steps_before_turn = 0
                    self.turn_intent = None
            else:
                heading = 0
                d_est = 0


        elif self.state == State.CROSSING_INTERSECTION:
            self.straight_steps_before_turn += 1
            if intent == Intent.FORWARD:
                if (yellow is not None) and (white is not None):
                    heading = get_heading(yellow, white)
                    self.change_state_to(State.IN_LANE_AND_FORWARD)
                elif (yellow is None) and (white is not None):  # left only intersection
                    heading = np.radians(angle_in_degrees(white) - 28)
                else:  # no lines visible, go straight
                    heading = 0
                    d_est = 0
            else:
                self.change_state_to(State.TURNING_AT_INTERSECTION)
                self.turn_intent = intent
                print("Current intent: ", intent, " Current goal: ", goal_tile)
                return self.get_next_action(observation, info)

        elif self.state == State.TURNING_AT_INTERSECTION:
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
