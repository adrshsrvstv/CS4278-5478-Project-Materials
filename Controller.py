import numpy as np
from Perception import *
from Plan import *
from Intentions import *


def get_pid_params_for(state):
    if state == State.INITIALIZING:
        k_p, k_d = 0.05, 2.5
    elif state == State.IN_LANE:
        k_p, k_d = 0.1, 12.5
    elif state == State.IN_LANE_USING_RED:
        k_p, k_d = 0, 12.5
    elif state == State.TURNING:
        k_p, k_d = 0.125, 12.5
    elif state == State.CROSSING_INTERSECTION:
        k_p, k_d = 0.075, 15
    else:
        raise ValueError("Invalid state")
    return k_p, k_d


def get_speed(steering):
    if abs(steering) < 5:
        speed = 1
    else:
        speed = math.exp(-abs(steering) / 25)
    return round(speed, 2)


def get_turn_steps(intent):
    if intent == Intent.LEFT:
        return 9
    if intent == Intent.RIGHT:
        return 7
    else:
        raise ValueError("Invalid Intent")


class Controller:
    def __init__(self, map_name, map_image, start_tile, goal_tile):
        self.state = State.INITIALIZING
        start = tuple(start_tile.split(','))
        goal = tuple(goal_tile.split(','))
        self.plan = get_plan(map_name, map_image, start_tile, goal_tile)
        self.steps_after_crossing_red_line = 0
        self.turn_steps_taken = 0
        self.turn_intent = None

    def check_plan(self, info):
        goal_tile, intent = self.plan.get_current_goal()
        if (goal_tile is not None) and (info['curr_pos'] == goal_tile):
            self.plan.mark_current_goal_done()

    def get_next_action(self, observation, info):

        self.check_plan(info)
        goal_tile, intent = self.plan.get_current_goal()

        yellow, white, red = get_lane_lines(observation, self.state)
        k_p, k_d = get_pid_params_for(self.state)
        d_est = get_d_est(yellow, white, get_mode_from_state(self.state))

        if self.state == State.INITIALIZING:
            heading = get_heading(yellow, white)
            if abs(np.degrees(heading)) < 10:  # and abs(d_est) < 40:
                self.state = State.IN_LANE
                print("State changed to ", self.state)

        elif self.state == State.IN_LANE:
            if red is not None:
                heading = get_heading_from_red_line(red)
                print("Saw red line.")
                if abs(np.degrees(heading)) <= 10:
                    self.state = State.IN_LANE_USING_RED
                    print("State changed to ", self.state)
            if (self.state == State.IN_LANE) and (
                    (yellow is None and white is not None) or (white is None and yellow is not None)):
                self.state = State.INITIALIZING
                print("Trying to get back in lane. State changed to ", self.state)
                k_p, k_d, d_est, heading = self.get_next_action(observation, info)
            # continue straight if you don't see red, or if you see red, but it's too inclined to correct solely based on it.
            # This is the case where either both yellow and white are present or neither are.
            else:
                heading = get_heading(yellow, white)

        elif self.state == State.IN_LANE_USING_RED:
            if red is not None:
                heading = get_heading_from_red_line(red)
            else:
                d_est = 0
                heading = 0
                self.state = State.CROSSING_INTERSECTION if intent == Intent.FORWARD else State.TURNING
                self.turn_intent = intent if self.state == State.TURNING else None
                print("State changed to ", self.state)
                print("Current intent: ", intent, " Current goal: ", goal_tile)

        elif self.state == State.CROSSING_INTERSECTION:
            self.steps_after_crossing_red_line += 1
            if intent == Intent.FORWARD:
                if (yellow is not None) and (white is not None):
                    heading = get_heading(yellow, white)
                    self.state = State.IN_LANE
                    print("State changed to ", self.state)
                elif (yellow is None) and (white is not None):  # left only intersection
                    heading = np.radians(np.degrees(np.arctan(slope(white))) - 27)
                else:  # no lines visible, go straight
                    heading = 0
                    d_est = 0
            else:
                self.state = State.TURNING
                self.turn_intent = intent
                print("State changed to ", self.state)
                print("Current intent: ", intent, " Current goal: ", goal_tile)
                k_p, k_d, d_est, heading = self.get_next_action(observation, info)

        elif self.state == State.TURNING:
            self.steps_after_crossing_red_line += 1
            if (self.turn_steps_taken > get_turn_steps(self.turn_intent)) and ((yellow is not None) and (white is not None)):
                heading = get_heading(yellow, white)
                if intent == Intent.FORWARD: # you would've reached goal tile by now
                    self.state = State.IN_LANE
                    self.steps_after_crossing_red_line = 0
                    self.turn_steps_taken = 0
                    self.turn_intent = None
                    print("State changed to ", self.state)
            elif (intent == Intent.LEFT and self.steps_after_crossing_red_line < 13) or (
                    intent == Intent.RIGHT and self.steps_after_crossing_red_line < 7):
                heading = np.radians(np.degrees(np.arctan(slope(white))) - 30) if (white is not None) else 0
                d_est = 0
                print("steps since crossing red: ", self.steps_after_crossing_red_line)
            else:
                if self.turn_steps_taken <= get_turn_steps(self.turn_intent):
                    heading = np.radians(25) if intent == Intent.LEFT else np.radians(-25)
                    d_est = 0
                    self.turn_steps_taken += 1
                else:
                    heading = 0
                    d_est = 0
        else:
            raise ValueError("Invalid state.")

        return k_p, k_d, d_est, heading
