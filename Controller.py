import numpy as np
from Perception import *
from Plan import *
from enum import Enum


class State(Enum):
    INITIALIZING = 1
    IN_LANE = 2
    TURNING = 3


class Mode(str, Enum):
    LOOKDOWN = "lookdown"
    LOOKAHEAD = "lookahead"
    INITIAL = "initial"


class Controller:
    def __init__(self, plan_file):
        self.state = State.INITIALIZING
        self.plan = Plan(plan_file)

    def get_speed(self, steering):
        if abs(steering) < 5:
            speed = 1
        else:
            speed = math.exp(-abs(steering) / 25)
        return round(speed, 2)


    def check_plan(self, info):
        goal_tile, intent = self.plan.get_current_goal()
        if (goal_tile is not None) and (info['curr_pos'] == goal_tile):
            self.plan.mark_current_goal_done()

    def get_next_action(self, observation, info):

        self.check_plan(info)
        goal_tile, intent = self.plan.get_current_goal()

        lookdown, lookahead, initial = preprocess(observation)

        if self.state == State.INITIALIZING:
            speed, k_p, k_d = 0.25, 0.05, 2.5
            yellow, white, red = get_lane_lines(initial, Mode.INITIAL)
            heading = get_heading(yellow, white)
            d_est = get_d_est(yellow, white, Mode.INITIAL)

            if abs(np.degrees(heading)) < 10:  # and abs(d_est) < 40:
                self.state = State.IN_LANE
                print("State changed to IN_LANE")

        elif self.state == State.IN_LANE:
            speed, k_p, k_d = 1, 0.125, 12.5
            yellow, white, red = get_lane_lines(lookdown, Mode.LOOKDOWN)

            if (intent == Intent.LEFT or intent == Intent.RIGHT) and (red is not None):
                speed, d_est, heading = 0, 0, 0
            else:  # continue straight
                heading = get_heading(yellow, white)
                d_est = get_d_est(yellow, white, Mode.LOOKDOWN)

        elif self.state == State.TURNING:
            pass

        else:
            raise ValueError("Invalid state.")

        return speed, k_p, k_d, d_est, heading
