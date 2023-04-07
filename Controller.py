import numpy as np

from Perception import *
from Plan import *


class State(Enum):
    INITIALIZING = 1
    STEADY = 2


class Mode(Enum):
    IN_LANE = 1
    TURNING = 2


LOOKDOWN = "lookdown"
LOOKAHEAD = "lookahead"
INITIAL = "initial"


class Controller:
    def __init__(self, plan_file, heading, d_est, k_p, k_d):
        self.state = State.INITIALIZING
        self.mode = None
        self.intent = None

        self.heading = heading
        self.d_est = d_est
        self.k_p = k_p
        self.k_d = k_d
        self.plan = Plan(plan_file)

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
            yellow, white, red = get_lane_lines(initial, mode=INITIAL)
            heading = get_heading(yellow, white)
            d_est = get_d_est(yellow, white, mode=INITIAL)

            if abs(np.degrees(heading)) < 10:  # and abs(d_est) < 40:
                self.state = State.STEADY
                self.mode = Mode.IN_LANE
                print("State changed to STEADY")

        elif self.state == State.STEADY:

            speed, k_p, k_d = 1, 0.125, 12.5
            yellow, white, red = get_lane_lines(lookdown, mode=LOOKDOWN)

            if self.mode == Mode.IN_LANE:
                if (intent == Intent.LEFT or intent == Intent.RIGHT) and (red is not None):
                    speed, d_est, heading = 0, 0, 0
                else:  # continue straight
                    heading = get_heading(yellow, white)
                    d_est = get_d_est(yellow, white, mode=LOOKDOWN)
                
            elif self.mode == Mode.TURNING:
                pass

        return speed, k_p, k_d, d_est, heading
