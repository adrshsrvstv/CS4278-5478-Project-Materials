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

    def get_next_action(self, observation, info):
        if self.state == State.INITIALIZING:
            d_est, heading, k_p, k_d = self.get_in_lane(observation)
            return 0.25, d_est, heading, k_p, k_d
        if self.state == State.STEADY:
            curr_tile = info['curr_pos']
            goal_tile, intent = self.plan.get_current_goal()
            if goal_tile is None:
                return 0, 0, 0, self.k_p, self.k_d
            elif curr_tile == goal_tile:
                self.plan.mark_current_goal_done()
                return self.get_next_action(observation, info)
            else:
                lookdown, lookahead, _ = preprocess(observation)
                yellow, white, red = get_lane_lines(lookdown, mode=LOOKDOWN)
                if self.mode == Mode.IN_LANE:
                    if (intent == Intent.LEFT or intent == Intent.RIGHT) and (red is not None):
                        # do interesting stuff
                        print(intent)
                        pass
                    else: # continue straight
                        heading = get_heading(yellow, white)
                        d_est = get_d_est(yellow, white, mode=LOOKDOWN)
                        print("d_est=", d_est, " , heading in rads=", heading)
                        speed = 1
            return speed, d_est, heading, 0.125, 12.5

    def get_in_lane(self, observation):
        lookdown, _, initial = preprocess(observation)
        yellow, white, red = get_lane_lines(initial, mode=INITIAL)
        heading = get_heading(yellow, white)
        d_est = get_d_est(yellow, white, mode=INITIAL)

        if abs(np.degrees(heading)) < 10:# and abs(d_est) < 40:
            self.state = State.STEADY
            self.mode = Mode.IN_LANE
            print("State changed to STEADYYYYY!!!!!!")

        return d_est, heading, 0.05, 2.5
