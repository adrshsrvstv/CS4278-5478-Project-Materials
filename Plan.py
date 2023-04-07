import csv
from enum import Enum


class Intent(Enum):
    FORWARD = 1
    LEFT = 2
    RIGHT = 3


class Plan:
    def __init__(self, filepath):
        self.plan = self.__read_plan__(filepath)

    def __read_plan__(self, filepath):
        plan = []
        with open(filepath) as csvfile:
            for row in csv.reader(csvfile):
                goal_tile = int(row[0][1]), int(row[1][1])
                direction = row[2][1:]
                if direction == 'forward':
                    intent = Intent.FORWARD
                elif direction == 'left':
                    intent = Intent.LEFT
                elif direction == 'right':
                    intent = Intent.RIGHT
                else:
                    raise ValueError('Invalid Intent in Control Path')
                plan.append((goal_tile, intent))
        return plan

    def get_current_goal(self):
        if len(self.plan) != 0:
            return self.plan[0]
        else:
            return None, None

    def mark_current_goal_done(self):
        if len(self.plan) != 0:
            self.plan.pop(0)
