import csv
from enum import Enum


class Intent(str, Enum):
    FORWARD = 'forward'
    LEFT = 'left'
    RIGHT = 'right'


class Plan:
    def __init__(self, filepath):
        self.plan = self.__read_plan__(filepath)

    def __read_plan__(self, filepath):
        plan = []
        with open(filepath) as csvfile:
            for row in csv.reader(csvfile):
                goal_tile = int(row[0][1]), int(row[1][1])
                intent = Intent(row[2][1:])
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
