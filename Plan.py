import csv
from enum import Enum


class Intent(str, Enum):
    FORWARD = 'forward'
    LEFT = 'left'
    RIGHT = 'right'


class Plan:
    def __init__(self, filepath=None, list_plan=None):
        if (filepath is not None) and (list_plan is None):
            self.plan = self.__read_plan_from_file__(filepath)
        elif (list_plan is not None) and (filepath is None):
            self.plan = self.fix_list_plan(list_plan)
        else:
            raise ValueError("Exactly one of `filepath` or `list_plan` needs to be None.")

    def __read_plan_from_file__(self, filepath):
        plan = []
        with open(filepath) as csvfile:
            for row in csv.reader(csvfile):
                goal_tile = int(row[0][1:]), int(row[1][1:-1])
                intent = Intent(row[2][1:])
                plan.append((goal_tile, intent))
        return plan

    def fix_list_plan(self, list_plan):
        new_list_plan = [(list_plan[0][0], Intent.FORWARD)]
        for i in range(1, len(list_plan)):
            current_goal = list_plan[i][0]
            previous_intent = Intent(list_plan[i-1][1])
            new_intent = Intent.FORWARD if previous_intent == Intent.FORWARD else (Intent.LEFT if previous_intent == Intent.RIGHT else Intent.RIGHT)
            new_list_plan.append((current_goal, previous_intent))
        return new_list_plan


    def get_current_goal(self):
        if len(self.plan) != 0:
            return self.plan[0]
        else:
            return None, None

    def get_next_goal(self):
        if len(self.plan) >= 2:
            return self.plan[1]
        else:
            return None, None

    def mark_current_goal_done(self):
        if len(self.plan) != 0:
            self.plan.pop(0)
