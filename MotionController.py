import argparse

import numpy as np
import pyglet
from gym_duckietown.envs import DuckietownEnv
from pyglet.window import key
import sys
import cv2

from Perception import *
from Controller import *


# noinspection DuplicatedCode
def str2bool(v):
    """
    Reads boolean value from a string
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# declare the arguments
parser = argparse.ArgumentParser()

# Do not change this
parser.add_argument('--max_steps', type=int, default=1500, help='max_steps')

# You should set them to different map name and seed accordingly
parser.add_argument('--map-name', '-m', default="map4_0", type=str)
parser.add_argument('--seed', '-s', default=2, type=int)
parser.add_argument('--start-tile', '-st', default="1,13", type=str, help="two numbers separated by a comma")
parser.add_argument('--goal-tile', '-gt', default="3,3", type=str, help="two numbers separated by a comma")
parser.add_argument('--control_path', default=None, type=str,
                    help="the control file to run")
parser.add_argument('--manual', default=False, type=str2bool, help="whether to manually control the robot")
args = parser.parse_args()

# simulator instantiation
env = DuckietownEnv(
    domain_rand=False,
    max_steps=1500,
    map_name=args.map_name,
    seed=args.seed,
    user_tile_start=args.start_tile,
    goal_tile=args.goal_tile,
    randomize_maps_on_reset=False
)

# obs = env.reset() # WARNING: never call this function during testing


map_img, goal, start_pos = env.get_task_info()
print("start tile:", start_pos, " goal tile:", goal)

# Show the map image
# White pixels are drivable and black pixels are not.
# Blue pixels indicate lan center
# Each tile has size 100 x 100 pixels
# Tile (0, 0) locates at left top corner.
cv2.imshow("map", map_img)
cv2.imwrite("observations_test/"+args.map_name+".jpg", cv2.cvtColor(map_img, cv2.COLOR_RGB2BGR))
cv2.waitKey(200)

env.render()

curr_pos = start_pos
total_reward = 0
actions = []


heading = 0
d_est = 0
k_p = 0.01
k_d = 0.1
speed = 0
steering = 0

if args.control_path is not None:
    controller = Controller(plan_file=args.control_path)
else:
    controller = Controller(map_image=map_img, start_tile=start_pos, goal_tile=goal)

while curr_pos != goal:
    obs, reward, done, info = env.step([speed, steering])

    total_reward += reward
    print(f"speed = {speed}, steering = {steering}, current pose = {info['curr_pos']}, step count = {env.unwrapped.step_count}, step reward = {reward:.3f}")
    env.render()

    if curr_pos == goal:
        break

    cv2.imwrite("observations_test/" + str(env.unwrapped.step_count) + ".jpg", cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
    speed, steering = controller.get_next_action(obs, info)
    actions.append((speed, steering))

    curr_pos = info['curr_pos']

print('Final Reward = %.3f' % total_reward)
np.savetxt(f'./{args.map_name}_seed{args.seed}_start_{start_pos[0]},{start_pos[1]}_goal_{goal[0]},{goal[1]}.txt', actions, delimiter=',')
cv2.imwrite("observations_test/map.jpg", cv2.cvtColor(map_img, cv2.COLOR_RGB2BGR))

env.close()
