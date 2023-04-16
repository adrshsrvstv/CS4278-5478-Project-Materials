from multiprocessing.sharedctypes import Value
import numpy as np
import cv2
import yaml
import json
import argparse

class Tile:

    def __init__(self, type=None, parent=None, loc=None, prev_move=None):
        self.type = type
        self.parent = parent
        self.loc = loc
        self.prev_move = prev_move
        self.next_move = None
        self.f, self.g, self.h = 0, 0, 0

def process_all_maps():
    with open('testcases/milestone2.json', 'r') as json_f:
        all_maps = json.load(json_f)
    for map_name in all_maps:
        output_file = get_plan(map_name, all_maps[map_name]['start'], all_maps[map_name]['goal'])
        print(f'Map {map_name} has its output stored in {output_file}.')

def process_one_map(map_img, map_name='map1_0'):
    with open('testcases/milestone2.json', 'r') as json_f:
        all_maps = json.load(json_f)
    if map_name not in all_maps:
        print('Map does not exist.')
        raise(ValueError('Map does not exist.'))
    output_file = get_plan(map_name, map_img, all_maps[map_name]['start'], all_maps[map_name]['goal'])
    print(f'Map {map_name} has its output stored in {output_file}.')

def get_allowed_moves(tile, prev_move):
    moves = {'right': [1, 0], 'up': [0, -1], 'left': [-1, 0], 'down': [0, 1]}
    move_names = ['right', 'up', 'left', 'down']
    cardinals = ['E', 'N', 'W', 'S']
    ok_moves = []
    if tile == '4way':
        return moves
    tile_type, dir = tile.split('/')
    if dir == 'N':
        ok_moves = [3]
        if tile_type == 'straight':
            ok_moves += [1]
        elif tile_type == 'curve_left':
            ok_moves += [2]
        elif tile_type == 'curve_right':
            ok_moves += [0]
        elif tile_type == '3way_left':
            ok_moves += [1, 2]
        elif tile_type == '3way_right':
            ok_moves += [1, 0]
    elif dir == 'E':
        ok_moves = [2]
        if tile_type == 'straight':
            ok_moves += [0]
        elif tile_type == 'curve_left':
            ok_moves += [1]
        elif tile_type == 'curve_right':
            ok_moves += [3]
        elif tile_type == '3way_left':
            ok_moves += [0, 1]
        elif tile_type == '3way_right':
            ok_moves += [0, 3]
    elif dir == 'S':
        ok_moves = [1]
        if tile_type == 'straight':
            ok_moves += [3]
        elif tile_type == 'curve_left':
            ok_moves += [0]
        elif tile_type == 'curve_right':
            ok_moves += [2]
        elif tile_type == '3way_left':
            ok_moves += [3, 0]
        elif tile_type == '3way_right':
            ok_moves += [3, 2]
    elif dir == 'W':
        ok_moves = [0]
        if tile_type == 'straight':
            ok_moves += [2]
        elif tile_type == 'curve_left':
            ok_moves += [3]
        elif tile_type == 'curve_right':
            ok_moves += [1]
        elif tile_type == '3way_left':
            ok_moves += [2, 3]
        elif tile_type == '3way_right':
            ok_moves += [2, 1]
    allowed_moves = {}
    rotate_factor = 0
    for move in ok_moves:
        allowed_moves[move_names[move]] = moves[move_names[move]]
    return allowed_moves

def get_all_entries(tile):
    cardinals = ['E', 'N', 'W', 'S']
    if tile == '4way':
        return cardinals
    tile_type, dir = tile.split('/')
    if tile_type == 'straight':
        return [dir, cardinals[(cardinals.index(dir)+2)%4]]
    elif tile_type == 'curve_left':
        return [cardinals[(cardinals.index(dir)+rotation)%4] for rotation in [2, 1]]
    elif tile_type == 'curve_right':
        return [cardinals[(cardinals.index(dir)+rotation)%4] for rotation in [-2, -1]]
    elif tile_type == '3way_left':
        return [cardinals[(cardinals.index(dir)+rotation)%4] for rotation in [2, 1, 0]]
    elif tile_type == '3way_right':
        return [cardinals[(cardinals.index(dir)+rotation)%4] for rotation in [-2, -1, 0]]

def convert_img_to_map(map_img):
    matrix = []
    for i in range(50, map_img.shape[0], 100):
        row = []
        for j in range(50, map_img.shape[1], 100):
            if map_img[i, j, 0] == 0:
                row.append(0)
            else:
                row.append(1)
        matrix.append(row)
    matrix = np.array(matrix)

    tiles = np.zeros_like(matrix)
    tiles = [arr.tolist() for arr in tiles]
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] == 0:
                tiles[i][j] = 'empty'
            else:
                if i-1<0:
                    up = 0
                else:
                    up = matrix[i-1, j]
                if i+1>=matrix.shape[0]:
                    down = 0
                else:
                    down = matrix[i+1, j]
                if j-1<0:
                    left = 0
                else:
                    left = matrix[i, j-1]
                if j+1>=matrix.shape[0]:
                    right = 0
                else:
                    right = matrix[i, j+1]
                if up==1 and down==1 and right==1 and left==1:
                    tiles[i][j] = '4way'
                elif up==1 and down==1 and right==1:
                    tiles[i][j] = '3way_right/N'
                elif up==1 and down==1 and left==1:
                    tiles[i][j] = '3way_left/N'
                elif up==1 and left==1 and right==1:
                    tiles[i][j] = '3way_left/E'
                elif down==1 and left==1 and right==1:
                    tiles[i][j] = '3way_right/E'
                elif up==1 and down==1:
                    tiles[i][j] = 'straight/N'
                elif left==1 and right==1:
                    tiles[i][j] = 'straight/E'
                elif up==1 and right==1:
                    tiles[i][j] = 'curve_right/W'
                elif right==1 and down==1:
                    tiles[i][j] = 'curve_left/W'
                elif down==1 and left==1:
                    tiles[i][j] = 'curve_right/E'
                elif left==1 and up==1:
                    tiles[i][j] = 'curve_left/E'
    return tiles

def get_plan(map_name, map_image, start, goal):
    # with open(f'gym-duckietown/gym_duckietown/map_2021/{map_name}.yaml', 'r') as yaml_f:
    #     map_info = yaml.safe_load(yaml_f)
    # tiles = np.array(map_info['tiles'])
    # map_image = cv2.imread(map_image)
    tiles = np.array(convert_img_to_map(map_image))
    tiles = tiles.T
    
    startx, starty = start
    goalx, goaly = goal
    start = (startx, starty)
    goal = (goalx, goaly)

    for x in range(tiles.shape[0]):
        for y in range(tiles.shape[1]):
            if tiles[x, y] in ['asphalt', 'grass', 'floor', 'empty']:
                tiles[x, y] = 'empty'
    
    cost = 1
    enter_from = {'up': 'S', 'down': 'N', 'right': 'W', 'left': 'E'}
    start_tile = Tile(tiles[start], None, start, None)
    goal_tile = Tile(tiles[goal], None, goal, None)
    new_list, old_list = [], []
    new_list.append(start_tile)

    while len(new_list) > 0:
        curr_tile = new_list[0]
        curr_index = 0
        for i, tile in enumerate(new_list):
            if tile.f < curr_tile.f:
                curr_tile = tile
                curr_index = i
        
        new_list.pop(curr_index)
        old_list.append(curr_tile)
        if curr_tile.loc == goal_tile.loc:
            break

        next_steps = []
        allowed_moves = get_allowed_moves(curr_tile.type, curr_tile.prev_move)
        for move in allowed_moves:
            entered_from = enter_from[move]
            direction = allowed_moves[move]
            new_loc = (curr_tile.loc[0]+direction[0], curr_tile.loc[1]+direction[1])
            if new_loc[0]<0 or new_loc[1]<0 or new_loc[0]>=tiles.shape[0] or new_loc[1]>=tiles.shape[1]:
                continue
            if tiles[new_loc] == 'empty':
                continue
            if entered_from not in get_all_entries(tiles[new_loc]):
                continue

            new_tile = Tile(tiles[new_loc], curr_tile, new_loc, move)
            next_steps.append(new_tile)
        
        for step in next_steps:
            flag = 0
            for tile in old_list:
                if tile.loc == step.loc:
                    flag = 1
                    break
            if flag:
                continue

            step.g = curr_tile.g + cost
            step.h = ((step.loc[0] - goal_tile.loc[0]) ** 2) + ((step.loc[1] - goal_tile.loc[1]) ** 2)
            step.f = step.g + step.h

            flag = 0
            for tile in new_list:
                if tile.loc == step.loc and tile.g < step.g:
                    flag = 1
                    break
            
            if flag:
                continue
            
            new_list.append(step)
    path = generate_intentions(curr_tile)
    if curr_tile.loc == goal_tile.loc:
        print(map_name, 'Goal Found')
    else:
        print(map_name, 'Goal Not Found')
    return path

def generate_intentions(final_tile):
    moves = ['right', 'up', 'left', 'down']
    cardinals = ['E', 'N', 'W', 'S']
    enter_from = {'up': 'S', 'down': 'N', 'right': 'W', 'left': 'E'}
    tiles = []
    curr_tile = final_tile
    while curr_tile is not None:
        tiles.append(curr_tile)
        if curr_tile.parent is not None:
            curr_tile.parent.next_move = curr_tile.prev_move 
        curr_tile = curr_tile.parent
    tiles = tiles[::-1]

    intentions = [[tiles[0].loc, 'forward']]
    tiles.pop(0)
    for tile in tiles:
        if tile.type == '4way':
            tile_type = '4way'
        else:
            tile_type, _ = tile.type.split('/')
        if tile_type in ['straight', 'curve_left', 'curve_right'] or tile.next_move is None:
            intention = 'forward'
        else:
            entered_from = enter_from[tile.prev_move]
            if entered_from == 'S':
                intention = tile.next_move
            elif entered_from == 'N':
                intention = moves[(moves.index(tile.next_move)+2)%4]
            elif entered_from == 'E':
                intention = moves[(moves.index(tile.next_move)-1)%4]
            elif entered_from == 'W':
                intention = moves[(moves.index(tile.next_move)+1)%4]
        if intention == 'up':
            intention = 'forward'
        # intentions.append((tile.loc, intention, tile.type, tile.prev_move))
        intentions.append((tile.loc, intention))
    return intentions        




if __name__ == "__main__":
    # process_all_maps()
    map_img = cv2.imread('map3.png')
    process_one_map(map_img, 'map3_0')
    # get_allowed_moves('curve_left\E')