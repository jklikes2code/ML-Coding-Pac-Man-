import random
from pacman_world import PacmanWorld, ACTIONS, play

world = PacmanWorld()

def random_move(world):
    return random.choice(ACTIONS)

play(world, random_move)