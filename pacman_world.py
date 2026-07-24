# pacman_world.py - the PacMan Lite world (PROVIDED - you don't edit this file)
#
# This file is the game. It knows the map, the dots, the ghost, and the
# rules. Your agent talks to it through a few simple functions:
#
#   world = PacmanWorld()
#   state = world.reset()              start a fresh game, get the state
#   state, reward, done = world.step(action)   make one move
#   world.render()                     print the board
#   feats = features(world)            12 numbers describing the situation
#
# The state is a tuple - like a list, but frozen so it can't change.
# The only thing you need to know about tuples today: they work as
# dictionary keys, which is exactly what your Q-table needs.
#
# The map:                # = wall   . = dot   P = you   G = ghost
#
#        #######
#        #P...o#
#        #.###.#
#        #o...o#
#        #######
#
# Rewards: eat a dot +10, eat ALL dots +50 and you win,
# caught by the ghost -50 and you lose, every step costs -1.

import time

ACTIONS = ["up", "down", "left", "right"]

# how each action changes (row, col)
MOVES = {
    "up":    (-1, 0),
    "down":  (1, 0),
    "left":  (0, -1),
    "right": (0, 1),
}

# the map: rows of text, # is a wall
LAYOUT = [
    "#######",
    "#.....#",
    "#.###.#",
    "#.....#",
    "#######",
]

PAC_START = (1, 1)
DOT_SPOTS = [(1, 5), (3, 1), (3, 5)]

# the ghost walks this loop forever: right, right, left, left, ...
GHOST_PATROL = [(3, 2), (3, 3), (3, 4), (3, 3)]


def is_wall(cell):
    """Checks whether a cell of the map is a wall.

    Args:
        cell: A (row, col) tuple.

    Returns:
        True if that spot on the map is a wall, False otherwise.
    """
    row = cell[0]
    col = cell[1]
    return LAYOUT[row][col] == "#"


class PacmanWorld:
    def __init__(self):
        self.reset()

    def reset(self):
        """Starts a fresh game.

        Returns:
            The starting state tuple (use it to begin an episode).
        """
        self.pac = PAC_START
        self.ghost_i = 0                 # where the ghost is in its patrol
        self.eaten = [0, 0, 0]           # one flag per dot: 0 = still there
        self.done = False
        self.won = False
        return self.get_state()

    def ghost(self):
        """Returns the ghost's current (row, col) position."""
        return GHOST_PATROL[self.ghost_i]

    def get_state(self):
        """Bundles the current situation into one dictionary-key-able value.

        Returns:
            A tuple of (pacman position, ghost patrol step, dots eaten).
            Use it as a key in your Q-table.
        """
        return (self.pac, self.ghost_i, tuple(self.eaten))

    def step(self, action):
        """Plays one turn: PacMan moves, then the ghost moves.

        Args:
            action: One of "up", "down", "left", or "right".

        Returns:
            Three values: (new_state, reward, done). done is True when
            the game ended (all dots eaten, or caught by the ghost).
        """
        old_ghost = self.ghost()
        old_pac = self.pac

        # move pacman (walls block the move: you stay put)
        move = MOVES[action]
        new_pac = (self.pac[0] + move[0], self.pac[1] + move[1])
        if not is_wall(new_pac):
            self.pac = new_pac

        # the ghost takes its next patrol step
        self.ghost_i = (self.ghost_i + 1) % len(GHOST_PATROL)
        new_ghost = self.ghost()

        # caught? same cell, or pacman and the ghost walked through each other
        caught = False
        if self.pac == new_ghost:
            caught = True
        if self.pac == old_ghost and new_ghost == old_pac:
            caught = True

        if caught:
            self.done = True
            return self.get_state(), -50, True

        # eat a dot?
        reward = -1                       # every step costs a little
        for i in range(len(DOT_SPOTS)):
            if self.eaten[i] == 0 and self.pac == DOT_SPOTS[i]:
                self.eaten[i] = 1
                reward = reward + 10

        # all dots eaten -> win!
        if self.eaten[0] == 1 and self.eaten[1] == 1 and self.eaten[2] == 1:
            self.done = True
            self.won = True
            return self.get_state(), reward + 50, True

        return self.get_state(), reward, False

    def render(self):
        """Prints the current board to the terminal."""
        for row in range(len(LAYOUT)):
            line = ""
            for col in range(len(LAYOUT[row])):
                cell = (row, col)
                if cell == self.pac:
                    line = line + "P"
                elif cell == self.ghost():
                    line = line + "G"
                elif cell in DOT_SPOTS and self.eaten[DOT_SPOTS.index(cell)] == 0:
                    line = line + "o"
                elif is_wall(cell):
                    line = line + "#"
                else:
                    line = line + " "
            print(line)
        print()


def manhattan(a, b):
    """Distance between two cells walking only along rows and columns.

    Args:
        a: A (row, col) tuple.
        b: Another (row, col) tuple.

    Returns:
        The number of steps between them, ignoring walls.
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def nearest_dot(world):
    """Finds the closest dot PacMan has not eaten yet.

    Args:
        world: The PacmanWorld object.

    Returns:
        The (row, col) of the nearest uneaten dot, or None if all
        dots are eaten.
    """
    best = None
    best_d = 999
    for i in range(len(DOT_SPOTS)):
        if world.eaten[i] == 0:
            d = manhattan(world.pac, DOT_SPOTS[i])
            if d < best_d:
                best_d = d
                best = DOT_SPOTS[i]
    return best


def features(world):
    """Describes the current situation as 12 numbers for a model.

    Three numbers per direction (up, down, left, right), in order:
    blocked by a wall?, toward the nearest dot?, ghost danger?

    Args:
        world: The PacmanWorld object.

    Returns:
        A list of 12 numbers, each 0 or 1.
    """
    feats = []
    dot = nearest_dot(world)
    next_ghost = GHOST_PATROL[(world.ghost_i + 1) % len(GHOST_PATROL)]
    for action in ACTIONS:
        move = MOVES[action]
        land = (world.pac[0] + move[0], world.pac[1] + move[1])

        # 1: would this move hit a wall?
        if is_wall(land):
            feats.append(1)
        else:
            feats.append(0)

        # 2: does this move take us closer to the nearest dot?
        if dot is not None and not is_wall(land) and manhattan(land, dot) < manhattan(world.pac, dot):
            feats.append(1)
        else:
            feats.append(0)

        # 3: would this move land dangerously close to the ghost?
        if manhattan(land, next_ghost) <= 1:
            feats.append(1)
        else:
            feats.append(0)
    return feats


def play(world, choose_action, delay=0.3, silent=False):
    """Watches an agent play one full game in the terminal.

    Args:
        world: The PacmanWorld object.
        choose_action: A function that takes the world and returns one
            of the four action strings. This is your agent's brain.
        delay: Seconds to pause between turns (0 for instant).
        silent: If True, skip the animation and just play.

    Returns:
        Two values: (won, total_reward) for the finished game.
    """
    world.reset()
    total = 0
    for turn in range(60):                # a game never runs forever
        if not silent:
            print("turn", turn, "  score so far:", total)
            world.render()
            time.sleep(delay)
        action = choose_action(world)
        state, reward, done = world.step(action)
        total = total + reward
        if done:
            break
    if not silent:
        world.render()
        if world.won:
            print("PacMan ate every dot! Final score:", total)
        else:
            print("Game over. Final score:", total)
    return world.won, total
