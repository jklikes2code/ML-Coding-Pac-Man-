# Capstone agent #2: the TRIAL-AND-REWARD learner (Q-learning PacMan)
#
# Put this file in the same folder as pacman_world.py.
# This is your GridWorld training loop pointed at a bigger world.
# Only three things are new:
#   1. the state is a tuple the world hands you (it works as a
#      dictionary key, which is all your Q-table needs)
#   2. the Q-table starts EMPTY and grows as the agent discovers
#      new situations - that's what make_sure_state_exists is for
#   3. world.step(action) hands back all three at once -
#      new_state, reward, done - instead of GridWorld's separate
#      step() and result() calls

import random
from pacman_world import PacmanWorld, ACTIONS, play

world = PacmanWorld()

Q = {}

def make_sure_state_exists(state):
    """Adds a state to the Q-table the first time it is ever seen.

    Args:
        state: The tuple the world hands you (it works as a
            dictionary key). After this call, Q[state] must exist
            with all four actions at 0.0.
    """
    # TODO: if state is not in Q yet, give it an inner dictionary
    # with all four ACTIONS starting at 0.0.
    if state not in Q:
        Q[state] = {}
        for action in ACTIONS:
            Q[state][action] = 0.0

def best_value(state):
    """Returns the highest Q-value available in this state.

    Args:
        state: A state tuple from the world.

    Returns:
        The largest of Q[state]'s four action values, as a float.
    """
    make_sure_state_exists(state)
    # TODO: paste your best_value from GridWorld (unchanged!)
    return max(Q[state][action] for action in ACTIONS) # a one-line shortcut for the whole scan

def best_action(state):
    """Returns the action with the highest Q-value in this state.

    Args:
        state: A state tuple from the world.

    Returns:
        One of the ACTIONS strings: "up", "down", "left", or "right".
    """
    make_sure_state_exists(state)
    # TODO: paste your best_action from GridWorld (unchanged!)
    best_a = "up"
    best = Q[state]["up"]
    for action in ACTIONS:
        if Q[state][action] > best:
            best = Q[state][action]
            best_a = action
    return best_a


# --- training ------------------------------------------------------------
learning_rate = 0.5
discount = 0.9
epsilon = 0.2
steps = 0

for episode in range(20000):
    state = world.reset()
    make_sure_state_exists(state)  # the start state may be brand new - it must
                                   # exist before anything looks it up (the very
                                   # first move can be an exploration move!)
    done = False
    for move in range(100):        # safety cap: no episode runs forever
        # TODO: your GridWorld training loop, almost unchanged. The
        # three differences are described on the site page; the lines
        # themselves are yours to bring over. Remember: break when done.
        if random.random() < epsilon:
            action = random.choice(ACTIONS)
        else:
            action = best_action(state)

        new_state, reward, done = world.step(action)

        old = Q[state][action]
        if done:
            target = reward
        else:
            target = reward + discount * best_value(new_state)
        Q[state][action] = old + learning_rate * (target - old)
        state = new_state
        steps += 1
        if done:
            break

print("states the agent has seen:", len(Q))   # expect roughly 125-150


# --- watch your creation play --------------------------------------------
def choose(world):
    """Picks the trained agent's move for the current situation.

    Args:
        world: The PacmanWorld object mid-game.

    Returns:
        The action string the trained Q-table rates highest.
    """
    return best_action(world.get_state())

# Un-comment this line when your TODOs above are filled in:
# play(world, choose, delay=0.3)


# ======================================================================
# TEST - check your own work, no peeking at the solution needed.
# WHEN TO RUN: after all the TODOs above are filled in and the file
# trains without errors. Un-comment the block below and re-run: a
# trained agent should win almost every game. (We check behaviour, not
# exact scores.) You should also see "states the agent has seen" land
# in the 125-150 range printed just above.
# ======================================================================

# def check(label, got, expected):
#     mark = "PASS" if got == expected else "FAIL"
#     extra = "" if got == expected else "   (got " + repr(got) + ")"
#     print(mark, label, extra)
#
# wins = 0
# for game in range(20):
#     won, score = play(world, choose, silent=True)
#     if won:
#         wins = wins + 1
# print("won", wins, "out of 20")
# check("trained agent wins at least 18 of 20", wins >= 18, True)
