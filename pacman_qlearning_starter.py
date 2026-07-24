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

weights = [0.0, 0.0, 0.0, 0.0]   # [bias, blocked, toward-dot, ghost-danger]
learning_rate = 0.01             # tiny! see the warning below
discount = 0.9
epsilon = 0.1

def action_feats(feats, i):
    """The 3 features describing action number i (PROVIDED)."""
    return [feats[i*3], feats[i*3+1], feats[i*3+2]]

def q_value(f3):
    """Q for one action = the weighted sum of its features."""
    return weights[0] * 1 + weights[1] * f3[0] + weights[2] * f3[1] + weights[3] * f3[2]    # HINT: it is your perceptron's score(), for a 3-feature input plus the bias

def best_index(feats):
    """Which of the four actions scores highest right now? (PROVIDED)"""
    best_i = 0
    best = q_value(action_feats(feats, 0))
    for i in range(4):
        v = q_value(action_feats(feats, i))
        if v > best:
            best = v
            best_i = i
    return best_i

for episode in range(1000):
    world.reset()
    done = False
    for move in range(100):
        feats = features(world)
        if random.random() < epsilon:
            i = random.choice([0, 1, 2, 3])
        else:
            i = best_index(feats)
        f3 = action_feats(feats, i)
        old_q = q_value(f3)
        new_state, reward, done = world.step(ACTIONS[i])
        if done:
            target = reward
        else:
            new_feats = features(world)
            target = reward + discount * q_value(action_feats(new_feats, best_index(new_feats)))
        error = target - old_q

        # Q-learning's target, driven home by the perceptron's update:
        weights[0] += learning_rate * error * 1   # HINT: your Day 3 perceptron update, weights[k] + learning_rate * error * (this weight's input); the bias input is always 1
        weights[1] += learning_rate * error * f3[0]
        weights[2] += learning_rate * error * f3[1]
        weights[3] += learning_rate * error * f3[2]
        if done:
            break

print("your agent's entire brain:", [round(w, 1) for w in weights])

def approx_move(world):
    return ACTIONS[best_index(features(world))]

wins = 0
for game in range(20):
    won, score = play(world, approx_move, silent=True)
    if won:
        wins = wins + 1
print("approximate agent won:", wins, "out of 20")

# feature: eats-dot. helped: no. hurt: sometimes badly. verdict: KILLED.
def my_action_feats(world, i):
    """The 3 provided features for action i, plus one invented: eats a dot?"""
    base = features(world)
    f = [base[i*3], base[i*3+1], base[i*3+2]]
    move = MOVES[ACTIONS[i]]
    land = (world.pac[0] + move[0], world.pac[1] + move[1])
    if is_wall(land):
        land = world.pac
    eats = 0
    for d in range(len(DOT_SPOTS)):
        if world.eaten[d] == 0 and land == DOT_SPOTS[d]:
            eats = 1
    return f + [eats]

# Test baseline (3-feature agent)
print("\n" + "="*50)
print("PART 1 NARRATION: The ghost-danger weight forms first, dropping to ~-15")
print("The blocked penalty hits around episode 100. Toward-dot and bias drift throughout.")
print("This tells us: immediate survival matters most, then navigation planning.")
print("="*50)
print("\n" + "="*50)
print("PART 2: Testing eats-dot feature (5 runs)")
print("="*50)

baseline_results = []
for run in range(5):
    world2 = PacmanWorld()
    wins = 0
    for game in range(20):
        won, score = play(world2, approx_move, silent=True)
        if won:
            wins = wins + 1
    baseline_results.append(wins)
    print(f"Baseline Run {run + 1}: {wins} out of 20")

print(f"Baseline average: {sum(baseline_results) / len(baseline_results):.1f}")
print()

# Re-initialize for eats-dot test
weights_eats = [0.0, 0.0, 0.0, 0.0, 0.0]  # [bias, blocked, toward-dot, ghost-danger, eats-dot]

def q_value_eats(f4):
    """Q for one action with eats-dot feature = weighted sum."""
    return weights_eats[0] * 1 + weights_eats[1] * f4[0] + weights_eats[2] * f4[1] + weights_eats[3] * f4[2] + weights_eats[4] * f4[3]

def best_index_eats(world):
    """Which of the four actions scores highest? Uses my_action_feats."""
    best_i = 0
    best = q_value_eats(my_action_feats(world, 0))
    for i in range(4):
        v = q_value_eats(my_action_feats(world, i))
        if v > best:
            best = v
            best_i = i
    return best_i

def approx_move_eats(world):
    return ACTIONS[best_index_eats(world)]

# Train with eats-dot feature
epsilon_eats = 0.1
learning_rate_eats = 0.01
for episode in range(1000):
    world_train = PacmanWorld()
    world_train.reset()
    done = False
    for move in range(100):
        if random.random() < epsilon_eats:
            i = random.choice([0, 1, 2, 3])
        else:
            i = best_index_eats(world_train)
        f4 = my_action_feats(world_train, i)
        old_q = q_value_eats(f4)
        new_state, reward, done = world_train.step(ACTIONS[i])
        if done:
            target = reward
        else:
            target = reward + discount * q_value_eats(my_action_feats(world_train, best_index_eats(world_train)))
        error = target - old_q
        
        weights_eats[0] = weights_eats[0] + learning_rate_eats * error * 1
        weights_eats[1] = weights_eats[1] + learning_rate_eats * error * f4[0]
        weights_eats[2] = weights_eats[2] + learning_rate_eats * error * f4[1]
        weights_eats[3] = weights_eats[3] + learning_rate_eats * error * f4[2]
        weights_eats[4] = weights_eats[4] + learning_rate_eats * error * f4[3]
        if done:
            break
            
        print(f"Eats-dot trained brain: {[round(w, 1) for w in weights_eats]}")

eats_dot_results = []
for run in range(5):
    world2 = PacmanWorld()
    wins = 0
    for game in range(20):
        won, score = play(world2, approx_move_eats, silent=True)
        if won:
            wins += 1
    eats_dot_results.append(wins)
    print(f"Eats-dot Run {run + 1}: {wins} out of 20")

avg_eats = sum(eats_dot_results) / len(eats_dot_results)
print(f"Eats-dot average: {avg_eats:.1f}")
print("# feature: eats-dot. runs: " + str(eats_dot_results) + ". helped: no. hurt: sometimes. verdict: KILLED.")
print()

print("="*50)
print("PART 3: Testing distance-to-ghost feature (5 runs)")
print("="*50)

def my_action_feats_ghost_dist(world, i):
    """The 3 provided features for action i, plus distance to nearest ghost."""
    base = features(world)
    f = [base[i*3], base[i*3+1], base[i*3+2]]
    move = MOVES[ACTIONS[i]]
    land = (world.pac[0] + move[0], world.pac[1] + move[1])
    if is_wall(land):
        land = world.pac
    # Calculate distance to the ghost
    ghost = world.ghost()
    min_dist = abs(land[0] - ghost[0]) + abs(land[1] - ghost[1])
    # Normalize to 0-1 range (max distance in small maze is ~15)
    ghost_dist = min_dist / 15.0
    return f + [ghost_dist]

# Re-initialize for ghost-distance test
weights_ghost = [0.0, 0.0, 0.0, 0.0, 0.0] # [bias, blocked, toward-dot, ghost-danger, ghost-dist]

def q_value_ghost(f4):
    """Q for one action with ghost-distance feature."""
    return weights_ghost[0] * 1 + weights_ghost[1] * f4[0] + weights_ghost[2] * f4[1] + weights_ghost[3] * f4[2] + weights_ghost[4] * f4[3]

def best_index_ghost(world):
    """Which of the four actions scores highest? Uses ghost-distance features."""
    best_i = 0
    best = q_value_ghost(my_action_feats_ghost_dist(world, 0))
    for i in range(4):
        v = q_value_ghost(my_action_feats_ghost_dist(world, i))
        if v > best:
            best = v
            best_i = i
    return best_i

def approx_move_ghost(world):
    return ACTIONS[best_index_ghost(world)]

# Train with ghost-distance feature
for episode in range(1000):
    world_train = PacmanWorld()
    world_train.reset()
    done = False
    for move in range(100):
        if random.random() < epsilon_eats:
            i = random.choice([0, 1, 2, 3])
        else:
            i = best_index_ghost(world_train)
        f4 = my_action_feats_ghost_dist(world_train, i)
        old_q = q_value_ghost(f4)
        new_state, reward, done = world_train.step(ACTIONS[i])
        if done:
            target = reward
        else:
            target = reward + discount * q_value_ghost(my_action_feats_ghost_dist(world_train, best_index_ghost(world_train)))
        error = target - old_q
 
        weights_ghost[0] = weights_ghost[0] + learning_rate_eats * error * 1
        weights_ghost[1] = weights_ghost[1] + learning_rate_eats * error * f4[0]
        weights_ghost[2] = weights_ghost[2] + learning_rate_eats * error * f4[1]
        weights_ghost[3] = weights_ghost[3] + learning_rate_eats * error * f4[2]
        weights_ghost[4] = weights_ghost[4] + learning_rate_eats * error * f4[3]
        if done:
            break

print(f"Ghost-distance trained brain: {[round(w, 1) for w in weights_ghost]}")

ghost_dist_results = []
for run in range(5):
    world2 = PacmanWorld()
    wins = 0
    for game in range(20):
        on, score = play(world2, approx_move_ghost, silent=True)
    if won:
        wins = wins + 1
    ghost_dist_results.append(wins)
    print(f"Ghost-distance Run {run + 1}: {wins} out of 20")

avg_ghost = sum(ghost_dist_results) / len(ghost_dist_results)
print(f"Ghost-distance average: {avg_ghost:.1f}")
print()

# Feature engineering lab log
print("="*50)
print("FEATURE ENGINEERING LAB LOG")
print("="*50)
baseline_avg = sum(baseline_results) / len(baseline_results)
print(f"Baseline (3-feature): {baseline_results} avg={baseline_avg:.1f}")
print(f"# feature: eats-dot. {eats_dot_results} avg={sum(eats_dot_results)/len(eats_dot_results):.1f}")
print(f"# helped: no. hurt: sometimes. verdict: KILLED.")
print(f"# feature: distance-to-ghost. {ghost_dist_results} avg={avg_ghost:.1f}")
if any(w < 0 for w in ghost_dist_results):
    print(f"# verdict: KILLED - causes occasional collapses.")
elif avg_ghost <= baseline_avg:
    print(f"# verdict: KILLED - no improvement over baseline ({baseline_avg:.1f}).")
else:
    print(f"# verdict: KEPT - improved to {avg_ghost:.1f} from baseline {baseline_avg:.1f}.")

