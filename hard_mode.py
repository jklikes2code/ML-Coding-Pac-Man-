import random
import pacman_world
import pacman_world_hard
from expert_data import expert_data

ACTIONS = ["up", "down", "left", "right"]

# ========================================================================
# TASK 1: Fresh Q-table trained on hard mode
# ========================================================================
print("="*70)
print("TASK 1: Training fresh Q-table on hard mode (chasing ghost)")
print("="*70)

easy_world = pacman_world.PacmanWorld()
hard_world = pacman_world_hard.PacmanWorldHard()

Q = {}

def make_sure_state_exists(state):
    """Adds a state to Q-table if it doesn't exist."""
    if state not in Q:
        Q[state] = {}
        for action in ACTIONS:
            Q[state][action] = 0.0

def best_value(state):
    """Returns the highest Q-value in this state."""
    make_sure_state_exists(state)
    return max(Q[state][a] for a in ACTIONS)

def best_action(state):
    """Returns the action with highest Q-value."""
    make_sure_state_exists(state)
    best_a = "up"
    best = Q[state]["up"]
    for a in ACTIONS:
        if Q[state][a] > best:
            best = Q[state][a]
            best_a = a
    return best_a

# Train on hard world
learning_rate = 0.5
discount = 0.9
epsilon = 0.2

for episode in range(20000):
    state = hard_world.reset()
    make_sure_state_exists(state)
    done = False
    for move in range(100):
        if random.random() < epsilon:
            action = random.choice(ACTIONS)
        else:
            action = best_action(state)

        new_state, reward, done = pacman_world_hard.PacmanWorldHard.step(hard_world, action)
        old = Q[state][action]
        if done:
            target = reward
        else:
            target = reward + discount * best_value(new_state)

        Q[state][action] += learning_rate * (target - old)

        state = new_state
        if done:
            break

print(f"Hard-trained Q-table sees {len(Q)} states")


def choose_hard_qtable(world):
    """Q-table policy for hard world."""
    return best_action(world.get_state())


# Test fresh Q-table on hard mode
wins = 0
for game in range(20):
    won, score = pacman_world_hard.play(hard_world, choose_hard_qtable, silent=True)
    if won:
        wins += 1
print(f"Hard-trained table on hard mode: {wins} out of 20")
print()

# ========================================================================
# TASK 2: Tournament - all three easy-world agents on hard mode
# ========================================================================
print("="*70)
print("TASK 2: Tournament - easy-trained agents vs chasing ghost")
print("="*70)
print()

# -------- AGENT 1: Easy-trained Q-table --------
print("Training Q-table on EASY world...")
Q_easy = {}


def make_sure_state_easy(state):
    if state not in Q_easy:
        Q_easy[state] = {}
        for action in ACTIONS:
            Q_easy[state][action] = 0.0


def best_value_easy(state):
    make_sure_state_easy(state)
    return max(Q_easy[state][a] for a in ACTIONS)


def best_action_easy(state):
    make_sure_state_easy(state)
    best_a = "up"
    best = Q_easy[state]["up"]
    for a in ACTIONS:
        if Q_easy[state][a] > best:
            best = Q_easy[state][a]
            best_a = a
    return best_a


# Train on easy world
easy_world_train = pacman_world.PacmanWorld()
for episode in range(20000):
    state = easy_world_train.reset()
    make_sure_state_easy(state)
    done = False
    for move in range(100):
        if random.random() < epsilon:
            action = random.choice(ACTIONS)
        else:
            action = best_action_easy(state)

        new_state, reward, done = easy_world_train.step(action)
        old = Q_easy[state][action]
        if done:
            target = reward
        else:
            target = reward + discount * best_value_easy(new_state)

        Q_easy[state][action] += learning_rate * (target - old)

        state = new_state
        if done:
            break


def choose_easy_qtable(world):
    """Easy-trained Q-table policy."""
    state = world.get_state()
    if state not in Q_easy:
        return random.choice(ACTIONS)
    best_a = "up"
    best = Q_easy[state]["up"] if "up" in Q_easy[state] else 0.0
    for a in ACTIONS:
        if a in Q_easy[state] and Q_easy[state][a] > best:
            best = Q_easy[state][a]
            best_a = a
    return best_a


# Test easy Q-table on hard mode
print("Testing easy-trained Q-table on hard mode...")
hard_world = pacman_world_hard.PacmanWorldHard()
wins_easy_qtable = 0
for game in range(20):
    won, score = pacman_world_hard.play(hard_world, choose_easy_qtable, silent=True)
    if won:
        wins_easy_qtable += 1
print(f"Easy-trained table on hard mode: {wins_easy_qtable} out of 20")
print()


# -------- AGENT 2: Imitator --------
print("Training imitator on EASY world...")


class Perceptron:
    def __init__(self, num_features):
        self.weights = []
        for i in range(num_features + 1):
            self.weights.append(0.0)
        self.learning_rate = 1

    def score(self, features):
        total = self.weights[0]
        for i in range(len(features)):
            total = total + self.weights[i + 1] * features[i]
        return total

    def predict(self, features):
        total = self.weights[0]
        for i in range(len(features)):
            total += self.weights[i + 1] * features[i]
        if total > 0:
            return 1
        else:
            return 0

    def train(self, data, epochs):
        for e in range(epochs):
            for row in data:
                features = row[:-1]
                label = row[-1]
                prediction = self.predict(features)
                error = label - prediction

                self.weights[0] += self.learning_rate * error
                for i in range(len(features)):
                    self.weights[i + 1] += self.learning_rate * error * features[i]


perceptrons = {}
for action in ACTIONS:
    data = []
    for row in expert_data:
        feats = row[:-1]
        chosen = row[-1]
        if chosen == action:
            data.append(feats + [1])
        else:
            data.append(feats + [0])

    p = Perceptron(12)
    p.train(data, 20)
    perceptrons[action] = p


def imitation_move(world) -> str:
    """Imitator policy for easy world."""
    feats = pacman_world.features(world)
    best_score = -999
    best_action = "up"
    for action in ACTIONS:
        score = perceptrons[action].score(feats)
        if score > best_score:
            best_score = score
            best_action = action
    return best_action


def imitation_move_hard(world):
    """Imitator policy on hard world (uses hard world's features)."""
    feats = pacman_world_hard.features(world)
    best_score = -999
    best_action = "up"
    for action in ACTIONS:
        score = perceptrons[action].score(feats)
        if score > best_score:
            best_score = score
            best_action = action
    return best_action


# Test imitator on hard mode
print("Testing imitator on hard mode...")
hard_world = pacman_world_hard.PacmanWorldHard()
wins_imitator = 0
for game in range(20):
    won, score = pacman_world_hard.play(hard_world, imitation_move_hard, silent=True)
    if won:
        wins_imitator += 1
print(f"Imitator on hard mode: {wins_imitator} out of 20")
print()


# -------- AGENT 3: Four-weight brain on easy world --------
print("Training four-weight brain on EASY world...")

weights_easy = [0.0, 0.0, 0.0, 0.0] # [bias, blocked, toward-dot, ghost-danger]
learning_rate_brain = 0.01
discount_brain = 0.9
epsilon_brain = 0.1

def action_feats(feats, i):
    """The 3 features for action i."""
    return [feats[i*3], feats[i*3+1], feats[i*3+2]]

def q_value_easy(f3):
    """Q-value for one action with weights."""
    return weights_easy[0] * 1 + weights_easy[1] * f3[0] + weights_easy[2] * f3[1] + weights_easy[3] * f3[2]

def best_index_easy(feats):
    """Which action scores highest."""
    best_i = 0
    best = q_value_easy(action_feats(feats, 0))
    for i in range(4):
        v = q_value_easy(action_feats(feats, i))
        if v > best:
            best = v
            best_i = i
    return best_i

# Train four-weight brain on easy world
easy_world_brain = pacman_world.PacmanWorld()
for episode in range(1000):
    easy_world_brain.reset()
    done = False
    for move in range(100):
        feats = pacman_world.features(easy_world_brain)
        if random.random() < epsilon_brain:
            i = random.choice([0, 1, 2, 3])
        else:
            i = best_index_easy(feats)
        f3 = action_feats(feats, i)
        old_q = q_value_easy(f3)
        new_state, reward, done = easy_world_brain.step(ACTIONS[i])
        if done:
            target = reward
        else:
            new_feats = pacman_world.features(easy_world_brain)
            target = reward + discount_brain * q_value_easy(action_feats(new_feats, best_index_easy(new_feats)))
        error = target - old_q

        weights_easy[0] = weights_easy[0] + learning_rate_brain * error * 1
        weights_easy[1] = weights_easy[1] + learning_rate_brain * error * f3[0]
        weights_easy[2] = weights_easy[2] + learning_rate_brain * error * f3[1]
        weights_easy[3] = weights_easy[3] + learning_rate_brain * error * f3[2]
        if done:
            break

print(f"Easy-trained brain weights: {[round(w, 1) for w in weights_easy]}")

def approx_move_easy(world):
    """Four-weight brain policy for easy world."""
    feats = pacman_world.features(world)
    return ACTIONS[best_index_easy(feats)]

# Test four-weight brain on hard mode
print("Testing four-weight brain on hard mode...")
hard_world = pacman_world_hard.PacmanWorldHard()
wins_brain = 0
for game in range(20):
    # Use hard world's features with easy-trained weights
    def brain_hard_move(world):
        feats = pacman_world_hard.features(world)
        return ACTIONS[best_index_easy(feats)]
 
    won, score = pacman_world_hard.play(hard_world, brain_hard_move, silent=True)
    if won:
        wins_brain += 1

print(f"Four-weight brain on hard mode: {wins_brain} out of 20")
print()

# ========================================================================
# SCOREBOARD
# ========================================================================
print("="*70)
print("SCOREBOARD: Easy-trained agents vs Hard mode (chasing ghost)")
print("="*70)
print(f"Easy-trained Q-table: {wins_easy_qtable:2d} out of 20")
print(f"Imitator: {wins_imitator:2d} out of 20")
print(f"Four-weight brain: {wins_brain:2d} out of 20")
print()
print("Expected results:")
print(" Easy-trained Q-table: 0 out of 20 (memorized old situations)")
print(" Imitator: 0 out of 20 (no chaser examples)")
print(" Four-weight brain: 20 out of 20 (generalizable features)")
print()

# ========================================================================
# TASK 3: Retrain four-weight brain on hard mode and compare
# ========================================================================
print("="*70)
print("TASK 3: Retraining four-weight brain ON hard mode")
print("="*70)

weights_hard = [0.0, 0.0, 0.0, 0.0] # [bias, blocked, toward-dot, ghost-danger]

def q_value_hard(f3):
    """Q-value for hard-world brain."""
    return weights_hard[0] * 1 + weights_hard[1] * f3[0] + weights_hard[2] * f3[1] + weights_hard[3] * f3[2]

def best_index_hard(world, feats):
    """Which action scores highest for hard world."""
    best_i = 0
    best = q_value_hard(action_feats(feats, 0))
    for i in range(4):
        v = q_value_hard(action_feats(feats, i))
        if v > best:
            best = v
            best_i = i
    return best_i

# Train four-weight brain on hard world
hard_world = pacman_world_hard.PacmanWorldHard()
for episode in range(1000):
    hard_world.reset()
    done = False
    for move in range(100):
        # Use hard world's features function
        feats = pacman_world_hard.features(hard_world)
        if random.random() < epsilon_brain:
            i = random.choice([0, 1, 2, 3])
        else:
            i = best_index_hard(hard_world, feats)
        f3 = action_feats(feats, i)
        old_q = q_value_hard(f3)
        new_state, reward, done = pacman_world_hard.PacmanWorldHard.step(hard_world, ACTIONS[i])
        if done:
            target = reward
        else:
            new_feats = pacman_world_hard.features(hard_world)
            target = reward + discount_brain * q_value_hard(action_feats(new_feats, best_index_hard(hard_world, new_feats)))
        error = target - old_q

        weights_hard[0] = weights_hard[0] + learning_rate_brain * error * 1
        weights_hard[1] = weights_hard[1] + learning_rate_brain * error * f3[0]
        weights_hard[2] = weights_hard[2] + learning_rate_brain * error * f3[1]
        weights_hard[3] = weights_hard[3] + learning_rate_brain * error * f3[2]
        if done:
            break

print(f"Easy-world brain weights: {[round(w, 1) for w in weights_easy]}")
print(f"Hard-world brain weights: {[round(w, 1) for w in weights_hard]}")
print()
print("Analysis:")
print(f" Bias change: {round(weights_hard[0] - weights_easy[0], 1)} (should be small)")
print(f" Blocked penalty change: {round(weights_hard[1] - weights_easy[1], 1)} (should be small or negative)")
print(f" Toward-dot weight change: {round(weights_hard[2] - weights_easy[2], 1)} (should be small)")
print(f" Ghost-danger weight change: {round(weights_hard[3] - weights_easy[3], 1)} (should be MORE negative - ghost is worse!)")
print()
print("Interpretation:")
print(" The core concepts (walls bad, dots good, ghost bad) stay stable.")
print(" But ghost-danger becomes MUCH MORE negative because the chaser")
print(" is a more immediate threat than a predictable patrol ghost.")
print()

# Final test of hard-trained brain
def brain_hard_trained_move(world):
    feats = pacman_world_hard.features(world)
    return ACTIONS[best_index_hard(world, feats)]

hard_world = pacman_world_hard.PacmanWorldHard()
wins_hard_trained = 0
for game in range(20):
    won, score = pacman_world_hard.play(hard_world, brain_hard_trained_move, silent=True)
    if won:
        wins_hard_trained += 1

print(f"Hard-trained brain on hard mode: {wins_hard_trained} out of 20")