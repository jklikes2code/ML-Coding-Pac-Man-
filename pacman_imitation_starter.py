# Capstone agent #1: the IMITATOR (perceptron-powered PacMan)
#
# Put this file in the same folder as pacman_world.py and expert_data.py.
# You will train four perceptrons - one per direction - on recorded
# games of an expert player, then let the most confident one drive.

from pacman_world import PacmanWorld, ACTIONS, features, play
from expert_data import expert_data

# --- Step 1: your Perceptron from Day 3, plus ONE new method ----------
class Perceptron:
    """Your Day 3 perceptron, plus one new method: score().

    Each direction gets its own Perceptron that learns "would the
    expert move my way in this situation?" from the recorded games.
    """

    def __init__(self, num_features):
        """Sets up an untrained perceptron.

        Args:
            num_features: How many input features (12 for PacMan).
        """
        # TODO: paste your __init__ from the perceptron unit
        self.weights = []
        for i in range(num_features + 1):
            self.weights.append(0.0)
        self.learning_rate = 1

    def score(self, features):
        """Measures how confidently this perceptron says yes (PROVIDED).

        Args:
            features: The 12 situation numbers from features(world).

        Returns:
            The raw weighted sum BEFORE the yes/no threshold. Bigger
            means more confident. Used to compare the four directions.
        """
        total = self.weights[0]
        for i in range(len(features)):
            total += self.weights[i + 1] * features[i]
        return total

    def predict(self, features):
        """Makes the yes/no decision for one situation.

        Args:
            features: The 12 situation numbers.

        Returns:
            1 if score(features) is greater than 0, otherwise 0.
        """
        # TODO: your Day 3 predict, or one line built on score().
        total = self.weights[0]  # start with the bias
        for i in range(len(features)):
            total += self.weights[i + 1] * features[i]
        if total > 0:
            return 1
        else:
            return 0

    def train(self, data, epochs):
        """Trains on rows of [12 features..., label] (label is 0 or 1).

        Args:
            data: The relabeled expert data for ONE direction.
            epochs: Full passes over the data (about 20 works well).
        """
        # TODO: paste your train from the perceptron unit (unchanged!)
        for e in range(epochs):
            for row in data:
                features = row[:-1]
                label = row[-1]
                prediction = self.predict(features)
                error = label - prediction

                # nudge the bias, then each weight, with the update rule
                self.weights[0] += self.learning_rate * error  # update bias
                for i in range(len(features)):
                    self.weights[i + 1] += self.learning_rate * error * features[i]  # update weights


# --- Step 2: train one perceptron per direction ------------------------
# For the "up" perceptron, relabel the expert data: 1 when the expert
# chose up, 0 when it chose anything else. Same idea for each direction.

perceptrons = {}
for action in ACTIONS:
    data = []
    for row in expert_data:
        feats = row[:-1]       # the 12 features
        chosen = row[-1]       # the move the expert made
        # TODO: relabel this row for THIS direction (1 if the expert
        # chose it, else 0) and add it to data.
        if chosen == action:
            data.append(feats + [1])
        else:
            data.append(feats + [0])

    p = Perceptron(12)
    # TODO: train p on data for about 20 epochs
    perceptrons[action] = p
    p.train(data, 20)


# --- Step 3: the agent - ask all four, take the most confident ---------
def imitation_move(world):
    """Picks PacMan's move by asking all four perceptrons.

    Args:
        world: The PacmanWorld object mid-game.

    Returns:
        The action string whose perceptron gives the highest
        score for the current situation.
    """
    feats = features(world)
    # TODO: ask all four perceptrons how confident they are about
    # feats, and return the most confident direction.
    best_score = 0
    best_action = None
    for action in ACTIONS:
        score = perceptrons[action].score(feats)
        if score > best_score:
            best_score = score
            best_action = action
    return best_action


# --- Step 4: watch your creation play -----------------------------------
world = PacmanWorld()
# Un-comment this line when your TODOs above are filled in:
play(world, imitation_move, delay=0.3)


# ======================================================================
# TEST - check your own work, no peeking at the solution needed.
# WHEN TO RUN: after all the TODOs above are filled in and the file runs
# without errors. Un-comment the block below and re-run: a well-trained
# imitator should win almost every game. (We check behaviour, not exact
# scores.) If it wins far fewer, the most likely culprit is the Step 2
# relabeling - print one perceptron's weights to check they are not all
# still zero.
# ======================================================================

def check(label, got, expected):
    mark = "PASS" if got == expected else "FAIL"
    extra = "" if got == expected else "   (got " + repr(got) + ")"
    print(mark, label, extra)

wins = 0
for game in range(20):
    won, score = play(world, imitation_move, silent=True)
    if won:
        wins = wins + 1
print("won", wins, "out of 20")
check("imitator wins at least 18 of 20", wins >= 18, True)
