from collections import defaultdict
import time


class StructuredPerceptron(object):
    """
    @description: Structured perceptron class
    @args:
        iterations (int)
        training_data(list of lists) - [[<word>, <POS tag>, <Syn Chunk Tag>, <Gold NER>],...]
    """

    def __init__(self, training_data, iterations=1):
        self.iterations = iterations  # iterations through training data before training completes
        self.classes = ["B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-MISC", "I-MISC", "O"]
        self.training_data = training_data
        self.weights = {klass: defaultdict(int) for klass in self.classes}
        self.historical_trainings = {klass: defaultdict(int) for klass in self.classes}
        self.averaged_weights = {klass: defaultdict(int) for klass in self.classes}

    def train(self):
        """
        @description: train the perceptron with training data provided
        @args:
            -averaged (bool) - perform averaging after training?
        """

        # Initialise variables and alias variables
        start = time.time()
        training_data = self.training_data
        num_trainings = 0

        for i in xrange(0, self.iterations):  # For each iteration
            for t in training_data:  # For each training item
                tag_path = self.classify(t)  # TODO - STUB Classify item
                correct_path = True
                for i in xrange(0,len(t)):  # Check if correct path
                    if tag_path[i] != t[i][3]:  # TODO - fix STUB
                        correct_path = False
                        break
                if not correct_path:  # UPDATE WEIGHTS - TODO
                    for word in t:  # Update Gold Classes
                        for feature in word['features']:
                            self.weights[word['gold']][feature] += 1
                    for word in tag_path:  # Update Output Classes
                        for feature in word['features']:
                            self.weights[word['tag']][feature] -= 1

                num_trainings += 1

        for c in self.classes:
            for f in self.features:
                self.averaged_weights[c][f] += (num_trainings - self.historical_trainings[c][f]) * self.weights[c][f]
                self.averaged_weights[c][f] /= num_trainings
                self.historical_trainings[c][f] = num_trainings
        print "Training Complete: Iterations: %d | Training Data: %d | Averaged: %s | Time Taken: %s" % (self.iterations, len(training_data), str(averaged), time.time() - start)

    def classify(self, sentence):
        """
        @description: sum the feature vector against the Perceptron weights to
        classify the vector
        @args:
            sentence (list of lists) -
                [[<word>, <gold>], ..] <-- "gold" key is optional
        """
        history = [{}]
        path = {}

        # Base case
        for tag in self.classes:
            history[0][tag] = 0
            path[tag] = [tag]

        # Viterbi algorithm
        for i in xrange(0, len(sentence)):
            history.append({})
            temp_path = {}
            for tag in self.classes: # For each "end case" tag --> find the best possib
                features = self._create_features(sentence, i, path, history)  # variable function to create features
                score, tag_class = self._compute_class(features)  # determine the score and class for this iteration
                history[i][tag] = score
                temp_path[tag] = path[tag_class] + [tag]
            path = temp_path
        # TODO - backtrack
        return path

    def _create_features(self, sentence, index, path, history):
        """
        @description: Variable function that returns a list of features
        :param sentence: list of lists
        :param index: int
        :param path: dict
        :param history: list of dicts
        :return: list of str
        """
        raise Exception("Cannot run base perceptron class")

    def _compute_class(self, features):
        """
        @description: compute the best NER Tag for the given set of features
        :param features: list of str
        :return: (int, str)
        """
        best = None
        best_class = float("-inf")
        weights = self.weights
        for c in self.classes:
            score = 0.0
            for f in features:
                score += weights[c][f]
            if score > best:
                best = score
                best_class = c
        return best, best_class

 #       return self._backtrack(sentence, viterbi_history, viterbi_path)

    def _backtrack(self, sentence, history, path):  # TODO - not sure if correct
        n = len(sentence) - 1
        (prob, state) = max((history[n][tag]) for tag in self.tags)
        return prob, path[state]


class FeaturePerceptronOne(StructuredPerceptron):
    """
    @description:
        Features
        ========
        - Only takes into account the previous word tag
    """

    def _create_features(self, sentence, index, path, history):
