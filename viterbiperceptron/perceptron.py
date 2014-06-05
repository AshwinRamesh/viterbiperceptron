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

        # Perform Training
        for i in xrange(0, self.iterations):  # For each iteration
            for t in training_data:  # For each training item
                tag_path = self.classify(t)  # TODO - STUB Classify item
                correct_path = True
                for i in xrange(0,len(t)):  # Check if correct path
                    if tag_path[i] != t[i][3]:
                        correct_path = False
                        break
                if not correct_path:  # Update weights if required
                    for word in t:  # Update Gold Classes
                        for feature in word['features']:
                            self.weights[word['gold']][feature] += 1
                    for word in tag_path:  # Update Output Classes
                        for feature in word['features']:
                            self.weights[word['tag']][feature] -= 1

                for tag in self.classes:  # Do summation for averaging
                    for feature in self.weights[tag].keys():
                    self.averaged_weights[tag][feature] += self.weights[tag][feature]

                num_trainings += 1

        # Calculate average weights
        for c in self.classes:
            for f in self.averaged_weights.keys():
                self.averaged_weights[c][f] /= (num_trainings * 1.0)

        print "Training Complete: Iterations: %d | Training Data: %d | Time Taken: %s" % (self.iterations, len(training_data), time.time() - start)

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
            path[tag] = ["<START>"]

        # Viterbi algorithm
        for i in xrange(0, len(sentence)):  # FOR EACH WORD (Column)
            history.append({})
            temp_path = {}
            for tag in self.classes: # FOR EACH TAG (Row)
                coll = []
                for t in self.classes:  # iterate through all tags for previous words
                    features = self._create_features(sentence, i, path, history)  # variable function to create features
                    score = 0
                    for f in features:  # Calculate score for tag-feature set
                        score += self.weights[t][f]
                    score += history[i][t]  # Add preceding weight to the output score
                    coll.append((score, t))  # determine the score and class for this iteration
                output_tag, score = max(coll)
                history[i+1][tag] = score
                temp_path[tag] = path[output_tag] + [tag]
            path = temp_path
        # TODO - backtrack
        return self._backtrack(sentence, history, path)


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
        features = []
        if index == 1:  # first word in the sentence
            features.append("prev-<START>")
        else:  # current word is in the middle of sentence - append best prev. word TAG as feature
            features.append("prev-%s", % history[])
