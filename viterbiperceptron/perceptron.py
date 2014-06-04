from collections import defaultdict
import time


class AveragedPerceptron(object):
    """
    @description: Averaged perceptron class
    """

    def __init__(self, features, classes, training_data=[], iterations=1, skip_averaging=False):
        self.iterations = iterations  # iterations through training data before training completes
        self.skip_averaging = skip_averaging  # Skip averaging
        self.features = features  # list
        self.classes = classes  # list
        self.training_data = training_data  # [{'class':X,'weights':{..}}] <-- structure
        self.weights = {}  # the current weights for the perceptron
        self.averaged_weights = {}  # the averaged weights (if averaging is enabled)
        self.historical_trainings = {}  # the number of iterations processed by a class (used for lazy averaging)

    def train(self):
        """
        @description: train the perceptron with training data provided
        @args:
            -averaged (bool) - perform averaging after training?
        """

        # Initialise variables and alias variables
        start = time.time()
        training_data = self.training_data
        averaged = not (self.skip_averaging)
        num_trainings = 0
        self.weights = {klass: defaultdict(int) for klass in self.classes}
        if averaged:
            self.historical_trainings = {klass: defaultdict(int) for klass in self.classes}
            self.averaged_weights = {klass: defaultdict(int) for klass in self.classes}

        for i in xrange(0, self.iterations):  # For each iteration
            for t in training_data:  # For each training item
                gold_class = t['class']
                output_score, output_class = self.classify(t['weights'])  # Classify item

                if not output_class == gold_class:  # Check classification and update weights
                    # Perform weight update
                    data = t['weights']

                    for f in data.keys():
                        if averaged:  # Perform lazy update
                            self.averaged_weights[gold_class][f] += self.weights[gold_class][f] * (num_trainings - self.historical_trainings[gold_class][f])
                            self.averaged_weights[output_class][f] += self.weights[output_class][f] * (num_trainings - self.historical_trainings[output_class][f])
                            self.historical_trainings[gold_class][f] = self.historical_trainings[output_class][f] = num_trainings
                        # Update weights
                        self.weights[gold_class][f] += data[f]
                        self.weights[output_class][f] -= data[f]

                num_trainings += 1
        if averaged:  # Update all classes / get avg
            for c in self.classes:
                for f in self.features:
                    self.averaged_weights[c][f] += (num_trainings - self.historical_trainings[c][f]) * self.weights[c][f]
                    self.averaged_weights[c][f] /= num_trainings
                    self.historical_trainings[c][f] = num_trainings
        print "Training Complete: Iterations: %d | Training Data: %d | Averaged: %s | Time Taken: %s" % (self.iterations, len(training_data), str(averaged), time.time() - start)

    def classify(self, feature_data_set):
        """
        @description: sum the feature vector against the Perceptron weights to
        classify the vector
        @feature_vector - dict e.g. {"featureA": 0, "featureB": 1 ..}
        """
        best = None
        best_class = float("-inf")
        weights = self.weights
        for c in self.classes:
            score = 0.0
            for f in feature_data_set.keys():
                score += feature_data_set[f] * weights[c][f]
            if score > best:
                best = score
                best_class = c
        return best, best_class
