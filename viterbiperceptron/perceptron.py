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
        for i in xrange(1, self.iterations+1):  # For each iteration

            trains_done_in_iteration = 0

            for t in training_data:  # For each training item

                if trains_done_in_iteration % 200 == 0:
                    print "On iteration: %d | Completed: %d / %d" % (i, trains_done_in_iteration, len(training_data))

                tag_path = self.classify(t)  # Determine the best NER tags for sentence

                # Check if Gold Standard Path == Output Path
                correct_path = True
                for j in xrange(0,len(t)):  # Check if correct path
                    if tag_path[j] != t[j][3]:
                        correct_path = False
                        break

                # Update weights accordingly
                if not correct_path:  # Update weights if required
                    for word in t:  # Update Gold Classes
                        for feature in self.weights[word[3]].keys(): # For each feature in the weights for that TAG
                            self.weights[word[3]][feature] += 1
                    for j in xrange(0, len(t)):  # Update Output Classes
                        for feature in self.weights[tag_path[j]].keys():  # For each feature in tagged TAG
                            self.weights[tag_path[j]][feature] -= 1

                # Sum up the weights into the averaged weights
                for tag in self.classes:  # Do summation for averaging
                    for feature in self.weights[tag].keys():
                        self.averaged_weights[tag][feature] += self.weights[tag][feature]

                num_trainings += 1
                trains_done_in_iteration += 1
            print "Finished iteration %d" % i

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

        # Initialise Viterbi history/score tables
        score_history = [{}]
        path_history = [{}]
        coll = []
        # Base case
        for tag in self.classes:
            score_history[0][tag] = 0
            path_history[0][tag] = "<START>"

        # Viterbi algorithm
        for i in xrange(1, len(sentence)):  # For each word in the sentence (Column)
            score_history.append({})
            path_history.append({})
            for tag in self.classes:  # For each tag in the classes (Row)
                coll = []
                for t in self.classes:  # iterate through all tags for previous words
                    features = self._create_features(sentence, i-1, tag, t, path_history, score_history)  # variable function to create features
                    score = score_history[i-1][t] # Add preceding weight to the output score
                    for f in features:  # Calculate score for tag-feature set
                        score += self.weights[tag][f]
                    coll.append((score, t))  # determine the score and class for this iteration
                out = max(coll)
                score = out[0]
                output_tag = out[1]
                score_history[i][tag] = score
                path_history[i][tag] = output_tag

        output_sentence = self._backtrack(len(sentence), path_history, score_history)  # Backtrack to return the output tag sequence
        return output_sentence


    def _create_features(self, sentence, index, outer_tag, inner_tag, path_history, score_history):
        """
        @description: Variable function that returns a list of features
        :param sentence: list of lists
        :param index: int
        :param outer_tag: str
        :param inner_tag: str
        :param path_history: dict
        :param score_history: list of dicts
        :return: list of str
        """
        raise Exception("Cannot run base perceptron class")

 #       return self._backtrack(sentence, viterbi_history, viterbi_path)

    def _backtrack(self, sentence_length, history, scores):  # TODO - not sure if correct
        sentence = [None] * sentence_length
        max_score = float("-inf")
        for tag in scores[sentence_length-1].keys():
            score = scores[sentence_length-1][tag]
            if score > max_score:
                max_score = score
                sentence[sentence_length-1] = tag

        next_tag = sentence[sentence_length-1]
        for i in xrange(sentence_length - 1, 0, -1):  # back track
            sentence[i-1] = history[i][next_tag]
            next_tag = sentence[i-1]

        return sentence


class FeaturePerceptronOne(StructuredPerceptron):
    """
    @description:
        Features
        ========
        - Only takes into account the previous word tag
    """

    def _create_features(self, sentence, word_index, outer_tag, inner_tag, path_history, score_history):
        features = []
        if word_index == 0:  # first word in the sentence
            features.append("prev-<START>")
        else:  # current word is in the middle of sentence - append best prev. word TAG as feature
            features.append("prev-<%s>" % inner_tag)
        return features