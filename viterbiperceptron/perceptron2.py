from collections import defaultdict
import time


class Word(object):
    """
    Object form of the word within a sentence
    """

    def __init__(self, word_array):
        """
        Initialise the word object
        :arg word_array (<list>) - contains all data for the word
        """

        self.name = word_array[0]  # Actual word

        if len(word_array) == 4:  # English
            self.lemma = None
        elif len(word_array) == 5:  # German
            self.lemma = word_array[1]
        else:
            raise Exception("Incorrect word_array param provided. Too many args")

        self.POS = word_array[-3]
        self.syn_chunk = word_array[-2]
        self.NER = word_array[-1]

        # Initialise other variables
        self.NER_out = None  # Output NER classification
        self.gold_features = []  # Features of the gold standard
        self.output_features = []  # Features of the output standard
        self.output_feature_sets = {}  # Dict of lists (features) - one for each NER tag
        self.output_score = float("-inf")  # The classification score up to this word

    def add_feature_set(self, key, feature_set):
        if key not in self.output_feature_sets.keys():
            self.output_feature_sets[key] = feature_set
        else:
            raise Exception("Feature set for key %s exists" % key)

    def set_output_features(self, classified_tag):
        """
        Set the output_feature set
        :param classified_tag: <str>
        """
        try:
            self.output_features = self.output_feature_sets[classified_tag]
        except:  # key error
            raise Exception("Setting output features failed. Tag set does not exist.")

    def update_output(self, tag, features, score):
        """
        Checks the current score against an input score to see if a given classification is better.
        If it is, the NER_out is updated
        :arg: tag <str> (NER tag)
        :arg: features <list - str>
        :arg: score <int>
        """
        if self.output_score < score:
            self.NER_out = tag
            self.output_features = features
            self.output_score = score


class Sentence(object):
    """
    Object wrapper for a sentence
    """

    def __init__(self, german=False):
        self.german = german  # Bool - see if data is german
        self.words = []

    def length(self):
        return len(self.words)

    def append(self, word):
        assert isinstance(word, Word)
        self.words.append(word)

    def get(self, index):
        try:
            return self.words[index]
        except:  # Out of range exception
            raise Exception("No word at that index")

    def convert_to_string(self):
        """
        Convert a sentence to standard output
        """
        output = ""
        if self.german:
            for word in self.words:
                output += "%s %s %s %s %s %s\n" % (word.name, word.lemma, word.POS, word.syn_chunk, word.NER, word.NER_out)
        else:
            for word in self.words:
                output += "%s %s %s %s %s\n" % (word.name, word.POS, word.syn_chunk, word.NER, word.NER_out)
        return output


class StructuredPerceptron(object):
    """
    @description: Structured perceptron class
    @args:
        iterations (int)
        training_data(list of Sentences) - [Sentence, ...]
    """

    SENTENCE_START_TAG = "<START>"

    def __init__(self, training_data, iterations=1):
        self.iterations = iterations
        self.training_data = training_data
        self.classes = ["B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-MISC", "I-MISC", "O"]
        self.weights = {klass: defaultdict(int) for klass in self.classes}
        #self.historical_trainings = {klass: defaultdict(int) for klass in self.classes}  - TODO later
        self.averaged_weights = {klass: defaultdict(int) for klass in self.classes}

    def update_weights(self, sentence):
        """
        @description: update the weights for the perceptron
        :arg: Sentence <Sentence>
        :return:
        """
        assert(isinstance(sentence, Sentence))

        # Check if Gold === Output for sentence
        correct = True
        for i in xrange(0, sentence.length()):
            word = sentence.get(i)
            if word.NER != word.NER_out:
                correct = False
                break

        if not correct:  # Not exact match. Need to update weights
            for i in xrange(0, sentence.length()):
                word = sentence.get(i)
                word.gold_features = self._create_gold_features(sentence, i)  # TODO - STUBBED
                for feature in word.gold_features:
                    self.weights[word.NER][feature] += 1
                for feature in word.output_features:
                    self.weights[word.NER_out][feature] -= 1

        # TODO - do lazy update here

        # Sum up the weights into the averaged weights  - TODO remove this for lazy update later
        for tag in self.classes:  # Do summation for averaging
            for feature in self.weights[tag].keys():
                self.averaged_weights[tag][feature] += self.weights[tag][feature]

    def perform_averaging(self, trainings):
        for c in self.classes:
            for f in self.averaged_weights.keys():
                self.averaged_weights[c][f] /= (trainings * 1.0)

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
            for t in training_data:  # For each training item
                self.classify(t)  # Classify sentence and store the output + features
                self.update_weights(t)  # Update weights if necessary
                num_trainings += 1  # Increment training item count
            print "Finished iteration %d" % i

        # Calculate average weights
        self.perform_averaging(num_trainings)

        print "Training Complete: Iterations: %d | Training Data: %d | Time Taken: %s" % (self.iterations, len(training_data), time.time() - start)

    def classify(self, sentence):
        """
        @description: perform the Viterbi algorithm to determine best sequence
            of NER tags for an input sentence
        :arg: sentence <list - Words>
        """

        # Initialise Viterbi history/score tables
        score_history = [{}]
        path_history = [{}]

        # Base case - Initialise start of sentence
        for tag in self.classes:
            score_history[0][tag] = 0
            path_history[0][tag] = self.SENTENCE_START_TAG

        # Viterbi algorithm
        final_top_score = float('-inf')
        final_top_tag = None
        sen_length = len(sentence)

        for i in xrange(1, sen_length):  # For each word in the sentence (Column)
            word = sentence.get(i)  # get the word object
            score_history.append({})
            path_history.append({})
            for tag in self.classes:  # For each tag in the classes (Row)
                best_score = float("-inf")
                best_tag = None
                best_features = []
                for t in self.classes:  # iterate through all tags for previous words
                    features = self._create_output_features(sentence, t, i)  # Create features for prev-curr
                    score = score_history[i-1][t]  # Add preceding weight to the output score
                    for f in features:  # Calculate score for tag-feature set
                        score += self.weights[tag][f]
                    if score > best_score:
                        best_features = features  # Add features to temporary dict
                        best_tag = t
                        best_score = score

                # Set history/score/feature_set
                word.add_feature_set(key=tag, feature_set=best_features)
                score_history[i][tag] = best_score
                path_history[i][tag] = best_tag

                # Last word being processed - determine the best final word tag (for effeciency)
                if i == sen_length - 1 and best_score > final_top_score:
                    final_top_score = best_score
                    final_top_tag = best_tag

        word.NER_out = final_top_tag  # set the final tag
        word.set_output_features(final_top_tag)  # set the final feature set for the word
        self._backtrack(sentence, path_history)  # Back track to determine output sequence

    def _backtrack(self, sentence, path_history):  # TODO - not sure if correct
        """
        Back tracking algorithm to find best sentence tag sequence
        :param sentence: <Sentence>
        :param path_history: <[{},{}..]>
        """

        for i in xrange(sentence.length(), 1, -1):  # iterate backwards through sentence
            current_word = sentence.get(i-1)
            tag = current_word.NER_out
            prev_tag = path_history[i][tag]
            if prev_tag != self.SENTENCE_START_TAG:  # not reached the start of sentence
                prev_word = sentence.get(i-2)
                prev_word.NER_out = prev_tag
                prev_word.set_output_features(prev_tag)

    def _create_output_features(self, sentence, prev_tag, index):
        return []

    def _create_gold_features(self, sentence, index):
        return []


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

    def _create_weight_update_features(self, sentence, index, gold_data):
        raise Exception("Cannot run base perceptron class")


























































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
            features.append("prev-NER-<%s>" % inner_tag)
        return features

    def _create_weight_update_features(self, sentence, index, gold_data):
        features = []
        if index == 0:
            features.append("prev-<START>")
        else:
            features.append("prev-NER-<%s>" % sentence[index-1])
        return features

class FeaturePerceptronTwo(StructuredPerceptron):
    """
    @description:
        Features
        ========
        - Takes into account Previous NER tag, +2, -2 and current word, +2, -2 and current POS tag
    """

    def _create_features(self, sentence, word_index, outer_tag, inner_tag, path_history, score_history):
        features = []
        features.append("current-WORD-<%s>" % sentence[word_index][0])  # Current word
        features.append("current-POS-<%s>" % sentence[word_index][1])   # Current POS tag
        try:
            features.append("current(%d)-NER-<%s>" % (-1, inner_tag))  # Current -1 NER tag
        except:
            features.append("current(%d)-NER-<%d>" % (-1, "START"))

        try:
            features.append("current(%d)-WORD-<%s>" % (1, sentence[word_index+1][0]))  # Current +1 word
        except:
            features.append("current(%d)-WORD-<%s>" % (1, "END"))
        try:
            features.append("current(%d)-WORD-<%s>" % (2, sentence[word_index+2][0]))  # Current +2 word
        except:
            features.append("current(%d)-WORD-<%s>" % (2, "END-END"))
        try:
            features.append("current(%d)-WORD-<%s>" % (-1, sentence[word_index-1][0]))  # Current -1 word
        except:
            features.append("current(%d)-WORD-<%s>" % (-1, "START"))
        try:
            features.append("current(%d)-WORD-<%s>" % (-2, sentence[word_index-2][0]))  # Current -2 word
        except:
            features.append("current(%d)-WORD-<%s>" % (-2, "START-START"))

        try:
            features.append("current(%d)-POS-<%s>" % (1, sentence[word_index+1][1]))  # Current +1 POS
        except:
            features.append("current(%d)-POS-<%s>" % (1, "END"))
        try:
            features.append("current(%d)-POS-<%s>" % (2, sentence[word_index+2][1]))  # Current +2 POS
        except:
            features.append("current(%d)-POS-<%s>" % (2, "END-END"))
        try:
            features.append("current(%d)-POS-<%s>" % (-1, sentence[word_index-1][1]))  # Current -1 POS
        except:
            features.append("current(%d)-POS-<%s>" % (-1, "START"))
        try:
            features.append("current(%d)-POS-<%s>" % (-2, sentence[word_index-2][1]))  # Current -2 POS
        except:
            features.append("current(%d)-POS-<%s>" % (-2, "START-START"))
        return features

    def _create_weight_update_features(self, sentence, index, gold_data):
        features = []
        features.append("current-WORD-<%s>" % gold_data[index][0])  # Current word
        features.append("current-POS-<%s>" % gold_data[index][1])   # Current POS tag
        try:
            features.append("current(%d)-NER-<%s>" % (-1, sentence[index]))  # Current -1 NER tag
        except:
            features.append("current(%d)-NER-<%d>" % (-1, "START"))

        try:
            features.append("current(%d)-WORD-<%s>" % (1, gold_data[index+1][0]))  # Current +1 word
        except:
            features.append("current(%d)-WORD-<%s>" % (1, "END"))
        try:
            features.append("current(%d)-WORD-<%s>" % (2, gold_data[index+2][0]))  # Current +2 word
        except:
            features.append("current(%d)-WORD-<%s>" % (2, "END-END"))
        try:
            features.append("current(%d)-WORD-<%s>" % (-1, gold_data[index-1][0]))  # Current -1 word
        except:
            features.append("current(%d)-WORD-<%s>" % (-1, "START"))
        try:
            features.append("current(%d)-WORD-<%s>" % (-2, gold_data[index-2][0]))  # Current -2 word
        except:
            features.append("current(%d)-WORD-<%s>" % (-2, "START-START"))

        try:
            features.append("current(%d)-POS-<%s>" % (1, gold_data[index+1][1]))  # Current +1 POS
        except:
            features.append("current(%d)-POS-<%s>" % (1, "END"))
        try:
            features.append("current(%d)-POS-<%s>" % (2, gold_data[index+2][1]))  # Current +2 POS
        except:
            features.append("current(%d)-POS-<%s>" % (2, "END-END"))
        try:
            features.append("current(%d)-POS-<%s>" % (-1, gold_data[index-1][1]))  # Current -1 POS
        except:
            features.append("current(%d)-POS-<%s>" % (-1, "START"))
        try:
            features.append("current(%d)-POS-<%s>" % (-2, gold_data[index-2][1]))  # Current -2 POS
        except:
            features.append("current(%d)-POS-<%s>" % (-2, "START-START"))
        return features