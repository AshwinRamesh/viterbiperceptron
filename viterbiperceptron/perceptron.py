from collections import defaultdict
import copy
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

    def add_feature_set(self, key, feature_set):
        self.output_feature_sets[key] = feature_set

    def set_output_features(self, classified_tag):
        """
        Set the output_feature set
        :param classified_tag: <str>
        """
        self.output_features = self.output_feature_sets[classified_tag]


class Sentence(object):
    """
    Object wrapper for a sentence
    """

    def __init__(self, german=False):
        self.german = german  # Bool - see if data is german
        self.words = []

    @staticmethod
    def create_sentence_from_list(word_list, german=False):
        """
        Convert a list of lists to a Sentence of Words
        """
        s = Sentence(german)
        for word in word_list:
            s.append(Word(word))
        return s

    def length(self):
        return len(self.words)

    def append(self, word):
        self.words.append(word)

    def get(self, index):
        return self.words[index]

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
        return output + "\n"


class Trellis:
    trell = []

    def __init__(self, perceptron, sentence):
        """
        :type perceptron: StructuredPerceptron
        :type sentence: Sentence
        """
        self.trell = []
        self.perceptron = perceptron
        self.sentence = sentence
        temp = {}
        for label in perceptron.classes:  # FOR EACH NER TAG
           temp[label] = [0, None] # CREATE BASE TABLE by tags
        for word in sentence.words:  # FOR EACH WORD
            self.trell.append([word, copy.deepcopy(temp)])  # extend base table by words
        self.fill_in()
        self.backtrack()

    def fill_in(self):
        perceptron = self.perceptron
        sentence = self.sentence
        for i in xrange(0, len(self.trell)): # For each word
            for token in self.trell[i][1]:  # For each tag
                word = self.trell[i][0] # Get the word
                if i == 0:  # If we are on the first word
                    output_features = perceptron._create_output_features(sentence, token, i)
                    word.add_feature_set(key=token, feature_set=output_features)
                    score = 0
                    for f in output_features:  # Calculate score for tag-feature set
                        score += perceptron.weights[token][f]
                    self.trell[i][1][token][0] = score  # Set the base score for the first word
                else:
                    max_score = None  # max score
                    guess = None  # best guess tag for current index
                    c = None  # temp score
                    for k in self.trell[i-1][1]:  # For each tag in the previous column (trellis)
                        output_features = perceptron._create_output_features(sentence, k, i)
                        word.add_feature_set(key=k, feature_set=output_features)
                        score = 0
                        for f in output_features:  # Calculate score for tag-feature set
                            score += perceptron.weights[token][f]
                        c = self.trell[i-1][1][k][0] + score  # Add the score of previous with computed score with current token + prev token
                        if max_score is None or c > max:  # Determine best score
                            max_score = c
                            guess = k
                    #max_score += hmm.e(token,word)  # Add weights
                    self.trell[i][1][token][0] = max_score  # Set the weight for current index
                    self.trell[i][1][token][1] = guess  # Set the determined previous tag for current index

    def backtrack(self):
        tokens = []
        token = None
        for i in xrange(len(self.trell)-1,-1,-1):
            if token == None:
                max = None
                guess = None
                for k in self.trell[i][1]:
                    if max == None or self.trell[i][1][k][0] > max:
                        max = self.trell[i][1][k][0]
                        token = self.trell[i][1][k][1]
                        guess = k
                tokens.append(guess)
            else:
                tokens.append(token)
                token = self.trell[i][1][token][1]
        tokens.reverse()

        for i in xrange(0, len(tokens)):
            self.sentence.words[i].set_output_features(tokens[i])
            self.sentence.words[i].NER_out = tokens[i]


class StructuredPerceptron(object):
    """
    @description: Structured perceptron class
    :args:
        iterations (int)
        training_data(list of Sentences) - [Sentence, ...]
    """

    SENTENCE_START_TAG = "<START>"
    SENTENCE_END_TAG = "<END>"

    def __init__(self, training_data, iterations=1):
        self.iterations = iterations
        self.training_data = training_data
        self.classes = ["B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-MISC", "I-MISC", "O"]
        self.weights = {klass: defaultdict(int) for klass in self.classes}
        self.historical_trainings = {klass: defaultdict(int) for klass in self.classes}
        self.averaged_weights = {klass: defaultdict(int) for klass in self.classes}

    def update_weights(self, sentence, trainings):
        """
        @description: update the weights for the perceptron
        :arg: Sentence <Sentence>
        :arg: trainings <int> - number of training instances done
        """
        for i in xrange(0, sentence.length()):
            word = sentence.get(i)
            if word.NER != word.NER_out:
                word.gold_features = self._create_gold_features(sentence, i)
                for feature in word.gold_features:
                    self.averaged_weights[word.NER][feature] += self.weights[word.NER][feature] * (trainings - self.historical_trainings[word.NER][feature])
                    self.historical_trainings[word.NER][feature] = trainings
                    self.weights[word.NER][feature] += 1
                for feature in word.output_features:
                    self.averaged_weights[word.NER_out][feature] += self.weights[word.NER_out][feature] * (trainings - self.historical_trainings[word.NER_out][feature])
                    self.historical_trainings[word.NER_out][feature] = trainings
                    self.weights[word.NER_out][feature] -= 1

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
                self.classify2(t)
                #self.classify(t)  # Classify sentence and store the output + features
                self.update_weights(t, num_trainings)  # Update weights if necessary
                num_trainings += 1  # Increment training item count
            print "Finished iteration %d | Time %s" % (i, time.time() - start)

        # Calculate average weights
        for c in self.classes:
            for f in self.averaged_weights[c].keys():
                self.averaged_weights[c][f] += (num_trainings - self.historical_trainings[c][f]) * self.weights[c][f]
                self.averaged_weights[c][f] /= num_trainings
                self.historical_trainings[c][f] = num_trainings

        print "Training Complete: Iterations: %d | Training Data: %d | Total Time Taken: %s" % (self.iterations, len(training_data), time.time() - start)


    def classify2(self, sentence):
        """
        :type sentence: Sentence
        :param sentence:
        :return:
        """
        # Setup
        trell = []
        temp = {}
        for label in self.classes:  # FOR EACH NER TAG
           temp[label] = [0, None] # CREATE BASE TABLE by tags
        for word in sentence.words:  # FOR EACH WORD
            trell.append([word, copy.deepcopy(temp)])  # extend base table by words

        # Viterbi
        for i in xrange(0, len(trell)): # For each word
            for token in trell[i][1]:  # For each tag
                word = trell[i][0] # Get the word
                if i == 0:  # If we are on the first word
                    output_features = self._create_output_features(sentence, token, i)
                    word.add_feature_set(key=token, feature_set=output_features)
                    score = 0
                    for f in output_features:  # Calculate score for tag-feature set
                        score += self.weights[token][f]
                    trell[i][1][token][0] = score  # Set the base score for the first word
                else:
                    max_score = None  # max score
                    guess = None  # best guess tag for current index
                    c = None  # temp score
                    for k in trell[i-1][1]:  # For each tag in the previous column (trellis)
                        output_features = self._create_output_features(sentence, k, i)
                        word.add_feature_set(key=k, feature_set=output_features)
                        score = 0
                        for f in output_features:  # Calculate score for tag-feature set
                            score += self.weights[token][f]
                        c = trell[i-1][1][k][0] + score  # Add the score of previous with computed score with current token + prev token
                        if max_score is None or c > max_score:  # Determine best score
                            max_score = c
                            guess = k
                    #max_score += hmm.e(token,word)  # Add weights
                    trell[i][1][token][0] = max_score  # Set the weight for current index
                    trell[i][1][token][1] = guess  # Set the determined previous tag for current index

        # Backtrack
        tokens = []
        token = None
        for i in xrange(len(trell)-1,-1,-1):
            if token is None:
                max = None
                guess = None
                for k in trell[i][1]:
                    if max is None or trell[i][1][k][0] > max:
                        max = trell[i][1][k][0]
                        token = trell[i][1][k][1]
                        guess = k
                tokens.append(guess)
            else:
                tokens.append(token)
                token = trell[i][1][token][1]
        tokens.reverse()
        a = 1
        for i in xrange(0, len(tokens)):
            word = sentence.get(i)
            word.set_output_features(tokens[i])
            word.NER_out = tokens[i]




    def classify(self, sentence):
        """

        :type sentence: Sentence
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
        sen_length = sentence.length()

        for i in xrange(0, sen_length+1):  # For each word in the sentence (Column)
            if i < sen_length:
                word = sentence.get(i)  # get the word object
            else:  # End of sentence
                word = Word([self.SENTENCE_END_TAG]*4)
            score_history.append({})
            path_history.append({})
            for tag in self.classes:  # For each tag in the classes (Row)
                best_score = float("-inf")
                best_tag = None
                best_features = []
                for t in self.classes:  # iterate through all tags for previous words
                    features = self._create_output_features(sentence, t, i)  # Create features for prev-curr
                    score = score_history[i][t]  # Add preceding weight to the output score
                    for f in features:  # Calculate score for tag-feature set
                        score += self.weights[tag][f]
                    if score > best_score:
                        best_features = features  # Add features to temporary dict
                        best_tag = t
                        best_score = score

                # Set history/score/feature_set
                word.add_feature_set(key=tag, feature_set=best_features)
                score_history[i+1][tag] = best_score
                path_history[i+1][tag] = best_tag

                # Last word being processed - determine the best final word tag (for effeciency)
                if i == sen_length and best_score > final_top_score:
                    final_top_score = best_score
                    final_top_tag = best_tag

        # Do Backtracking
        sentence.words[-1].NER_out = final_top_tag  # set the final tag
        sentence.words[-1].set_output_features(final_top_tag)  # set the final feature set for the word
        for i in xrange(sentence.length()-1, 0, -1):  # iterate backwards through sentence
            current_word = sentence.get(i)
            tag = current_word.NER_out
            prev_tag = path_history[i][tag]
            if prev_tag != self.SENTENCE_START_TAG:  # not reached the start of sentence
                prev_word = sentence.get(i-1)
                prev_word.NER_out = prev_tag
                prev_word.set_output_features(prev_tag)

    def _create_output_features(self, sentence, prev_tag, index):
        """
        STUB - need child classes to return data
        """
        raise Exception("Cannot call this method for parent class - StructuredPerceptron")

    def _create_gold_features(self, sentence, index):
        """
        STUB - need child classes to return data
        """
        raise Exception("Cannot call this method for parent class - StructuredPerceptron")


class FeaturePerceptronOne(StructuredPerceptron):
    """
    @description:
        Features
        ========
        - Only takes into account the previous word tag
    """

    def _create_gold_features(self, sentence, index):
        """
        Features:
            * previous NER tag
        """
        features = []
        if index == 0:
            features.append("prev-NER-%s" % self.SENTENCE_START_TAG)
        else:
            features.append("prev-NER-<%s>" % sentence.get(index-1).NER)
        return features

    def _create_output_features(self, sentence, prev_tag, index):
        """
        Features:
            * previous NER tag
        """
        features = []
        if index == 0:
            features.append("prev-NER-%s" % self.SENTENCE_START_TAG)
        else:
            features.append("prev-NER-<%s>" % prev_tag)
        return features


class FeaturePerceptronTwo(StructuredPerceptron):
    """
    @description:
        Features
        ========
        - NER -1
        - WORD -2, -1, 0, 1, 2
        - POS -2, -1, 0, 1, 2
    """

    def _create_gold_features(self, sentence, index):
        features = []
        length = sentence.length()

        if index == length:  # end of sentence </s>
            features.append("prev-NER-%s" % sentence.words[-1].NER)                         # Prev NER
            features.append("current-POS-<%s>" % self.SENTENCE_END_TAG)                     # Current POS tag
            features.append("current-WORD-<%s>" % self.SENTENCE_END_TAG)                    # Current word
            #features.append("current(%d)-WORD-<%s>" % (1, self.SENTENCE_END_TAG*2))         # Current +1 word
            #features.append("current(%d)-WORD-<%s>" % (2, self.SENTENCE_END_TAG*3))         # Current +2 word
            #features.append("current(%d)-POS-<%s>" % (1, self.SENTENCE_END_TAG*2))          # Current +1 POS
            #features.append("current(%d)-POS-<%s>" % (2, self.SENTENCE_END_TAG*3))          # Current +2 POS
            features.append("current(%d)-WORD-<%s>" % (-1, sentence.get(index-1).name))     # Current -1 word
            features.append("current(%d)-POS-<%s>" % (-1, sentence.get(index-1).POS))       # Current -1 POS
            if index - 2 >= 0:  # Checking 2 words back
                features.append("current(%d)-WORD-<%s>" % (-2, sentence.get(index-2).name)) # Current -2 word
                features.append("current(%d)-POS-<%s>" % (-2, sentence.get(index-2).POS))   # Current -2 POS
            else:  # Start of sentence 2 words back
                features.append("current(%d)-WORD-<%s>" % (-2, self.SENTENCE_START_TAG))    # Current -2 word
                features.append("current(%d)-POS-<%s>" % (-2, self.SENTENCE_START_TAG))     # Current -2 POS
        elif index == 0:  # first word in sentence
            features.append("prev-NER-%s" % self.SENTENCE_START_TAG)                        # Prev NER
            features.append("current-POS-<%s>" % sentence.get(index).POS)                   # Current POS tag
            features.append("current-WORD-<%s>" % sentence.get(index).name)                 # Current word
            #features.append("current(%d)-WORD-<%s>" % (-1, self.SENTENCE_START_TAG*2))      # Current -1 word
            #features.append("current(%d)-WORD-<%s>" % (-2, self.SENTENCE_START_TAG*3))      # Current -2 word
            #features.append("current(%d)-POS-<%s>" % (-1, self.SENTENCE_START_TAG*2))       # Current -1 POS
            #features.append("current(%d)-POS-<%s>" % (-2, self.SENTENCE_START_TAG*3))       # Current -2 POS
            try:
                features.append("current(%d)-WORD-<%s>" % (1, sentence.get(index+1).name))  # Current +1 word
            except:
                features.append("current(%d)-WORD-<%s>" % (1, self.SENTENCE_END_TAG))       # Current +1 word
            try:
                features.append("current(%d)-WORD-<%s>" % (2, sentence.get(index+2).name))  # Current +2 word
            except:
                pass #features.append("current(%d)-WORD-<%s>" % (1, self.SENTENCE_END_TAG*2))     # Current +2 word
            try:
                features.append("current(%d)-POS-<%s>" % (1, sentence.get(index+1).POS))    # Current +1 POS
            except:
                features.append("current(%d)-POS-<%s>" % (1, self.SENTENCE_END_TAG))        # Current +1 POS
            try:
                features.append("current(%d)-POS-<%s>" % (2, sentence.get(index+2).POS))    # Current +2 POS
            except:
                pass # features.append("current(%d)-POS-<%s>" % (1, self.SENTENCE_END_TAG*2))      # Current +2 POS
        else:  # 2nd to nth word
            features.append("prev-NER-%s" % sentence.words[index-1].NER)                    # Prev NER
            features.append("current-POS-<%s>" % sentence.get(index).POS)                   # Current POS tag
            features.append("current-WORD-<%s>" % sentence.get(index).name)                 # Current word
            features.append("current(%d)-WORD-<%s>" % (-1, sentence.get(index-1).name))     # Current -1 word
            features.append("current(%d)-POS-<%s>" % (-1, sentence.get(index-1).POS))       # Current -1 POS
            if index - 2 >= 0:  # Checking 2 words back
                features.append("current(%d)-WORD-<%s>" % (-2, sentence.get(index-2).name)) # Current -2 word
                features.append("current(%d)-POS-<%s>" % (-2, sentence.get(index-2).POS))   # Current -2 POS
            else:  # Start of sentence 2 words back
                features.append("current(%d)-WORD-<%s>" % (-2, self.SENTENCE_START_TAG))    # Current -2 word
                features.append("current(%d)-POS-<%s>" % (-2, self.SENTENCE_START_TAG))     # Current -2 POS
            try:
                features.append("current(%d)-WORD-<%s>" % (1, sentence.get(index+1).name))  # Current +1 word
            except:
                features.append("current(%d)-WORD-<%s>" % (1, self.SENTENCE_END_TAG))       # Current +1 word
            try:
                features.append("current(%d)-WORD-<%s>" % (2, sentence.get(index+2).name))  # Current +2 word
            except:
                pass #features.append("current(%d)-WORD-<%s>" % (1, self.SENTENCE_END_TAG*2))     # Current +2 word
            try:
                features.append("current(%d)-POS-<%s>" % (1, sentence.get(index+1).POS))    # Current +1 POS
            except:
                features.append("current(%d)-POS-<%s>" % (1, self.SENTENCE_END_TAG))        # Current +1 POS
            try:
                features.append("current(%d)-POS-<%s>" % (2, sentence.get(index+2).POS))    # Current +2 POS
            except:
                pass #features.append("current(%d)-POS-<%s>" % (1, self.SENTENCE_END_TAG*2))      # Current +2 POS

        return features


    def _create_output_features(self, sentence, prev_tag, index):
        features = []
        length = sentence.length()

        if index == length:  # end of sentence </s>
            features.append("prev-NER-%s" % prev_tag)                                       # Prev NER
            features.append("current-POS-<%s>" % self.SENTENCE_END_TAG)                     # Current POS tag
            features.append("current-WORD-<%s>" % self.SENTENCE_END_TAG)                    # Current word
            #features.append("current(%d)-WORD-<%s>" % (1, self.SENTENCE_END_TAG*2))         # Current +1 word
            #features.append("current(%d)-WORD-<%s>" % (2, self.SENTENCE_END_TAG*3))         # Current +2 word
            #features.append("current(%d)-POS-<%s>" % (1, self.SENTENCE_END_TAG*2))          # Current +1 POS
            #features.append("current(%d)-POS-<%s>" % (2, self.SENTENCE_END_TAG*3))          # Current +2 POS
            features.append("current(%d)-WORD-<%s>" % (-1, sentence.get(index-1).name))     # Current -1 word
            features.append("current(%d)-POS-<%s>" % (-1, sentence.get(index-1).POS))       # Current -1 POS
            if index - 2 >= 0:  # Checking 2 words back
                features.append("current(%d)-WORD-<%s>" % (-2, sentence.get(index-2).name)) # Current -2 word
                features.append("current(%d)-POS-<%s>" % (-2, sentence.get(index-2).POS))   # Current -2 POS
            else:  # Start of sentence 2 words back
                features.append("current(%d)-WORD-<%s>" % (-2, self.SENTENCE_START_TAG))    # Current -2 word
                features.append("current(%d)-POS-<%s>" % (-2, self.SENTENCE_START_TAG))     # Current -2 POS
        elif index == 0:  # first word in sentence
            features.append("prev-NER-%s" % self.SENTENCE_START_TAG)                        # Prev NER
            features.append("current-POS-<%s>" % sentence.get(index).POS)                   # Current POS tag
            features.append("current-WORD-<%s>" % sentence.get(index).name)                 # Current word
            #features.append("current(%d)-WORD-<%s>" % (-1, self.SENTENCE_START_TAG*2))      # Current -1 word
            #features.append("current(%d)-WORD-<%s>" % (-2, self.SENTENCE_START_TAG*3))      # Current -2 word
            #features.append("current(%d)-POS-<%s>" % (-1, self.SENTENCE_START_TAG*2))       # Current -1 POS
            #features.append("current(%d)-POS-<%s>" % (-2, self.SENTENCE_START_TAG*3))       # Current -2 POS
            try:
                features.append("current(%d)-WORD-<%s>" % (1, sentence.get(index+1).name))  # Current +1 word
            except:
                features.append("current(%d)-WORD-<%s>" % (1, self.SENTENCE_END_TAG))       # Current +1 word
            try:
                features.append("current(%d)-WORD-<%s>" % (2, sentence.get(index+2).name))  # Current +2 word
            except:
                pass #features.append("current(%d)-WORD-<%s>" % (1, self.SENTENCE_END_TAG*2))     # Current +2 word
            try:
                features.append("current(%d)-POS-<%s>" % (1, sentence.get(index+1).POS))    # Current +1 POS
            except:
                features.append("current(%d)-POS-<%s>" % (1, self.SENTENCE_END_TAG))        # Current +1 POS
            try:
                features.append("current(%d)-POS-<%s>" % (2, sentence.get(index+2).POS))    # Current +2 POS
            except:
                pass #features.append("current(%d)-POS-<%s>" % (1, self.SENTENCE_END_TAG*2))      # Current +2 POS
        else:  # 2nd to nth word
            features.append("prev-NER-%s" % self.SENTENCE_START_TAG)                        # Prev NER
            features.append("current-POS-<%s>" % sentence.get(index).POS)                   # Current POS tag
            features.append("current-WORD-<%s>" % sentence.get(index).name)                 # Current word
            features.append("current(%d)-WORD-<%s>" % (-1, sentence.get(index-1).name))     # Current -1 word
            features.append("current(%d)-POS-<%s>" % (-1, sentence.get(index-1).POS))       # Current -1 POS
            if index - 2 >= 0:  # Checking 2 words back
                features.append("current(%d)-WORD-<%s>" % (-2, sentence.get(index-2).name)) # Current -2 word
                features.append("current(%d)-POS-<%s>" % (-2, sentence.get(index-2).POS))   # Current -2 POS
            else:  # Start of sentence 2 words back
                features.append("current(%d)-WORD-<%s>" % (-2, self.SENTENCE_START_TAG))    # Current -2 word
                features.append("current(%d)-POS-<%s>" % (-2, self.SENTENCE_START_TAG))     # Current -2 POS
            try:
                features.append("current(%d)-WORD-<%s>" % (1, sentence.get(index+1).name))  # Current +1 word
            except:
                features.append("current(%d)-WORD-<%s>" % (1, self.SENTENCE_END_TAG))       # Current +1 word
            try:
                features.append("current(%d)-WORD-<%s>" % (2, sentence.get(index+2).name))  # Current +2 word
            except:
                pass #features.append("current(%d)-WORD-<%s>" % (1, self.SENTENCE_END_TAG*2))     # Current +2 word
            try:
                features.append("current(%d)-POS-<%s>" % (1, sentence.get(index+1).POS))    # Current +1 POS
            except:
                features.append("current(%d)-POS-<%s>" % (1, self.SENTENCE_END_TAG))        # Current +1 POS
            try:
                features.append("current(%d)-POS-<%s>" % (2, sentence.get(index+2).POS))    # Current +2 POS
            except:
                pass #features.append("current(%d)-POS-<%s>" % (1, self.SENTENCE_END_TAG*2))      # Current +2 POS

        return features
