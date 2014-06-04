"""
@author: Ashwin Ramesh
"""

from perceptron import AveragedPerceptron as Perceptron


class Viterbi(object):
    """
    @description: This is the viterbi algorithm implementation
    """

    tags = []
    start_sentence_identifier = "<START-SENTENCE>"

    def __init__(self, perceptron):
        self.perceptron = perceptron
        assert isinstance(self.perceptron, Perceptron)

    def process_viterbi(self, sentence, training=False):  # TODO - take training into account
        """
        @description: Processes an input sentence with viterbi.
        @args:
            sentence (list) - list of words with features (dict)
            training (bool) - relates to whether to process the sentence as a training|test against the perceptron
        """

        # Initialise local variables
        sentence = [self.start_sentence_identifier] + sentence  # Add start of string tag to sentence
        viterbi_history = [{}]
        viterbi_path = {}

        # Initialise Start Base case
        for tag in self.tags:
            viterbi_history[0][tag] = 0  # TODO - do I initialise the start as 0?
            viterbi_path[tag] = [tag]

        # Run Viterbi for t > 0
        for i in xrange(1, len(sentence)):
            viterbi_history.append({})
            temp_path = {}

            for tag in self.tags:
                (prob, state) = self._compute_probability(sentence, i, tag, viterbi_history, viterbi_path)
                viterbi_history[i][tag] = prob
                temp_path[tag] = viterbi_path[state] + [tag]

            viterbi_path = temp_path

        # return the best path
        return self._backtrack(sentence, viterbi_history, viterbi_path)

    def _backtrack(self, sentence, history, path):
        n = len(sentence) - 1
        (prob, state) = max((history[n][tag]) for tag in self.tags)
        return prob, path[state]

    def _compute_probability(self, sentence, index, tag, history, path, training=False):
        # TODO - THIS IS WHERE WE DO THE PERCEPTRON STUFF.
        return 0, 0





class NerViterbi(Viterbi):
    """
    @description: Viterbi implementation for Named Entity Recognition (NER)
    """

    tags = ["B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-MISC", "I-MISC", "O"]



