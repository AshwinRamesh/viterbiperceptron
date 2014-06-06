from viterbiperceptron.perceptron import FeaturePerceptronOne, FeaturePerceptronTwo, Sentence
from viterbiperceptron.data_parser import read_file, load_perceptron_from_file, write_output_to_file
import json

# Set of all perceptron subclasses to run
suites = {
            1: FeaturePerceptronOne,
            2: FeaturePerceptronTwo
        }

# Defines which suites to actually run
run_suites = [1]


# Read the file into documents
docs = read_file("../data/conll03/eng.train")
training_data = []
for doc in docs:
    for sentence in doc:
        training_data.append(Sentence.create_sentence_from_list(sentence))

docs = read_file("../data/conll03/eng.testa")
testa_data = []
for doc in docs:
    for sentence in doc:
        testa_data.append(Sentence.create_sentence_from_list(sentence))

docs = read_file("../data/conll03/eng.testb")
testb_data = []
for doc in docs:
    for sentence in doc:
        testb_data.append(Sentence.create_sentence_from_list(sentence))

print "Processed all data.. iterating now.."

for suite in run_suites:
    perceptron = suites[suite](training_data=training_data, iterations=10)

    print "Beginning Training"
    perceptron.train()
    f = file("eng.train.out.suite%d" % suite, "w+")
    f.write(json.dumps(perceptron.weights))
    f.close()
    print "Processed Training Data"

    for sentence in testa_data:
        perceptron.classify(sentence)
    write_output_to_file(testa_data, "eng.testa.out.suite%d" % suite)

    for sentence in testb_data:
        perceptron.classify(sentence)
    write_output_to_file(testa_data, "eng.testb.out.suite%d" % suite)

print "Done testing."