from viterbiperceptron.perceptron import FeaturePerceptronOne as Perceptron
from viterbiperceptron.data_parser import read_file
import json


# Read the file into documents
docs = read_file("/Users/Ash/Documents/University/comp5046/assignments/assignment3/data/conll03/eng.train")
process = 1
if process == 0:
    # Convert documents into a large list of sentences == training data
    training_data = []
    for doc in docs:
        training_data = training_data + doc
    print len(training_data)
    perceptron = Perceptron(training_data=training_data, iterations=3)
    perceptron.train()

    f = file("out.eng.train", "w+")
    f.write(json.dumps(perceptron.weights))
    f.close()
    print "done"
elif process == 1:
    perceptron = Perceptron.load_from_json("out.eng.train")
    print perceptron.weights