from viterbiperceptron.perceptron import FeaturePerceptronOne as Perceptron
from viterbiperceptron.data_parser import read_file, load_perceptron_from_file, write_output_to_file
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

    f = file("out.eng.train2", "w+")
    f.write(json.dumps(perceptron.weights))
    f.close()
    print "done"
elif process == 1:
    training_data = []
    perceptron = load_perceptron_from_file(Perceptron, "out.eng.train2")
    assert isinstance(perceptron, Perceptron)
    docs = read_file("/Users/Ash/Documents/University/comp5046/assignments/assignment3/data/conll03/eng.testa")
    for doc in docs:
        training_data = training_data + doc
    for i in xrange(0, len(training_data)):
        out_tags = perceptron.classify(training_data[i])
        print out_tags
        for j in xrange(0, len(training_data[i])):
            training_data[i][j].append(out_tags[j])
    write_output_to_file(training_data, "eng.testa.classified")
