"""
@description: Parses the data file provided as input into a list of format:
    List of Documents (list)
        List of Sentences (list)
            List of Words (list)
                [0]: Word (str)
                [1]: POS Tag (str)
                [2]: Syntactic Chunk Tag (str)
                [3]: Gold Standard NER Tag (str)

    The german data will have an extra index in the words level for "lemma base" (index 2)
    Output: .JSON encoded string in a file
"""

import sys
import os.path
import json


def main():
    args = sys.argv
    DOC_START_STRING = "-DOCSTART- -X- -X- O"

    # Input param validation
    if len(args) != 3:
        print "Usage: parse_data.py <input_file> <output_file>"
        exit(1)
    if not os.path.isfile(args[1]):
        print "Input file does not exist"
        exit(1)

    print "Beginning parsing file"
    docs = []
    document = []
    sentence = []

    # Read file into memory
    with open(args[1]) as f:
        content = f.readlines()

    i = 0  # line number index
    while i < len(content):
        if content[i].strip() == DOC_START_STRING or i == len(content) - 1:  # Deal with new documents
            if len(document) != 0:  # Sentences exist
                docs.append(list(document))
                document = []
            i += 2  # Jump 2 indexes (new line + next word)

        elif content[i] == "\n":  # Line is new line --> end of previous sentence
            if len(sentence) > 0:  # Only append if sentence + words exist
                document.append(list(sentence))
                sentence = []
            i += 1

        else:  # Word line
            sentence.append(list(content[i].strip().split(" ")))  # split word line into 4 sections
            i += 1

    f = file(args[2], "w+")
    f.write(json.dumps(docs))
    f.close()
    print "Finished parsing file"

main()