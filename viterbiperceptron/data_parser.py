import os
"""
@description: Additonal functions that do data file processing
"""

def read_file(input_file):
    """
    @description: Reads one of the input data files
    :return
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

    DOC_START_STRING = "-DOCSTART- -X- -X- O"

    # Input file validation
    if not os.path.isfile(input_file):
        raise Exception("Input file does not exist")

    docs = []
    document = []
    sentence = []

    # Read file into memory
    with open(input_file) as f:
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

    return docs


def write_output_to_file(output, output_file):
    """
    @description: Writes the output of a NER tagger to a output file for statistical analysis
    :return:
        True|False
    """
    pass