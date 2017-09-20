from nltk.tag.perceptron import PerceptronTagger
from nltk import word_tokenize

# Load and initialize Perceptron tagger
tagger = PerceptronTagger()


def posTagger(sentence):
    # [example] sentence: 'hi man how are you'
    print('Sentence: ', sentence)
    tokenizedSentence = word_tokenize(sentence, language='russian')
    # [example] tokenizedSentence: ['hi', 'man', 'how', 'are', 'you']
    print('Tokenized: ', tokenizedSentence)
    posTaggedSentence = tagger.tag(tokenizedSentence)
    # [example] posTaggedSentence: [('hi', 'NN'), ('man', 'NN'), ('how', 'WRB'), ('are', 'VBP'), ('you', 'PRP')]
    print('Tagged: ', posTaggedSentence)
    return posTaggedSentence


def posTagAndLabel(sentence):
    taggedSentence = posTagger(sentence)
    taggedSentenceJson = []
    for token, postag in taggedSentence:
        taggedSentenceJson.append([token, postag, "O"])

    # [example] posTaggedSentence: [['hi', 'NN', 'O'], ['man', 'NN', 'O'], ['how', 'WRB', 'O'], ['are', 'VBP', 'O'], ['you', 'PRP', 'O']]
    return taggedSentenceJson


def sentenceTokenize(sentences):
    tokenizedSentences = word_tokenize(sentences, language='russian')
    tokenizedSentencesPlainText = ""
    for t in tokenizedSentences:
        tokenizedSentencesPlainText += " " + t
    return tokenizedSentencesPlainText
