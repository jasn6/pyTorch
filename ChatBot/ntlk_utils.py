import nltk
nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

test = "How long does shipping take?"
print(test)
test = tokenize(test)
print(test)

test2 = ["organize","organizing","organizes"]
print(test2)
test2 = [stem(word) for word in test2]
print(test2)