from nltk import wordpunct_tokenize, pos_tag, ne_chunk
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
from nltk import pos_tag, wordpunct_tokenize, ne_chunk
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

with open('Input', 'r') as text_file:
    read_data = text_file.read()


# Tokenization
stokens = nltk.sent_tokenize(read_data)
wtokens = nltk.word_tokenize(read_data)

print("======== Tokenization Output ============= \n")

print("No.of Sentences: ", len(stokens))
print("No.of Words: ", len(wtokens))

print("===================== \n")

# POS

print("==========  POS Output =========== \n")

print("POS tags", pos_tag(wtokens))

print("===================== \n")

# Stemming
pStememr = PorterStemmer()

print("=========== PorterStemmer output ========== \n")

for word in wtokens:
    print(pStememr.stem(word), end=' ')

print("===================== \n")

lStemmaer = LancasterStemmer()
print("=========== LancasterStemmer output ========== \n")

for word in wtokens:
    print(pStememr.stem(word), end=' ')

print("===================== \n")


sStemmer = SnowballStemmer('english')

print("=========== SnowballStemmer output ========== \n")

for word in wtokens:
    print(pStememr.stem(word), end=' ')

print("===================== \n")


# Lemmatization


lemmatizer = WordNetLemmatizer()

print("========= Lemmatizer Output ============ \n")
for word in wtokens:
    print(lemmatizer.lemmatize(word), end=' ')
print("===================== \n")


# Trigram

trigrams = list(ngrams(wtokens, 3))
print("========= Trigram Output ============ \n")
print(trigrams)
print("===================== \n")


# Named Entity Recognition

chunks = ne_chunk(pos_tag(wordpunct_tokenize(read_data)))
for chunk in chunks:
    print(chunk)
