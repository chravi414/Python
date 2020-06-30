import urllib.request
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize, wordpunct_tokenize
from bs4 import BeautifulSoup
# Reading from URL
wikiurl = "https://en.wikipedia.org/wiki/Google"
openurl = urllib.request.urlopen(wikiurl)
soup = BeautifulSoup(openurl.read(), "lxml")

# get text
text = soup.body.get_text()

# break into lines and remove leading and trailing space on each
lines = (line.strip() for line in text.splitlines())
# break multi-headlines into a line each
chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
# drop blank lines
text = ' '.join(chunk for chunk in chunks if chunk)

# Saving to a Text File
with open('Input', 'w') as text_file:
    text_file.write(str(text.encode("utf-8")))
