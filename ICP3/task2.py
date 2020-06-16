
from bs4 import BeautifulSoup
from urllib import request
import os

url = "https://en.wikipedia.org/wiki/Deep_learning"
page = request.urlopen(url)
soup = BeautifulSoup(page, "html.parser")
heading = soup.find(id="firstHeading")
print(heading.string)
links = soup.find_all('a')
link_file = open("links.txt", "a")
for link in links:
    url = str(link.get("href")) + "\n"
    link_file.write(url)
print('Completed writing links to file')
link_file.close()
