# -*- coding: utf-8 -*-

import pandas as pd
import requests
import bs4
import os
# import os.path
import re
import time
t = time.time()

from nltk.tokenize import sent_tokenize, word_tokenize

# get html source of a patent
def get_pathtml(pat):
    # tar_soup = []
    # for pat in patno:
    filename = "./pattxt/" + pat + ".txt"
    if os.path.isfile(filename):
        print("Read patent", pat, "from disk...")
        with open(filename, 'r') as file:
            pattxt = file.read()
            pat_soup = bs4.BeautifulSoup(pattxt, 'lxml') #'html.parser'
            return pat_soup
    else:
        url = "https://patents.google.com/patent/" + pat +"/en"
        resp = requests.get(url)
        print("Getting ", pat, "from Google Patents...")
        with open(filename, 'w') as file:
            file.write(resp.text)
        pat_soup = bs4.BeautifulSoup(resp.text, 'lxml') #'html.parser'
        return pat_soup

# get patent's text from html source of patent file
def pattxt(pno):
    p_html = get_pathtml(pno)
    desc = p_html.find(attrs={'itemprop': 'description'}).get_text()
    claim = p_html.find(attrs={'itemprop': 'claims'}).get_text()
    ptxt = desc + claim
    return ptxt

dirname = 'pattxt'

fname = os.listdir(dirname)
file = [f[:-4] for f in fname[2:]]
pat_txt = [pattxt(f) for f in file]
pat_line = [sent_tokenize(p) for p in pat_txt]
pat_cleantxt = [[re.sub('[^A-Za-z]+', ' ', string).lower() for string in line] for line in pat_line]
pat_token = [[word_tokenize(pline) for pline in p_line] for p_line in pat_cleantxt]

pat_sentences = []
for p in pat_token:
    for l in p:
        print(l)
        if len(l) != 0:
            pat_sentences.append(l)

with open('./tf_w2v/pat_sentences.txt', mode='w', encoding='utf-8') as f:
    print(f)
    for lines in pat_sentences:
        f.write(' '.join(str(line) for line in lines))
        f.write('\n')

print("Time to make source txt from patent html files:", (time.time()-t))
