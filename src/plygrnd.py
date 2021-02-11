import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.cm as cm
from sklearn.svm import SVC
import sklearn
import tqdm
import arff
import re

import os
from multiprocessing.pool import Pool
import shutil

from tools.Logger import Logger
from tools.paths import itemized as paths
from tools.params import itemized as params
from sklearn import svm

import nltk
from nltk.corpus import stopwords  # Import the stop word list
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier


def clear(s):
    return re.sub(r'\W+', '', s)


# a = [['piper calling you to join him'],
#      ['losing sleep dreaming'],
#      ['seen'],
#      ['shield']
#      ]

s0 = '''Your head is humming and it won't go, in case you don't know
The piper's calling you to join him
Dear lady, can you hear the wind blow? And did you know
Your stairway lies on the whispering wind?'''

s1 = '''losing sleep
Dreaming about the things that we could be
But baby Ive been, Ive been praying hard
Said no more counting dollars
Well be counting stars
Yeah well be counting stars'''
s2 = 'seen'
s3 = 'shield'
a = [s0, s1, s2, s3]

from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError


def url_ok(url):
    req = Request(url)
    try:
        response = urlopen(req)
    except HTTPError as e:
        # print('The server couldn\'t fulfill the request.')
        # print('Error code: ', e.code)
        return 1
    except URLError as e:
        # print('We failed to reach a server.')
        # print('Reason: ', e.reason)
        return 2
    else:
        return 0


test_url = r'https://www.youtube.com/'
is_hit = url_ok(test_url)
print(f"Test: {is_hit}")
for starter in ['https', 'http']:
    for a1 in a[0].split():
        for a2 in a[1].split():
            for a3 in a[2].split():
                for a4 in a[3].split():
                    url = fr"{starter}://www.{clear(a1)}{clear(a2)}un{clear(a3)}{clear(a4)}.com/"
                    is_hit = url_ok(url)
                    if is_hit != 2:
                        print(f"{url} is a HIT!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    else:
                        pass
                        # print(f"{url}\tmissed.")
