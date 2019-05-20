# Essential Libraries
from os import path
import requests
from collections import Counter
import wordcloud
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
from wordcloud import WordCloud


def wordCloud():
    # reads the words from a txt fileC:\Users\ebin1\Desktop\new11
    tags = open("C:/Users/ebin1/Desktop/tags.txt", "r").read().split("\n")

    # reads the words from a csv file(used to display the frequency of words)
    dataset = pd.read_csv("C:/Users/ebin1/Desktop/Question/Tags.csv")
    # display frequency of each word
    df_new = dataset[dataset['Tags'].notnull()]
    x=(pd.Series(np.concatenate([word.split() for word in df_new['Tags']])).value_counts())

    y=x[0:20]
    print(y)
    y.to_csv('C:/Users/ebin1/Desktop/Question/ds.csv')

    # dictionary with the frequecy of each word
    tagFreq = Counter()
    for word in tags:
        tagFreq[word] += 1
    # cloud mask from local machine
    #mask = np.array(Image.open(path.join("path/to/mask/file", "mask_file_name.png")))
    # cloud mask from an online resource
    mask = np.array(Image.open(requests.get('http://www.clker.com/cliparts/O/i/x/Y/q/P/yellow-house-hi.png', stream=True).raw))
    wordcloud = WordCloud(width=1000, height=900, background_color='white', relative_scaling=.8
                          ).generate_from_frequencies(tagFreq)
    # save the word cloud as a png file
    wordcloud.to_file("myfile.png")
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()


wordCloud()