import pkuseg
import pandas as pd
import pickle
import numpy as np
from collections import Counter


# load data
df = pd.read_csv("../chinese_news.csv")
df = df[df.content.notnull()].reset_index()
df.headline.apply(lambda x: x.replace("\n", "")).to_csv("headlines.csv",
                                                        index=False)
df.content.apply(lambda x: x.replace("\n", "")).to_csv("contents.csv",
                                                       index=False)
docs = []
raw_text = []

# word segmentation
seg = pkuseg.pkuseg()
for i in range(df.shape[0]):
    print(i)
    curdoc = seg.cut(df.content[i])
    docs.append(curdoc)
    raw_text += seg.cut(df.content[i])

# with open(r"docs.pickle", "wb") as f:
#     pickle.dump(docs, f)

# with open(r"docs.pickle", "rb") as f:
#     a = pickle.load(f)

# remove stop words
stopwords = []
with open("stopwords_cn.txt") as f:
    stopwords = f.read()

clean_text = []
for w in raw_text:
    if w not in stopwords:
        clean_text.append(w)

# generate dictionary
counter = Counter(clean_text)
wordlist = counter.keys()
with open("dictionary.txt", "w+") as f:
    for word in wordlist:
        f.write(word + "\n")


dict_file = "dictionary.txt"
token_list = open(dict_file).read().split("\n")
token2id_dic = {token: id for id, token
                in enumerate(token_list)}

def doc2id(doc, dic):
    # convert a list of words into a list of token_id
    token_ids = [dic[word] for word in doc if word in dic]
    result = np.zeros(len(token_ids), dtype=np.int32)
    for i in range(len(token_ids)):
        result[i] = token_ids[i]
    return result


def doc2sentences(doc):
    # convert a list of words (including "。") into a list of sentences, where
    # each sentence is a list of words
    result = []
    cur_sentence = []
    for word in doc:
        if word not in ["。", "?", "！", "；"]:
            cur_sentence.append(word)
        else:
            result.append(cur_sentence)
            cur_sentence = []

    # result.append(cur_sentence)
    return result

def doc2sentokens(doc, dic):
    sentences = doc2sentences(doc)
    result = [doc2id(sentence, dic) for sentence in sentences]
    return result

token_docs = [doc2id(doc, token2id_dic) for doc in docs]
sentence_tokens = [doc2sentokens(doc, token2id_dic) for doc in docs]
