from mylda import LDA
from mylda.sample import _sample
import numpy as np
import pickle

dict_file = "../preprocessing/dictionary.txt"
token_list = open(dict_file).read().split("\n")
token2id_dic = {token: id for id, token
                in enumerate(token_list)}

# lda_model = LDA(K=20, n_early_stop=20, dict_file=dict_file)
# token_docs = [gen_rand_tokens(50, np.random.randint(10, 20)) for i in range(30)]
# lda_model.fit(token_docs)


with open(r"../preprocessing/docs.pickle", "rb") as f:
    docs = pickle.load(f)


def doc2id(doc, dic):
    # convert a list of words into a list of token_id
    token_ids = [dic[word] for word in doc
                 if word in dic]
    return token_ids

token_docs = [doc2id(d, token2id_dic) for d in docs]

# model = LDA(K=20, n_early_stop=20, dict_file=dict_file)
# model.fit(token_docs)

# with open(r"m_k20.pickle", "wb") as f:
#     pickle.dump(lda_model, f)

with open(r"m_k20.pickle", "rb") as f:
    model = pickle.load(f)


def _doc_phi(corpus_phi, doc, alpha):
    K, T = corpus_phi.shape
    n = len(doc)
    z = np.random.randint(K, size=(n), dtype=np.int32)
    n_z = np.zeros(K, dtype=np.int32)
    for i in range(n):
        n_z[z[i]] += 1
    num_loops = 3
    for i in range(num_loops):
        _sample._predict(corpus_phi, doc, z, n_z, alpha)
    smooth = n_z + alpha
    return smooth / smooth.sum()
    # return (z, n_z)


def doc_phi(model, doc):
    # n_mk = model.n_mk
    return _doc_phi(model._phi(), doc, model.alpha)


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


def _phi_sentences(model, doc):
    sen_tokens = doc2sentokens(doc, model.dictionary._token2id)
    # phis = [doc_phi(model, sentence) for sentence in sen_tokens]
    n = len(sen_tokens)               # number of sentences
    result = np.zeros((n, model.K))
    for i in range(n):
        result[i] = doc_phi(model, sen_tokens[i])
    return result


def best_sentence(model, docs, doc_id):
    doc = docs[doc_id]
    sentences = doc2sentences(doc)
    sentences = [''.join(x) for x in sentences]
    phis = _phi_sentences(model, doc)
    scores = phis.dot(model.n_mk[doc_id])
    best = scores.argmax()
    return sentences[best]

if __name__ == "__main__":
    # select the docid to extract the corresponding summary
    doc_id = 1
    print(best_sentence(model, docs, doc_id))
