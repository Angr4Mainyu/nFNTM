import numpy as np
import matplotlib.pyplot as plt
import os
from gensim.models.coherencemodel import CoherenceModel
import requests
from seaborn.utils import relative_luminance
from tomorrow3 import threads
from sklearn.decomposition import PCA
from retrying import retry
from transformers import BertTokenizer


def build_bert_embedding(embedding_fn, vocab, data_dir):
    print(f"building bert embedding matrix for dit {len(vocab)}")
    tokenize = BertTokenizer.from_pretrained('bert-base-uncased')
    embedding_mat_fn = os.path.join(data_dir, f"bert_emb_{len(vocab)}.npy")

    if os.path.exists(embedding_mat_fn):
        embedding_mat = np.load(embedding_mat_fn)
        return embedding_mat

    index = np.array(
        tokenize.encode(list(vocab.token2id.keys()), add_special_tokens=False))
    bert_mat = np.load(embedding_fn)
    bert_emb = bert_mat[index]
    np.save(embedding_mat_fn, bert_emb)
    return bert_emb


def build_embedding(embedding_fn, vocab, data_dir):
    print(f"building embedding matrix for dict {len(vocab)} if need...")
    embedding_mat_fn = os.path.join(data_dir,
                                    f"embedding_mat_{len(vocab)}.npy")

    if os.path.exists(embedding_mat_fn):
        embedding_mat = np.load(embedding_mat_fn)
        return embedding_mat

    embedding_index = {}
    with open(embedding_fn, encoding='UTF-8') as fin:
        first_line = True
        l_id = 0
        for line in fin:
            if l_id % 100000 == 0:
                print("loaded %d words embedding..." % l_id)
            if ("glove" not in embedding_fn) and first_line:
                first_line = False
                continue
            line = line.rstrip()
            values = line.split(' ')
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embedding_index[word] = coefs
            l_id += 1

    embedding_dim = len(list(embedding_index.values())[0])
    embedding_mat = np.zeros(
        (len(vocab) + 1, embedding_dim))  # -1 is for padding
    for i, word in vocab.items():
        embedding_vec = embedding_index.get(word)
        if embedding_vec is not None:
            embedding_mat[i] = embedding_vec
    np.save(embedding_mat_fn, embedding_mat)
    return embedding_mat


def build_embedding_PCA(embedding_fn, vocab, data_dir):
    print(f"building embedding matrix to PCA for dict {len(vocab)} if need...")
    embedding_PCA_fn = os.path.join(data_dir,
                                    f"embedding_mat_PCA_{len(vocab)}.npy")

    if os.path.exists(embedding_PCA_fn):
        embedding_PCA_mat = np.load(embedding_PCA_fn)
        return embedding_PCA_mat

    embedding_mat = build_embedding(embedding_fn, vocab, data_dir)
    pca = PCA(n_components=2)
    embedding_PCA_mat = pca.fit_transform(embedding_mat)
    np.save(embedding_PCA_fn, embedding_PCA_mat)
    return embedding_PCA_mat


def compute_coherence(doc_word, topic_word, ave_N=10):
    topic_size, word_size = np.shape(topic_word)
    doc_size = np.shape(doc_word)[0]
    
    coherence = []
    for N in [5, 10, 15]:
        topic_list = []
        for topic_idx in range(topic_size):
            top_word_idx = np.argpartition(topic_word[topic_idx, :], -N)[-N:]
            topic_list.append(top_word_idx)

        sum_coherence_score = 0.0
        for i in range(topic_size):
            word_array = topic_list[i]
            sum_score = 0.0
            for n in range(N):
                flag_n = doc_word[:, word_array[n]] > 0
                p_n = np.sum(flag_n) / doc_size
                for l in range(n + 1, N):
                    flag_l = doc_word[:, word_array[l]] > 0
                    p_l = np.sum(flag_l)
                    p_nl = np.sum(flag_n * flag_l)
                    if p_n * p_l * p_nl > 0:
                        p_l = p_l / doc_size
                        p_nl = p_nl / doc_size
                        sum_score += np.log(p_nl / (p_l * p_n)) / -np.log(p_nl)
            sum_coherence_score += sum_score * (2 / (N * N - N))
        sum_coherence_score = sum_coherence_score / topic_size
        coherence.append(sum_coherence_score)
    return np.mean(coherence)


def evaluate_coherence(topic_words, texts, vocab):
    coherence = {}
    methods = ["c_v", "c_npmi", "c_uci", "u_mass"]
    for method in methods:
        coherence[method] = CoherenceModel(topics=topic_words,
                                           texts=texts,
                                           dictionary=vocab,
                                           coherence=method).get_coherence()
    return coherence


def evaluate_topic_diversity(topic_words):
    '''topic_words is in the form of [[w11,w12,...],[w21,w22,...]]'''
    vocab = set(sum(topic_words, []))
    total = sum(topic_words, [])
    return len(vocab) / len(total)


def compute_topic_specialization(topic_dist, corpus):
    curpus_vector = corpus.sum(axis=0)
    curpus_vector = curpus_vector / np.linalg.norm(curpus_vector)
    for i in range(topic_dist.shape[0]):
        topic_dist[i] = topic_dist[i] / np.linalg.norm(topic_dist[i])
    topics_spec = 1 - topic_dist.dot(curpus_vector)
    depth_spec = np.mean(topics_spec)
    return depth_spec


def compute_overlap(level1, level2):
    sum_overlap_score = 0.0
    for N in [5, 10, 15]:
        word_idx1 = np.argpartition(level1, -N)[-N:]
        word_idx2 = np.argpartition(level2, -N)[-N:]
        c = 0
        for n in word_idx1:
            if n in word_idx2:
                c += 1
        sum_overlap_score += c / N
    return sum_overlap_score / 3


def compute_clnpmi(level1, level2, doc_word):

    sum_coherence_score = 0.0
    c = 0

    for N in [5,10,15]:
        word_idx1 = np.argpartition(level1, -N)[-N:]
        word_idx2 = np.argpartition(level2, -N)[-N:]
        
        sum_score = 0.0
        set1 = set(word_idx1)
        set2 = set(word_idx2)
        inter = set1.intersection(set2)
        word_idx1 = list(set1.difference(inter))
        word_idx2 = list(set2.difference(inter))

        for n in range(len(word_idx1)):
            flag_n = doc_word[:, word_idx1[n]] > 0
            p_n = np.sum(flag_n) / len(doc_word)
            for l in range(len(word_idx2)):
                flag_l = doc_word[:, word_idx2[l]] > 0
                p_l = np.sum(flag_l)
                p_nl = np.sum(flag_n * flag_l)
                if p_nl == len(doc_word):
                    sum_score += 1
                elif p_n * p_l * p_nl > 0:
                    p_l = p_l / len(doc_word)
                    p_nl = p_nl / len(doc_word)
                    p_nl += 1e-10
                    sum_score += np.log(p_nl / (p_l * p_n)) / -np.log(p_nl)
                c += 1
        if c > 0:
            sum_score /= c
        else:
            sum_score = 0
        sum_coherence_score += sum_score
    return sum_coherence_score / 3

def compute_hierarchical_affinity(topic_dist, relation):
    child_ha, unchild_ha = [], []
    topic_dist = topic_dist / np.linalg.norm(topic_dist, axis=1, keepdims=True)

    for child in relation.keys():
        for parent in relation.values():
            if parent == child:
                continue
            ha = topic_dist[child].dot(topic_dist[parent])
            if relation[child] == parent:
                child_ha.append(ha)
            else:
                unchild_ha.append(ha)
            
    return np.mean(child_ha), np.mean(unchild_ha)


def print_topic_word(topic_word, vocab, N):
    topic_size, word_size = np.shape(topic_word)

    top_word_idx = np.argsort(topic_word, axis=1)
    top_word_N = top_word_idx[:, -N:]

    for k, top_word_k in enumerate(top_word_N[:, ::-1]):
        top_words = [vocab[id] for id in top_word_k]
        print(f'Topic {k}:{top_words}')


@retry(stop_max_attempt_number=10)
def get_palmetto(topic, url):
    res = requests.get(url, {'words': ' '.join(topic)})
    coh = np.float(res.text)
    return coh


def palmetto_coherence(topic_words, coherence_type="ca"):
    url = f'http://172.18.166.73:7777/palmetto-webapp/service/{coherence_type}'

    coherence = [get_palmetto(topic, url) for topic in topic_words]
    return coherence


def palmetto_evaluate(topic_words):
    scores = {}
    methods = ["ca", "cp", "cv", "npmi", "uci", "umass"]
    for method in methods:
        scores[method] = np.mean(
            palmetto_coherence(topic_words, coherence_type=method))
        print(method, scores[method])
    return scores


def build_level(adj_matrix):
    adj_matrix[adj_matrix < 0.3] = 0
    adj_matrix[adj_matrix > 0.3] = 1

    topic_num = adj_matrix.shape[0]
    roots = np.where((adj_matrix.sum(axis=1) == 0) | (np.diag(adj_matrix) > 0 ))[0]
    adj_matrix = adj_matrix - np.diag(np.diag(adj_matrix))
    
    trees = {}
    for root in roots:
        level = {}
        cur = np.zeros(topic_num)
        l = 0
        cur[root] = 1
        while cur.sum() != 0:
            cur_node = np.where(cur > 0)
            level[l] = cur_node[0]
            cur = cur @ adj_matrix.T
            l += 1
        trees[root] = level
    relation = np.where(adj_matrix == 1)
    relation = dict(zip(relation[0], relation[1]))
    return trees, relation