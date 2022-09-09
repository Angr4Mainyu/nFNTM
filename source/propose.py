import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import utils
from reader import TextReader
from sklearn import metrics
from torch.distributions import Beta, Independent, kl_divergence

import time
import os
import sys
from tqdm import tqdm

import yaml

np.random.seed(0)
torch.manual_seed(0)
torch.autograd.set_detect_anomaly(True)

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


class net(nn.Module):
    def __init__(self,
                 max_topic_num=None,
                 batch_size=None,
                 embed_num=200,
                 emb_mat=None,
                 vocab_num=None,
                 hidden_num=None,
                 prior_beta=None,
                 **kwargs):
        super(net, self).__init__()

        self.prior_beta = prior_beta
        self.max_topic_num = max_topic_num
        self.embed_num = embed_num

        if emb_mat == None:
            self.word_embed = nn.Parameter(torch.rand(vocab_num, embed_num))
            freeze = False
        else:
            print("Using pre-train word embedding")
            self.word_embed = nn.Parameter(emb_mat)
            freeze = True

        self.pz_x = Independent(
            Beta(
                torch.ones(batch_size, max_topic_num).to(device),
                torch.ones(batch_size, max_topic_num).to(device) *
                self.prior_beta), 1)

        # whether to train word embedding
        if freeze:
            self.word_embed.requires_grad = False

        self.topic_embed = nn.Parameter(torch.randn(max_topic_num, embed_num))

        self.encoder = nn.Sequential(nn.Linear(vocab_num, hidden_num),
                                     nn.Tanh())

        self.l_alpha = nn.Linear(hidden_num, max_topic_num)
        self.l_beta = nn.Linear(hidden_num, max_topic_num)

        self.Q_vec = nn.Parameter(torch.randn(embed_num, embed_num))
        self.K_vec = nn.Parameter(torch.randn(embed_num, embed_num))

        self.gamma = 1

        self.temperature = 5

    # GEM distribution
    def stick_break_process(self, alpha, beta):
        qz_x = Independent(Beta(torch.exp(alpha), torch.exp(beta)), 1)
        x = qz_x.rsample()

        v_mid1 = x.unsqueeze(-1).expand(-1, -1, self.max_topic_num)
        v_mid2 = torch.ones_like(v_mid1) - torch.diag_embed(
            torch.ones_like(x)) - torch.triu(v_mid1)
        stick_segment = -v_mid2.prod(dim=-2)

        return stick_segment, qz_x

    def get_topic_dist(self):
        try:
            return self.beta
        except:
            self.beta = torch.softmax(self.topic_embed @ self.word_embed.T,
                                      dim=1)
            return self.beta

    def infer(self, x):
        a, b = self.encode(x)
        sbp, qz_x = self.stick_break_process(a, b)
        return sbp, qz_x

    def encode(self, x):
        pi = self.encoder(x)

        return self.l_alpha(pi), self.l_beta(pi)

    def attention(self, sbp):
        a = self.topic_embed

        q, k = a @ self.Q_vec, a @ self.K_vec

        d = torch.Tensor([self.embed_num]).to(device)
        sqrt_d = torch.sqrt(d)

        self.update_adjacency_matrix(q, k, sqrt_d)

    def update_adjacency_matrix(self, q, k, sqrt_d):
        n, d = q.shape
        w = q @ k.T
        # w = w / sqrt_d
        w = w / d
        w = w / self.temperature
        w = torch.softmax(w, dim=1)

        self.adjacency_matrix = w

    def decode(self, sbp):

        self.beta = torch.softmax(self.topic_embed @ self.word_embed.T, dim=1)

        self.pi = sbp * self.gamma + sbp @ self.adjacency_matrix * (1 -
                                                                    self.gamma)

        theta_1 = sbp @ self.beta
        theta_2 = sbp @ self.adjacency_matrix @ self.beta

        self.theta = self.pi @ self.beta
        return self.theta, theta_1 - theta_2

    def forward(self, x):
        a, b = self.encode(x)

        sbp, qz_x = self.stick_break_process(a, b)
        self.attention(sbp)
        d, d_delta = self.decode(sbp)
        return d, d_delta, qz_x, self.pz_x, sbp, self.adjacency_matrix


class AMM(object):
    def __init__(self,
                 reader=None,
                 max_topic_num=200,
                 model_path=None,
                 emb_mat=None,
                 epochs=None,
                 batch_size=None,
                 learning_rate=None,
                 rho_max=None,
                 rho=None,
                 phi=None,
                 epsilon=None,
                 lam=None,
                 threshold_1=None,
                 threshold_2=None,
                 **kwargs):
        # prepare dataset
        if reader == None:
            raise Exception(" [!] Expected data reader")
        self.train_data, self.train_label, self.train_text = reader.get_sparse_matrix(
            'all', mode='count')
        self.test_data, self.test_label, self.test_text = reader.get_matrix(
            'test', mode='count')

        self.reader = reader
        self.model_path = model_path
        self.n_classes = self.reader.get_n_classes()

        print("AMM init model.")
        if emb_mat is None:
            self.Net = net(max_topic_num, batch_size, **kwargs).to(device)
        else:
            emb_mat = torch.from_numpy(emb_mat.astype(np.float32)).to(device)
            self.Net = net(max_topic_num, batch_size, emb_mat.shape[-1],
                           emb_mat, **kwargs).to(device)

        print(self.Net)

        self.topic_num = max_topic_num

        self.rho_max = rho_max
        self.rho = rho
        self.phi = phi
        self.epsilon = epsilon
        self.lam = lam
        self.threshold_1 = threshold_1
        self.threshold_2 = threshold_2

        self.epochs = epochs
        self.batch_size = batch_size

        # optimizer uses ADAM
        self.optimizer = optim.Adam(self.Net.parameters(), lr=learning_rate)

    def compute_loss(self, x, y, qz_x, pz_x, pi, adj_matrix, R):
        d = adj_matrix.shape[0]
        # reconstruct loss
        likelihood = -torch.sum(torch.log(y) * x, dim=1)

        # kl divergence
        kld = kl_divergence(qz_x, pz_x).mean()

        # Restricted to constructable DAG
        M = adj_matrix - torch.diag_embed(adj_matrix.diag())

        h = torch.trace(torch.matrix_exp(M)) - d  # (Zheng et al. 2018)

        f = 0.5 * torch.square(R.norm()) + self.lam * adj_matrix.norm(1)

        return likelihood, kld, h, f

    def to_np(self, x):
        return x.data.cpu().numpy()

    def save_model(self):
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        torch.save(self.Net.state_dict(), f'{self.model_path}/model.pkl')
        torch.save(self.Net.adjacency_matrix,
                   f'{self.model_path}/adj_matrix.pkl')
        with open(f'{self.model_path}/topic_num.txt', 'w') as f:
            f.write(str(self.topic_num))
        np.save(f'{self.model_path}/pi_ave.npy', self.pi_ave)
        print(f'Models save to  {self.model_path}/model.pkl')

    def load_model(self, model_filename='model.pkl'):
        model_path = os.path.join(self.model_path, model_filename)

        self.Net.load_state_dict(torch.load(model_path))
        with open(f'{self.model_path}/topic_num.txt', 'r') as f:
            self.topic_num = int(f.read())
        self.adj_matirx = self.to_np(
            torch.load(f'{self.model_path}/adj_matrix.pkl'))
        self.pi_ave = np.load(f'{self.model_path}/pi_ave.npy')
        print('AMM model loaded from {}.'.format(model_path))

    def update_topic_select(self):
        # sbp sum until reach 0.95
        
        # get children topics
        indices_child = np.argsort(self.pi_ave)[::-1]
        pi_ave_cumsum = np.cumsum(self.pi_ave[indices_child])

        self.child_topics_num = np.argmax(pi_ave_cumsum > self.threshold_1) + 1
        self.child_topics = indices_child[:self.child_topics_num].copy()

        # get parent topics
        self.pi_parent = self.pi_ave @ self.adj_matirx
        indices_parent = np.argsort(self.pi_parent)[::-1]
        pi_parent_cumsum = np.cumsum(self.pi_parent[indices_parent])
        
        self.parent_topics_num = np.argmax(pi_parent_cumsum > self.threshold_2) + 1
        self.parent_topics = indices_parent[:self.parent_topics_num].copy()
        
        # combine
        self.topics = np.union1d(self.child_topics, self.parent_topics)
        self.topic_num = self.topics.shape[0]

    def get_word_topic(self, data):
        word_topic = self.Net.infer(torch.from_numpy(data).to(device))
        word_topic = self.to_np(word_topic)
        return word_topic

    def get_topic_dist(self, kind='both'):
        topic_dist = self.Net.get_topic_dist()
        if kind == 'both':
            return topic_dist[self.topics]
        elif kind == 'child':
            return topic_dist[self.child_topics]
        elif kind == 'parent':
            return topic_dist[self.parent_topics]
        else:
            return topic_dist

    def get_topic_word(self, top_k=15, kind='both'):
        topic_dist = self.get_topic_dist(kind)
        vals, indices = torch.topk(topic_dist, top_k, dim=1)
        indices = self.to_np(indices).tolist()
        topic_words = [[self.reader.vocab[idx] for idx in indices[i]]
                       for i in range(topic_dist.shape[0])]
        return topic_words

    def inference(self, data, label):
        word_topic = self.get_word_topic(data)
        cluster = np.argmax(word_topic, axis=1)

        acc = {}
        acc['km'] = km.acc_rate(label, cluster)
        acc['ari'] = metrics.adjusted_rand_score(label, cluster)
        acc['ami'] = metrics.adjusted_mutual_info_score(label, cluster)
        acc['fmi'] = metrics.fowlkes_mallows_score(label, cluster)
        return acc

    def eval_level(self):
        trees, relation = utils.build_level(
            self.adj_matirx[self.topics][:, self.topics])
        topic_dist = self.to_np(self.get_topic_dist())
        topic_word = self.get_topic_word(top_k=10)

        for root in trees.keys():
            level = trees[root]
            for l in level.keys():
                level_topic_dist = topic_dist[level[l]]
                level_topic_word = np.array(topic_word)[level[l]].tolist()

                # print top words for each topic
                print('\t' * l + f"Level:{l}")
                for k in level[l]:
                    print('\t' * l + f'Topic {k}:', *topic_word[k])
                quality = {}
                quality['TU'] = utils.evaluate_topic_diversity(
                    level_topic_word)
                quality['c_a'] = utils.compute_coherence(
                    self.test_data, level_topic_dist, 10)
                quality['specialization'] = utils.compute_topic_specialization(
                    level_topic_dist, self.test_data)

                print('\t' * l + f"Topic quality: {quality}")

        clnpmi = []
        overlap = []
        for child in relation.keys():
            father = relation[child]
            child_topic = topic_dist[child]
            father_topic = topic_dist[father]
            clnpmi.append(
                utils.compute_clnpmi(child_topic, father_topic,
                                     self.test_data))
            overlap.append(utils.compute_overlap(child_topic, father_topic))
            print(
                f"{child}->{father}, clnpmi:{clnpmi[-1]}, overlap:{overlap[-1]}"
            )
        clnpmi_mean, overlap_mean = np.mean(clnpmi), np.mean(overlap)
        print(f"Total clnpmi:{clnpmi_mean}, total overlap:{overlap_mean}")

        child_hier_aff, unchild_hier_aff = utils.compute_hierarchical_affinity(topic_dist, relation)
        print(f'Topic Hierarchical Affinity:child {child_hier_aff}, unchild {unchild_hier_aff}')

    def evaluate(self):
        topic_dist = self.to_np(self.get_topic_dist())
        topic_word = self.get_topic_word(top_k=10, kind='both')

        # print top N words

        for k in range(self.topics.shape[0]):
            coh_topic = utils.compute_coherence(self.test_data, topic_dist[[k]], 10)

            rela = ""
            if self.topics[k] in self.child_topics:
                rela += "C "
            if self.topics[k] in self.parent_topics:
                rela += "P "

            print(
                f'Topic {k} coh[{coh_topic:.3f}]: {self.pi_ave[self.topics[k]]:.4f} {self.pi_parent[self.topics[k]]:.4f} {rela}{topic_word[k]}'
            )

        TU_1 = utils.evaluate_topic_diversity(
            self.get_topic_word(top_k=10, kind='child'))
        TU_2 = utils.evaluate_topic_diversity(
            self.get_topic_word(top_k=10, kind='parent'))
        TU = utils.evaluate_topic_diversity(topic_word)
        coherence = {}

        print(f"Total TU:{TU}, child TU:{TU_1}, parent TU:{TU_2}", coherence)
        return TU, coherence

    def sample(self):
        # get topic_word and print Top word
        child_topic_dist = self.to_np(self.get_topic_dist(kind='child'))
        parent_topic_dist = self.to_np(self.get_topic_dist(kind='parent'))

        # compute coherence
        child_coherence = utils.compute_coherence(self.test_data,
                                                  child_topic_dist, 10)
        parent_coherence = utils.compute_coherence(self.test_data,
                                                   parent_topic_dist, 10)
        print(
            f"Topic coherence: child {child_coherence}, parent {parent_coherence}"
        )
        if child_coherence > self.best_coherence:
            self.best_coherence = child_coherence
            print("New best coherence found!!")
            self.save_model()

        print(f"Current topic number:{self.topic_num}")
        pass

    def get_batches(self, batch_size=512, rand=True):
        n, d = self.train_data.shape
        self.word_count = self.train_data.sum(axis=1).A.squeeze()

        batchs = n // batch_size
        while True:
            idxs = np.arange(self.train_data.shape[0])

            if rand:
                np.random.shuffle(idxs)

            for count in range(batchs):
                beg = count * batch_size
                end = (count + 1) * batch_size

                idx = idxs[beg:end]
                data = self.train_data[idx].toarray()
                data = torch.from_numpy(data).to(device)
                count = self.word_count[idx]
                yield data, count

    def train(self):
        self.t_begin = time.time()

        self.train_generator = self.get_batches(self.batch_size)
        data_size = self.train_data.shape[0]
        n_batchs = data_size // self.batch_size

        self.best_coherence = -1
        ones = torch.ones(self.batch_size).to(device)
        loss_all = []
        t_interver_0 = time.time()
        for epoch in tqdm(range(self.epochs)):
            self.Net.train()
            epoch_loss_all = []
            epoch_likelihood_all = []
            epoch_kld_all = []
            epoch_h_all = []
            epoch_pi_ave_all = []
            eopch_f_all = []

            epoch_count_all = []
            epoch_rec_all = []

            for i in tqdm(range(n_batchs), leave=False):
                self.optimizer.zero_grad()

                ori_docs, word_count = next(self.train_generator)
                gen_docs, docs_delta, qz_x, pz_x, pi, adj_matirx = self.Net(
                    ori_docs)

                likelihood, kld, h, f = self.compute_loss(
                    ori_docs, gen_docs, qz_x, pz_x, pi, adj_matirx, docs_delta)

                batch_loss = likelihood + kld + self.phi * f + 0.5 * self.rho * h * h + self.epsilon * h
                batch_loss.backward(ones)
                self.optimizer.step()

                # save record
                epoch_loss_all.append(self.to_np(batch_loss))
                epoch_likelihood_all.append(self.to_np(likelihood))
                epoch_kld_all.append(self.to_np(kld))
                epoch_h_all.append(self.to_np(h))
                epoch_pi_ave_all.append(self.to_np(pi.mean(dim=0)))
                eopch_f_all.append(self.to_np(f))
                epoch_rec_all.append(self.to_np(likelihood + kld))
                epoch_count_all.append(word_count)

            if epoch > 40:

                # update $\rho$ and $\alpha$
                if h > 0.25 * epoch_h_all[-1] and self.rho < self.rho_max:
                    self.rho *= 2
                    self.epsilon += self.rho * h.item()

                # update Net Gamma and Temperature
                self.Net.gamma = 0.5 + 1 / 2 * np.exp(-0.02 * epoch)
                self.Net.temperature = 5 * np.exp(-0.02 * epoch) + 1e-4

            epoch_loss = np.mean(epoch_loss_all)
            epoch_likelihood = np.mean(epoch_likelihood_all)
            epoch_kld = np.mean(epoch_kld_all)
            epoch_h = np.mean(epoch_h_all)
            epoch_f = np.mean(eopch_f_all)

            epoch_ppl = np.exp(
                np.mean(epoch_rec_all) / np.mean(epoch_count_all))
            epoch_ppl_perdoc = np.exp(
                np.divide(epoch_rec_all, epoch_count_all).mean())
            loss_all.append(epoch_loss)

            # update topic number
            self.pi_ave = np.mean(epoch_pi_ave_all, axis=0)
            self.adj_matirx = self.to_np(adj_matirx)
            self.update_topic_select()

            print(
                f'Epoch: {epoch}/{self.epochs}, loss: {epoch_loss:.3f}, likelihood: {epoch_likelihood:.3f}, kld: {epoch_kld:.3f}, h: {epoch_h:.6f}, f: {epoch_f:.4f}, PPL: {epoch_ppl:.3f}, PPL per doc: {epoch_ppl_perdoc:.3f}'
            )

            self.Net.eval()
            if (epoch + 1) % 10 == 0:
                self.sample()

            if (epoch + 1) % 30 == 0:
                self.evaluate()

            if (epoch + 1) % 100 == 0 and epoch_h == 0:
                self.eval_level()

            t_interver_1 = time.time()
            if (epoch + 1) % 5 == 0:
                print("Time consumption:", t_interver_1 - t_interver_0)
            t_interver_0 = t_interver_1

        self.t_end = time.time()
        print('Time of training-{}'.format((self.t_end - self.t_begin)))

    def test(self):
        self.load_model()
        self.Net.eval()
        self.best_coherence = 999

        self.update_topic_select()
        self.evaluate()
        self.sample()
        self.eval_level()


def main(mode='Train',
         dataset="20news",
         max_topic_num=200,
         emb_type="glove",
         **kwargs):
    base_path = ".."

    data_path = f"{base_path}/data/{dataset}"
    reader = TextReader(data_path)

    if emb_type == "bert":
        bert_emb_path = f"{base_path}/emb/bert.npy"
        embedding_mat = utils.build_bert_embedding(bert_emb_path, reader.vocab,
                                                   data_path)
    elif emb_type == "glove":
        emb_path = f"{base_path}/emb/glove.6B.300d.txt"
        embedding_mat = utils.build_embedding(emb_path, reader.vocab,
                                              data_path)[:-1]
    else:
        embedding_mat = None

    model_path = f'{base_path}/model/AMM/{dataset}_{max_topic_num}_{reader.vocab_size}'
    model = AMM(reader, max_topic_num, model_path, embedding_mat, **kwargs)

    if mode == 'Train':
        model.train()
    elif mode == 'Test':
        model.test()
    else:
        print(f'Unknowned mode {mode}!')


if __name__ == '__main__':
    _, dataset, mode = sys.argv
    config = yaml.load(open('config.yaml'), yaml.FullLoader)
    print(config[dataset])
    main(mode=mode, **config[dataset])