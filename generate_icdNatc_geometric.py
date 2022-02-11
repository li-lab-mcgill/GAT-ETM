import os, time
import numpy as np
import pickle 
import csv
import networkx as nx
import sys
sys.path.append('..')
from icdcodex import hierarchy
from generate_code_embedding import Code2Vec
from sklearn.metrics.cluster import adjusted_rand_score
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
import community as community_louvain
import leidenalg as la
from sklearn.neighbors import kneighbors_graph

from visualize import CodeDict

from IPython import embed


atc2icd_df = pickle.load(open('atc_to_icd9_dataframe.pkl', 'rb'))
atc2icd = np.array(atc2icd_df)
atc_tree = pickle.load(open('atc_graph.pkl', 'rb'))
icd_tree = hierarchy.icd9()[0]
icd10_tree = pickle.load(open('icd10_graph_networkx.pkl', 'rb'))
icd9_tree = pickle.load(open('icd9_networkx.pkl', 'rb'))

class Code_Dict__(CodeDict):
    def __init__(self):
        super().__init__()

    def queryICD(self, icd):
        icd_input = icd
        for suf in ['', '0', '00', '1', '10', '11']:
            if icd + suf in self.ICD_dict:
                icd = icd + suf
                break
        if not icd in self.ICD_dict:
            icd = icd[:-1]
            for suf in ['', '0', '00', '1', '10', '11']:
                if icd + suf in self.ICD_dict:
                    icd = icd + suf
                    break

        if icd in self.ICD_dict:
            return icd
        else:
            return None

icd_tree = icd9_tree
code_dict = Code_Dict__()
nx.set_node_attributes(icd_tree, name='type', values='ICD9')
nx.set_node_attributes(atc_tree, name='type', values='ATC')
dis_from_root_icd = nx.single_source_shortest_path_length(icd_tree, 'ICD.9')
nx.set_node_attributes(icd_tree, name='level', values=dis_from_root_icd)
dis_from_root_atc = nx.single_source_shortest_path_length(atc_tree, 'root')
nx.set_node_attributes(atc_tree, name='level', values=dis_from_root_atc)
G = nx.compose(atc_tree, icd_tree)

def graph_augmentation(): 
    e = 0.9
    for i in icd_tree.nodes():
        if len(list(icd_tree.neighbors(i))) == 0:
            a = list(icd_tree.predecessors(i))[0]
            w = e
            while len(list(icd_tree.predecessors(a))) > 0:
                a = list(icd_tree.predecessors(a))[0]
                G.add_edge(i, a, weight=w)
                # G.add_edge(a, i)
                w *= e

    for i in atc_tree.nodes():
        if len(list(atc_tree.neighbors(i))) == 0:
            a = list(atc_tree.predecessors(i))[0]
            w = e
            while len(list(atc_tree.predecessors(a))) > 0:
                a = list(atc_tree.predecessors(a))[0]
                G.add_edge(i, a, weight=w)
                # G.add_edge(a, i)
                w *= e

AUGMENTED = False 
EMBEDDING = False
print('Augmented: '+str(AUGMENTED))
print('Embedding: '+str(EMBEDDING))
if AUGMENTED:
    graph_augmentation()

# FIXME: rename the roots!!
for row in atc2icd:
    x, y, y_desc = row[0], row[3], row[4]
    # y = y.replace('.','')
    # if not x in G:
    #     print(f'{x} is not in ATC tree')
    # if not y in G:
    #     if not y_desc in G:
    #         if y+'0' in G:
    #             y += '0'
    #         elif y+'9' in G:
    #             y += '9'
    #         elif y+'1' in G:
    #             y += '1'
    #         else:
    #             print(f'{y}, {y_desc} is not in ICD tree')
    #     else:
    #         y = y_desc
    if x in G and y in G:
        G.add_edge(x,y)
    else:
        print((x,y))


G = G.to_undirected()


def cluster_rho(mode, rho, vocab, label, display=True):
    print('cluster rho....')
    vocab = np.array(vocab)
    # code_class = np.array([i[0] for i in vocab])
    code_class = label
    num_class = len(np.unique(code_class))
    V = len(vocab)

    if display: 
        # Display rhos
        tsne_30 = TSNE()
        rho_2d = tsne_30.fit_transform(rho)

        fig, ax = plt.subplots()
        # code_class = np.array(self.code_dict.classes)[code_class]
        scatter = plt.scatter(rho_2d[:,0], rho_2d[:,1], s=2, c=code_class, cmap=cm.tab20) # color=[colors.get(i) for i in code_class])
        legend1 = ax.legend(*scatter.legend_elements(num=20), title='classes')
        ax.add_artist(legend1)
        plt.show()

        # Only display topic words 
        # beta_collapsed = self.betas_collapsed[mode]
        # words, topic_code, topic_code_ref = self.get_topic_words(mode, vocab, beta_collapsed)
        # unique_words = np.unique(words)
        # fig, ax = plt.subplots()
        # scatter = plt.scatter(rho_2d[unique_words, 0], rho_2d[unique_words, 1], c=np.array(code_class)[unique_words], cmap=cm.tab20)
        # legend1 = ax.legend(*scatter.legend_elements(num=20), title='classes')
        # ax.add_artist(legend1)
        # plt.show()


    # K-Means
    kmeans = KMeans(n_clusters=num_class, random_state=42).fit(rho)
    kmeans_partition = kmeans.labels_
    ari_kmeans = adjusted_rand_score(kmeans_partition, code_class)

    # Louvain
    def jaccard_similarity(A):
        if isinstance(A, csr_matrix):
            intersection = A.dot(A.T)
            addition = A.sum(1).T + A.sum(1)
            union = addition - intersection
            norm_A = intersection / (union +1e-6)
            return norm_A
        return None
    if mode == 'ATC': 
        num_neighbors = 90
        res = 1
    else:
        num_neighbors = 150
        res = 1
    A = kneighbors_graph(rho, num_neighbors, mode='connectivity', include_self=False)
    norm_A = jaccard_similarity(A)#.todense() # jaccard_similarity
    G = nx.from_numpy_matrix(norm_A)
    
    louvain_partition = list(community_louvain.best_partition(G, resolution=res,random_state=42).values())
    ari_louvain = adjusted_rand_score(louvain_partition, code_class)
    ari_louvain

    print(f'{mode}  ARI: KMeans: {ari_kmeans}  Louvain: {ari_louvain}')

def evaluate():
    # evaluate
    # out_deg = dict(atc.out_degree())
    # vocab = np.array(list(atc.nodes()))[np.array(list(out_deg.values())) == 0]
    vocab_atc = np.array(list(set(list(atc_tree.nodes())) - set(['root'])))
    label_atc = np.array([i[0] for i in vocab_atc])
    rhos_atc = embedder.to_vec(vocab_atc)
    cluster_rho('ATC', rhos_atc, vocab_atc, label_atc, display=False)

    out_deg = dict(icd9_tree.out_degree())
    vocab_icd = np.array(list(icd9_tree.nodes()))[np.array(list(out_deg.values())) == 0]
    label_icd = np.array([code_dict.query_class_ICD(i) for i in vocab_icd])
    rhos_icd = embedder.to_vec(vocab_icd)
    cluster_rho('ICD', rhos_icd, vocab_icd, label_icd, display=False)

WINDOW = 8 # int(sys.argv[1]) # 3
WALK_LENGTH = 20 # int(sys.argv[2]) # 10  # default 10
NUM_WALKS = 10 # int(sys.argv[3]) # 20   # default 10 
NUM_FEATURE = 256
print((WINDOW, WALK_LENGTH, NUM_WALKS,  NUM_FEATURE))

if EMBEDDING:
    embedder = Code2Vec(num_embedding_dimensions=NUM_FEATURE, walk_length=WALK_LENGTH, num_walks=NUM_WALKS, window=WINDOW)
    codes = G.nodes()
    print('fit node2vec model...')
    embedder.fit(G.to_undirected(), codes)
    print(f'fitting time: {embedder.node2vec.total_train_time}')
    
    evaluate()




# generate nodes for dataset
renumber = {}
vocab_icd = pickle.load(open('icd_vocab.pkl','rb'))
rho_icd = []
for id, i in enumerate(vocab_icd):
    icd = code_dict.queryICD(i)
    if len(icd) > 3: 
        icd = icd[:3]+'.'+icd[3:]
    if icd == None or icd == 'Not Found':
        print(i)
        rho.append(np.zeros((1, 256)))
    else:
        if not icd in G.nodes():
            print(i)
        renumber[icd] = id
        # rho_icd.append(embedder.to_vec([icd]))
print(len(renumber), len(vocab_icd))

new_codes = {'A10BX04': 'A10BJ01', 'A10BX07': 'A10BJ02', 'C01CA74': 'S01EA51',  'C05AA51': 'R01AD60', 'D07AC30': 'D07AB30', 'G03AB11': 'G03FA13', 'J05AB04': 'J05AP01', 'J05AE06': 'J05AR10', 'J05AE11': 'J05AP02', 'J05AE12': 'J05AP03', 'J05AX08': 'J05AJ01', 'J05AX12': 'J05AJ03', 'J05AX65': 'J05AP51', 'L01XE01': 'L01EA01', 'L01XE02': 'L01EB01', 'L01XE03': 'L01EB02', 'L01XE04': 'L01EX01', 'L01XE05': 'L01EX02', 'L01XE06': 'L01EA02', 'L01XE07': 'L01EH01', 'L01XE08': 'L01EA03', 'L01XE10': 'L01EG02', 'L01XE11': 'L01EX03', 'L01XE12': 'L01EX04', 'L01XE13': 'L01EB03', 'L01XE15': 'L01EC01', 'L01XE16': 'L01ED01', 'L01XE17': 'L01EK01', 'L01XE18': 'L01EJ01', 'L01XE21': 'L01EX05', 'L01XE23': 'L01EC02', 'L01XX14': 'L01XF01', 'L01XX43': 'L01XJ01', 'N02AX52': 'N02AJ13', 'N07XX09': 'L04AX07', 'S01AX12': 'S01AE02', 'S01ED53': 'S01ED03'}
vocab_atc = pickle.load(open('atc_vocab.pkl','rb'))
for id, atc in enumerate(vocab_atc):
    if atc in G.nodes():
        renumber[atc] = id + len(vocab_icd)
        # rho_atc.append(embedder.to_vec([code]))
    else:
        new_atc = new_codes[atc]
        if new_atc in set(vocab_atc):
            print((atc, new_atc))
            G.add_node(atc, type='ATC', level=5)
            neighbors = list(G.neighbors(new_atc))
            G.add_edges_from(zip([atc]*len(neighbors), list(neighbors)))
            renumber[atc] = id + len(vocab_icd)
        else:
            renumber[new_atc] = id + len(vocab_icd)
        # rho_atc.append(embedder.to_vec([code]))
print(len(renumber), len(vocab_atc))
    
V = len(vocab_icd) + len(vocab_atc)

remained_nodes = set(list(G.nodes()))-set(list(renumber.keys()))
renumber.update(dict(zip(remained_nodes, list(range(V, V+len(remained_nodes))))))

if EMBEDDING:
    node2vec_embeddings = np.zeros((V+len(remained_nodes), NUM_FEATURE))
    for k, v in renumber.items():
        try:
            node2vec_embeddings[v] = embedder.to_vec([k])
        except:
            node2vec_embeddings[v] = embedder.to_vec([new_codes[k]])

graphnode_vocab = dict(zip(list(renumber.values()), list(renumber.keys())))
pickle.dump(graphnode_vocab, open('graphnode_vocab.pkl', 'wb'))
G_ = nx.relabel_nodes(G, renumber)
nx.set_node_attributes(G_, name='code', values=graphnode_vocab)
embed()


if not AUGMENTED:
    pickle.dump(G_, open(f'icdatc_graph_{WINDOW}_{WALK_LENGTH}_{NUM_WALKS}_{NUM_FEATURE}_renumbered_by_vocab.pkl', 'wb'))
    pickle.dump(node2vec_embeddings, open(f'icdatc_embed_{WINDOW}_{WALK_LENGTH}_{NUM_WALKS}_{NUM_FEATURE}_by_vocab.pkl', 'wb'))

else:
    print('augmented')
    pickle.dump(G_, open(f'augmented_icdatc_graph_{WINDOW}_{WALK_LENGTH}_{NUM_WALKS}_{NUM_FEATURE}_renumbered_by_vocab.pkl', 'wb'))
    pickle.dump(node2vec_embeddings, open(f'augmented_icdatc_embed_{WINDOW}_{WALK_LENGTH}_{NUM_WALKS}_{NUM_FEATURE}_by_vocab.pkl', 'wb'))

    G_ = pickle.load(open(f'augmented_icdatc_graph_{WINDOW}_{WALK_LENGTH}_{NUM_WALKS}_{NUM_FEATURE}_renumbered_by_vocab.pkl', 'rb'))
    # pickle.dump(G, open(f'w_augmented_icdatc_graph_{NUM_FEATURE}_renumbered_by_vocab.pkl', 'wb'))