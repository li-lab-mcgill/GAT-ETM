import torch
import torch.nn.functional as F 
import numpy as np 
import math 

from torch import nn
import torch_geometric
import torch_geometric.nn as gnn


from IPython import embed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_bounds(mat, upper, lower):
    y_upper = (torch.ones(mat.size())*upper).float().to(mat.device)
    y_lower = (torch.ones(mat.size())*lower).float().to(mat.device)
    mat = torch.where(mat >= lower, mat, y_lower)
    mat = torch.where(mat <= upper, mat, y_upper)
    return mat

class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()

        self.linear_or_not = True #default is linear model
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            #Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            #Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()
        
            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            #If linear model
            return self.linear(x)
        else:
            #If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
            return self.linears[self.num_layers - 1](h)


class GCNet(nn.Module):
    def __init__(self, num_nodes, num_feature, node_embeddings=None, dropout_rate=0.4):
        nn.Module.__init__(self)
        node_embeddings = None # FIXME: for etm
        if not isinstance(node_embeddings, torch.Tensor):
            print('node2vec embedding: False')
            self.init = nn.Embedding(num_nodes, num_feature)
        else:
            print('node2vec embedding: True')
            # self.init = nn.Embedding.from_pretrained(node_embeddings, freeze=False)
            self.init = nn.Embedding.from_pretrained(node_embeddings, freeze=True)

        self.in_drop = nn.Dropout(0.1)

        self.num_layers = 3
        self.gcns = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        for i in range(self.num_layers):
            self.gcns.append(gnn.GATConv(num_feature, num_feature, heads=4, concat=False))
            self.batch_norms.append(nn.BatchNorm1d(num_feature))

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.pooling = nn.MaxPool1d(self.num_layers+1)
        self.fc = nn.Linear(num_feature * (self.num_layers+1), num_feature)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # FIXME: getm 
        # return x


        # FIXME: initialize node embedding 
        x = self.init.weight
        # FIXME: etm if x is randomly initialized and never feed to GAT
        return x
        x = self.in_drop(x)
        embed_rep = [x]

        for i in range(self.num_layers):
            x = self.gcns[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = self.activation(x)
            x = self.dropout(x)
            embed_rep.append(x)

        # FIXME:  fc
        # embed_rep = torch.cat(embed_rep, dim=1)
        # output = self.fc(embed_rep)

        # FIXME: pooling
        embed_rep = torch.stack(embed_rep, dim=2)
        output = self.pooling(embed_rep).squeeze(-1)
        # output = torch.stack(embed_rep).max(0)[0]
        # output = torch.stack(embed_rep).mean(0)[0]
        return output

class GETM(nn.Module):
    def init_graph(self, G, embed):
        x = torch.from_numpy(embed).to(self.device).float()

        edge_index = torch.tensor(list(G.edges()), dtype=torch.long, device=self.device)
        edge_index = torch.vstack([edge_index, edge_index.flip(1)])
        print(f'# of edges: {edge_index.shape[0]}')

        self.graph = torch_geometric.data.Data(x=x, edge_index=edge_index.t().contiguous()).to(self.device)
        

    def __init__(self, device, num_topics, code_types, vocab_size, t_hidden_size, rho_size, emsize, theta_act, 
                graph, graph_embed, 
                embeddings=None, train_embeddings=True, enc_drop=0.1, upper=100, lower=-100, share_alpha=False):
        super(GETM, self).__init__()
        self.device = device

        ## define hyperparameters
        self.num_topics = num_topics
        self.code_types = code_types
        self.vocab_size = vocab_size
        self.t_hidden_size = t_hidden_size
        self.rho_size = rho_size
        self.enc_drop = enc_drop
        self.emsize = emsize
        self.t_drop = nn.Dropout(enc_drop)

        self.theta_act = self.get_activation(theta_act)

        self.init_graph(graph, graph_embed)

        self.train_embeddings = train_embeddings

        self.upper = upper
        self.lower = lower
        
        ## FIXME: define the GCN for word embedding
        self.GCN = GCNet(self.graph.x.shape[0], rho_size, node_embeddings=torch.from_numpy(graph_embed).float())
        # self.GCN = GCNet(self.graph.x.shape[0], rho_size, node_embeddings=None)

        ## define the matrix containing the topic embeddings
        self.share_alpha = share_alpha
        if not share_alpha:
            self.alphas = {}
            for i, c in enumerate(self.code_types):
                self.alphas[c] = nn.Linear(rho_size, num_topics, bias=False) # L x K
            self.alphas = nn.ModuleDict(self.alphas)
        else:
            self.alphas = nn.Linear(rho_size, num_topics, bias=False) # L x K
    
        self.dropout = nn.Dropout(0.1)
        ## define variational distribution for \theta_{1:D} via amortizartion
        self.q_theta = nn.Sequential(
                nn.Linear(sum(vocab_size), t_hidden_size),
                self.theta_act,
                self.dropout,
                nn.Linear(t_hidden_size, t_hidden_size),
                self.theta_act,
            )
        self.mu_q_theta = nn.Linear(t_hidden_size, num_topics, bias=True)
        self.logsigma_q_theta = nn.Linear(t_hidden_size, num_topics, bias=True)

    def get_activation(self, act):
        if act == 'tanh':
            act = nn.Tanh()
        elif act == 'relu':
            act = nn.ReLU()
        elif act == 'softplus':
            act = nn.Softplus()
        elif act == 'rrelu':
            act = nn.RReLU()
        elif act == 'leakyrelu':
            act = nn.LeakyReLU()
        elif act == 'elu':
            act = nn.ELU()
        elif act == 'selu':
            act = nn.SELU()
        elif act == 'glu':
            act = nn.GLU()
        else:
            print('Defaulting to tanh activations...')
            act = nn.Tanh()
        return act 

    # theta ~ mu + std N(0,1)
    def reparameterize(self, mu, logvar):
        """Returns a sample from a Gaussian distribution via reparameterization.
        """
        if self.training:
            std = torch.exp(0.5 * logvar) 
            eps = torch.randn_like(std)
            return eps.mul_(std).add_(mu)
        else:
            return mu

    def encode(self, bows):
        """Returns paramters of the variational distribution for \theta.

        input: bows
                batch of bag-of-words...tensor of shape bsz x V
        output: mu_theta, log_sigma_theta
        """
        q_theta = self.q_theta(bows)
        if self.enc_drop > 0:
            q_theta = self.t_drop(q_theta)
        mu_theta = self.mu_q_theta(q_theta)
        logsigma_theta = self.logsigma_q_theta(q_theta)
        logsigma_theta = set_bounds(logsigma_theta, self.upper, self.lower)

        # KL[q(theta)||p(theta)] = lnq(theta) - lnp(theta)
        kl_theta = -0.5 * torch.sum(1 + logsigma_theta - mu_theta.pow(2) - logsigma_theta.exp(), dim=-1).mean()
        return mu_theta, logsigma_theta, kl_theta
    
    def get_rho(self):
        self.rho = {}
        partition = np.cumsum([0]+self.vocab_size)
        #FIXME: 
        GCN_out = self.GCN(self.graph)
        # GCN_out = self.graph.x
        for i, c in enumerate(self.code_types):
            self.rho[c] = GCN_out[partition[i]:partition[i+1]]
            

    def get_beta(self):
        self.get_rho()
        beta = {}
        for i, c in enumerate(self.code_types):
            # FIXME:
            if not self.share_alpha:
                logit = self.alphas[c](self.rho[c])
            else:
                logit = self.alphas(self.rho[c])
            beta[c] = F.softmax(logit, dim=0).transpose(1,0) ## softmax over vocab dimension
        return beta

    def get_theta(self, normalized_bows):
        mu_theta, logsigma_theta, kld_theta = self.encode(normalized_bows)
        z = self.reparameterize(mu_theta, logsigma_theta)
        theta = F.softmax(z, dim=-1) 
        return theta, kld_theta

    def decode(self, theta, beta, bow):

        nll = torch.zeros(len(self.code_types)).to(self.device)
        partition = np.cumsum([0]+self.vocab_size)
        for i, c in enumerate(self.code_types):
            bows_i = bow[:,partition[i]:partition[i+1]]
            res_i = torch.mm(theta, beta[c])
            preds_i = torch.log(res_i+1e-6)
            nll[i] = -(preds_i * bows_i).sum(1).mean()
        return nll 

    def forward(self, bows, norm_bows, theta=None, aggregate=True):
        ## get \theta
        if theta is None:
            theta, kld_theta = self.get_theta(norm_bows)
        else:
            kld_theta = None

        ## get \beta
        beta = self.get_beta()

        ## get prediction loss
        nll = self.decode(theta, beta, bows)
        recon_loss = nll.sum()

        return recon_loss, kld_theta

