import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, matthews_corrcoef
from torch.utils.data import DataLoader
from Bio import SeqIO
from Processing import Processing
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import joblib

def read_fasta(fasta_file, len_criteria=1000):
    result = []
    seq_ids = []
    fp = open(fasta_file)
    for seq_record in SeqIO.parse(fp, 'fasta'):
        seq = seq_record.seq.upper()
        seq_id = seq_record.id
        if len(seq) <= len_criteria:
            seq += '_' * (len_criteria - len(seq))
        result.append(str(seq))
        seq_ids.append(int(seq_id[-1]))
    fp.close()
    return result, seq_ids


test = "Your_fasta_file"
tests, test_ids = read_fasta(test)
testDataset = Processing(tests, test_ids)
testDataloader = DataLoader(testDataset, batch_size=32, shuffle=False)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dropout = 0.1
        self.n_vocab = 21
        self.dim_model = 20
        self.embed = 20
        self.pad_size = 32
        self.num_encoder = 1
        self.hidden = 512
        self.last_hidden = 256
        self.num_head = 5
        self.embedding = nn.Embedding(self.n_vocab, self.embed, padding_idx=self.n_vocab - 1)
        self.postion_embedding = Positional_Encoding(self.embed, self.pad_size, self.dropout, self.device)
        self.encoder = Encoder(self.dim_model, self.num_head, self.hidden, self.dropout)
        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder)
            for _ in range(self.num_encoder)])

        self.explainECs = [0, 1]
        self.layer_info = [[4,32], [24,12], [12,24]]
        self.cnn0 = CNN(self.layer_info)
        self.pool = nn.MaxPool1d(kernel_size=962, stride=1)
        self.fc1 = nn.Linear(in_features=384, out_features=512)
        self.bn1 = nn.BatchNorm1d(num_features=512)
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.selu = nn.SELU()
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(in_features=512, out_features=2)
        self.bn2 = nn.BatchNorm1d(num_features=2)
        self.out_act = nn.Sigmoid()
        self.init_weights()



    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)


    def forward(self, x):
        out = self.embedding(x)
        for encoder in self.encoders:
            out = encoder(out)
        x = out.permute(0, 2, 1)
        x = self.cnn0(x)
        x = self.pool(x)
        x = x.view(-1, 128 * 3)
        embedding = x
        x = self.gelu(self.bn1(self.fc1(x)))
        x = self.out_act(self.bn2(self.fc2(x)))
        return x, embedding


class CNN(nn.Module):
    def __init__(self, layer_info):
        super(CNN, self).__init__()
        self.layers = nn.ModuleList()
        pooling_sizes = []
        for subnetwork in layer_info:
            pooling_size = 0
            self.layers += [self.make_subnetwork(subnetwork)]
            for kernel in subnetwork:
                pooling_size += (-kernel + 1)
            pooling_sizes.append(pooling_size)

        if len(set(pooling_sizes)) != 1:
            raise "Different kernel sizes between subnetworks"
        num_subnetwork = len(layer_info)



        self.conv = nn.Conv1d(in_channels=128 * num_subnetwork, out_channels=128 * 3, kernel_size=1)
        self.batchnorm = nn.BatchNorm1d(num_features=128 * 3)
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.gelu = nn.GELU()



    def make_subnetwork(self, subnetwork):
        subnetworks = []
        for i, kernel in enumerate(subnetwork):
            if i == 0:
                subnetworks.append(
                    nn.Sequential(
                        nn.Conv1d(in_channels=20, out_channels=128, kernel_size=kernel),
                        nn.BatchNorm1d(num_features=128),
                        nn.ReLU(),
                        nn.Dropout(p=0.1)
                    )
                )
            else:
                subnetworks.append(
                    nn.Sequential(
                        nn.Conv1d(in_channels=128, out_channels=128, kernel_size=kernel),
                        nn.BatchNorm1d(num_features=128),
                        nn.ReLU(),
                        nn.Dropout(p=0.1)
                    )
                )
        return nn.Sequential(*subnetworks)

    def forward(self, x):
        xs = []
        for layer in self.layers:
            xs.append(layer(x))
        x = torch.cat(xs, dim=1)
        x = self.gelu(self.batchnorm(self.conv(x)))

        return x



class Encoder(nn.Module):
    def __init__(self, dim_model, num_head, hidden, dropout):
        super(Encoder, self).__init__()
        self.attention = Multi_Head_Attention(dim_model, num_head, dropout)
        self.feed_forward = Position_wise_Feed_Forward(dim_model, hidden, dropout)

    def forward(self, x):
        out = self.attention(x)
        out = self.feed_forward(out)

        return out


class Positional_Encoding(nn.Module):
    def __init__(self, embed, pad_size, dropout, device):
        super(Positional_Encoding, self).__init__()
        self.device = device
        self.pe = torch.tensor([[pos / (10000.0 ** (i // 2 * 2.0 / embed)) for i in range(embed)] for pos in range(1000)])
        self.pe[:, 0::2] = np.sin(self.pe[:, 0::2])
        self.pe[:, 1::2] = np.cos(self.pe[:, 1::2])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        out = x + nn.Parameter(self.pe, requires_grad=False).to(self.device)
        out = self.dropout(out)

        return out


class Scaled_Dot_Product_Attention(nn.Module):
    def __init__(self):
        super(Scaled_Dot_Product_Attention, self).__init__()

    def forward(self, Q, K, V, scale=None):

        attention = torch.matmul(Q, K.permute(0, 2, 1))
        if scale:
            attention = attention * scale
        attention = F.softmax(attention, dim=-1)
        context = torch.matmul(attention, V)
        return context


class Multi_Head_Attention(nn.Module):
    def __init__(self, dim_model, num_head, dropout=0):
        super(Multi_Head_Attention, self).__init__()
        self.num_head = num_head
        assert dim_model % num_head == 0
        self.dim_head = dim_model // self.num_head
        self.fc_Q = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_K = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_V = nn.Linear(dim_model, 2*num_head * self.dim_head)
        self.activation = nn.GLU()
        self.attention = Scaled_Dot_Product_Attention()
        self.fc = nn.Linear(num_head * self.dim_head, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.float()
        Q = self.fc_Q(x)
        K = self.fc_K(x)
        V = self.activation(self.fc_V(x))
        Q = Q.view(batch_size * self.num_head, -1, self.dim_head)
        K = K.view(batch_size * self.num_head, -1, self.dim_head)
        V = V.view(batch_size * self.num_head, -1, self.dim_head)
        scale = K.size(-1) ** -0.5
        context = self.attention(Q, K, V, scale)
        context = context.view(batch_size, -1, self.dim_head * self.num_head)
        out = self.fc(context)
        out = self.dropout(out)
        out = out + x
        out = self.layer_norm(out)
        return out


class Position_wise_Feed_Forward(nn.Module):
    def __init__(self, dim_model, hidden, dropout=0):
        super(Position_wise_Feed_Forward, self).__init__()
        self.fc1 = nn.Linear(dim_model, hidden)
        self.fc2 = nn.Linear(hidden, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = out + x  # 残差连接
        out = self.layer_norm(out)
        return out


model = Model()
model = torch.load('Model_Parameters.pkl')
model.cuda()
classifier = joblib.load("svm.m")


total_valid_accuracy = 0
total_valid_loss = 0

i = 0
k = 0
label = []
start_time = time.time()
model.eval()
with torch.no_grad():
    for data in testDataloader:
        seq, targets = data
        seq = seq.cuda()
        targets = targets.cuda()
        outputs, embedding_test = model(seq.long())
        if k == 0:
            test_data = embedding_test
        else:
            test_data = torch.cat((test_data, embedding_test), dim=0)
        k = 1


test_data = test_data.cpu().detach().numpy()
prediction = classifier.predict(test_data)
end = time.time()
print(end - start_time)
print('Precision: ', precision_score(prediction, test_ids))
print('Recall: ', recall_score(prediction, test_ids))
print('Accuracy: ', accuracy_score(prediction, test_ids))
print('MCC: ', matthews_corrcoef(prediction, test_ids))