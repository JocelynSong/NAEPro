import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules.loss import _Loss


aas = ['L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I',
       'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H',
       'W', 'C', 'X', 'B', 'U', 'Z']
aa2idx = {aa: idx for idx, aa in enumerate(aas)}

parser = argparse.ArgumentParser()
parser.add_argument("--protein", type=str, default="avGFP", help="protein family")
args = parser.parse_args()
protein = args.protein


class ProteinDataset(torch.utils.data.Dataset):
    def __init__(self, file):
        super(ProteinDataset, self).__init__()
        self.tokens_list = []
        self.lines = []
        self.sizes = []

        with open(file, 'r', encoding="utf-8") as f:
            for line in f:
                line = line.strip().split()[0].strip()
                self.tokens_list.append(self.encode_line(line))
                self.lines.append(line)
                self.sizes.append(len(line))
        self.sizes = np.array(self.sizes)
        self.max_size = max(self.sizes)

    def encode_line(self, line):
        ids = torch.IntTensor(len(line))
        for idx, aa in enumerate(line):
            ids[idx] = aa2idx[aa]
        return ids

    def __getitem__(self, index):
        return self.tokens_list[index]

    def __len__(self):
        return len(self.tokens_list)


train_dataset = ProteinDataset('/mnt/data/zhenqiaosong/protein_design/datasets/{}.txt'.format(protein))
# valid_dataset = ProteinDataset('./avGFP.valid.txt')
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
# test_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=1)


def Embedding(num_embeddings, embedding_dim):
    m = nn.Embedding(num_embeddings, embedding_dim)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    return m


class MRF(nn.Module):
    def __init__(self, length):
        super(MRF, self).__init__()
        self.single_energy = Embedding(length * 20, 1)
        self.pair_energy = Embedding(length * length * 400, 1)
        self.length = length
        self.lambda_single = 1.0
        self.lambda_pair = 0.2 * (self.length-1)

    def forward(self, x):
        pos = torch.tensor(range(self.length)).to("cuda")  # [length]
        singles = self.single_energy(pos * 20 + x)  # [batch, length, 1]
        all_singles = self.single_energy.weight.reshape(-1, 20).repeat(x.size()[0], 1, 1)   # [batch, length, 20]

        pairs = torch.FloatTensor(x.size()[0], self.length, self.length).to("cuda")  # [batch, length, length]
        for i in range(self.length):
            enr = self.pair_energy(((i * self.length + pos) * 400).unsqueeze(0) + x[:, i].unsqueeze(1) * 20 + x).squeeze(1)
            pairs[:, i] = enr.squeeze(2)

        all_pairs = torch.FloatTensor(x.size()[0], self.length, self.length, 20).to("cuda")
        for i in range(self.length):
            for j in range(20):
                temp_x = x
                temp_x[:, i] = j
                pair_enr = self.pair_energy((i * self.length + pos).unsqueeze(0) * 400 + temp_x[:, i].unsqueeze(1) * 20 + temp_x).squeeze(1)
                all_pairs[:, i, :, j] = pair_enr.squeeze(2)

        singles = singles.squeeze(2)
        energy = singles + torch.sum(pairs, dim=2)  # [batch, length]
        total_energy = torch.log(torch.sum(torch.exp(all_singles + torch.sum(all_pairs, dim=2)), dim=-1))  # [batch, length]
        final_energy = torch.mean(torch.sum(- energy + total_energy, dim=1))
        single_reg = self.lambda_single * torch.sum(torch.pow(self.single_energy.weight, 2))
        pair_reg = self.lambda_pair * torch.sum(torch.pow(self.pair_energy.weight, 2))
        single_lasso_reg = self.lambda_single * torch.sum(torch.abs(self.single_energy.weight))
        pair_lasso_reg = self.lambda_pair * torch.sum(torch.abs(self.pair_energy.weight))
        loss = final_energy + single_reg + pair_reg + single_lasso_reg + pair_lasso_reg
        return loss


class MRFLoss(_Loss):
    def __init__(self):
        super().__init__()

    def forward(self, model, sample):
        loss = model(sample)
        return loss


mrf = MRF(train_dataset.max_size)
mrf.cuda()
criterion = MRFLoss()
optimizer = optim.Adam(mrf.parameters(), lr=0.01)

for epoch in range(5):
    for i, sample in enumerate(train_loader, 0):
        sample = sample.to("cuda")
        optimizer.zero_grad()
        loss = criterion(mrf, sample)
        loss.backward()
        optimizer.step()
        print('epoch=%d, step=%d, loss=%f\n' % (epoch, i, loss.item()), flush=True)

    save_single_file = "/mnt/data/zhenqiaosong/protein_design/datasets/{}/single.lasso.pt".format(protein)
    save_pair_file = "/mnt/data/zhenqiaosong/protein_design/datasets/{}/pair.lasso.pt".format(protein)
    torch.save(mrf.single_energy.weight, save_single_file)
    torch.save(mrf.pair_energy.weight, save_pair_file)

print('Training done!')



