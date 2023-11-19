import torch
import numpy as np
import transformers
from tqdm import tqdm
import linear_rep_geometry as lrg

device = torch.device("cuda:0")

### load unembdding vectors ###
gamma = lrg.model.lm_head.weight.detach().to(device)
W, d = gamma.shape
gamma_bar = torch.mean(gamma, dim = 0)
centered_gamma = gamma - gamma_bar

### compute Cov(gamma) and tranform gamma to g ###
Cov_gamma = centered_gamma.T @ centered_gamma / W
eigenvalues, eigenvectors = torch.linalg.eigh(Cov_gamma)
inv_sqrt_Cov_gamma = eigenvectors @ torch.diag(1/torch.sqrt(eigenvalues)) @ eigenvectors.T
sqrt_Cov_gamma = eigenvectors @ torch.diag(torch.sqrt(eigenvalues)) @ eigenvectors.T
g = gamma @ inv_sqrt_Cov_gamma

### compute concept directions ###
filenames = ['word_pairs/[verb - 3pSg].txt',
             'word_pairs/[verb - Ving].txt',
             'word_pairs/[verb - Ved].txt',
             'word_pairs/[Ving - 3pSg].txt',
             'word_pairs/[Ving - Ved].txt',
             'word_pairs/[3pSg - Ved].txt',
             'word_pairs/[verb - V + able].txt',
             'word_pairs/[verb - V + er].txt',
             'word_pairs/[verb - V + tion].txt',
             'word_pairs/[verb - V + ment].txt',
             'word_pairs/[adj - un + adj].txt',
             'word_pairs/[adj - adj + ly].txt',
             'word_pairs/[small - big].txt',
             'word_pairs/[thing - color].txt',
             'word_pairs/[thing - part].txt',
             'word_pairs/[country - capital].txt',
             'word_pairs/[pronoun - possessive].txt',
             'word_pairs/[male - female].txt',
             'word_pairs/[lower - upper].txt',
             'word_pairs/[noun - plural].txt',
             'word_pairs/[adj - comparative].txt',
             'word_pairs/[adj - superlative].txt',
             'word_pairs/[frequent - infrequent].txt',
             'word_pairs/[English - French].txt',
             'word_pairs/[French - German].txt',
             'word_pairs/[French - Spanish].txt',
             'word_pairs/[German - Spanish].txt'
             ]

concept_names = []

for name in filenames:
    content = name.split("/")[1].split(".")[0][1:-1]
    parts = content.split(" - ")
    concept_names.append(r'${} \Rightarrow {}$'.format(parts[0], parts[1]))

concept_gamma = torch.zeros(len(filenames), d)
concept_g = torch.zeros(len(filenames), d)

count = 0
for filename in filenames:
    base_ind, target_ind, base_name, target_name = lrg.get_counterfactual_pairs(filename)

    mean_diff_gamma, diff_gamma = lrg.concept_direction(base_ind, target_ind, gamma)
    concept_gamma[count] = mean_diff_gamma

    mean_diff_g, diff_g = lrg.concept_direction(base_ind, target_ind, g)
    concept_g[count] = mean_diff_g

    count += 1

### save everything ###
torch.save(gamma, "matrices/gamma.pt")
torch.save(g, "matrices/g.pt")
torch.save(sqrt_Cov_gamma, "matrices/sqrt_Cov_gamma.pt")
torch.save(concept_gamma, "matrices/concept_gamma.pt")
torch.save(concept_g, "matrices/concept_g.pt")

with open('matrices/concept_names.txt', 'w') as f:
    for item in concept_names:
        f.write(f"{item}\n")
        
with open('matrices/filenames.txt', 'w') as f:
    for item in filenames:
        f.write(f"{item}\n")