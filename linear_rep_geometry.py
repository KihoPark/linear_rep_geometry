import transformers
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import json
from tqdm import tqdm

device = torch.device("cuda:0")

sns.set_theme(
    context="paper",
    style="white",  # 'whitegrid', 'dark', 'darkgrid', ...
    palette="colorblind",
    font="sans-serif",  # 'serif'
    font_scale=1.75,  # 1.75, 2, ...
)

# MODEL_PATH = ### Path where the weights for LLaMA-2-7B are stored ###
tokenizer = transformers.LlamaTokenizer.from_pretrained(MODEL_PATH)
model = transformers.LlamaForCausalLM.from_pretrained(MODEL_PATH, low_cpu_mem_usage=True, device_map="auto")



## get indices of counterfactual pairs
def get_counterfactual_pairs(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    words_pairs = [line.strip().split('\t') for line in lines if line.strip()]

    base_ind = []
    target_ind = []

    for i in range(len(words_pairs)):
        first = tokenizer.encode(words_pairs[i][0])
        second = tokenizer.encode(words_pairs[i][1])
        if len(first) == len(second) == 2 and first[1] != second[1]:
            base_ind.append(first[1])
            target_ind.append(second[1])
    base_name = [tokenizer.decode(i) for i in base_ind]
    target_name = [tokenizer.decode(i) for i in target_ind]

    return base_ind, target_ind, base_name, target_name

## get concept direction
def concept_direction(base_ind, target_ind, data):
    base_data = data[base_ind,]; target_data = data[target_ind,]

    diff_data = target_data - base_data
    mean_diff_data = torch.mean(diff_data, dim = 0)
    mean_diff_data = mean_diff_data / torch.norm(mean_diff_data)

    return mean_diff_data, diff_data

## get embeddings of each text
def get_embeddings(text_batch):
    tokenizer.pad_token = tokenizer.eos_token
    tokenized_output = tokenizer(text_batch, return_tensors="pt", padding=True, truncation=True).to(device)
    input_ids = tokenized_output["input_ids"]

    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states = True)
    hidden_states = outputs.hidden_states

    seq_lengths = tokenized_output.attention_mask.sum(dim=1).tolist()
    last_token_positions = [length - 1 for length in seq_lengths]
    text_embeddings = torch.stack([hidden_states[-1][i, pos, :] for i, pos in enumerate(last_token_positions)])

    return text_embeddings






####### Experiment 1: subspace #######
## get projection for leave-one-out estimate
def inner_product_loo(base_ind, target_ind, data):
    base_data = data[base_ind,]; target_data = data[target_ind,]

    diff_data = target_data - base_data
    products = []
    for i in range(diff_data.shape[0]):
        mask = torch.ones(diff_data.shape[0], dtype = bool)
        mask[i] = False
        loo_diff = diff_data[mask]
        mean_diff_data = torch.mean(loo_diff, dim = 0)
        loo_mean = mean_diff_data / torch.norm(mean_diff_data)
        products.append(loo_mean @ diff_data[i])
    return torch.stack(products), diff_data

## show the histogram comparing the projections
def show_histogram_LOO(inner_product_with_counterfactual_pairs_LOO,
                        random_pairs, concept, concept_names, fig_name = "gamma"):
    fig, axs = plt.subplots(7, 4, figsize=(16, 20))

    axs = axs.flatten()

    for i in range(concept.shape[0]):
        target = inner_product_with_counterfactual_pairs_LOO[i]
        baseline = random_pairs @ concept[i]

        axs[i].hist(baseline.cpu().numpy(), bins=50, alpha=0.6, color = 'blue', label='random pairs', density=True)
        axs[i].hist(target.cpu().numpy(), alpha=0.7, color = 'red', label='counterfactual pairs', density=True)
        axs[i].set_yticks([])
        axs[i].set_title(concept_names[i])

    handles, labels = axs[0].get_legend_handles_labels()
    axs[concept.shape[0]].legend(handles, labels, loc='center')
    axs[concept.shape[0]].axis('off')

    plt.tight_layout()
    plt.savefig("figures/appendix_right-skewed_LOO_" + fig_name + ".pdf", bbox_inches='tight')

    plt.show()




####### Experiment 2: heatmap #######
## draw heatmap of the inner products
def draw_heatmaps(data_matrices, concept_labels, cmap = 'PiYG'):
    fig = plt.figure(figsize=(14, 8.5))
    gs = gridspec.GridSpec(2, 3, wspace=0.2)
    
    vmin = min([data.min() for data in data_matrices])
    vmax = max([data.max() for data in data_matrices])
    
    ticks = list(range(2, 27, 3))
    labels = [str(i+1) for i in ticks]
    
    ytick = list(range(27))
    ims = []

    ax_left = plt.subplot(gs[0:2, 0:2])
    im = ax_left.imshow(data_matrices[0], cmap=cmap)
    ims.append(im)
    ax_left.set_xticks(ticks)
    ax_left.set_xticklabels(labels)
    ax_left.set_yticks(ytick)
    ax_left.set_yticklabels(concept_labels)
    ax_left.set_title(r'$M = \mathrm{Cov}(\gamma)^{-1}$')

    ax_top_right = plt.subplot(gs[0, 2])
    im = ax_top_right.imshow(data_matrices[1], cmap=cmap)
    ims.append(im)
    ax_top_right.set_xticks([])
    ax_top_right.set_yticks([])
    ax_top_right.set_title(r'$M = I_d$')

    ax_bottom_right = plt.subplot(gs[1, 2])
    im = ax_bottom_right.imshow(data_matrices[2], cmap=cmap)
    ims.append(im)
    ax_bottom_right.set_xticks([])
    ax_bottom_right.set_yticks([])
    ax_bottom_right.set_title(r'Random $M$')
    
    divider = make_axes_locatable(ax_left)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    cbar = plt.colorbar(ims[-1], cax=cax, orientation='vertical')

    plt.tight_layout()
    plt.savefig(f"figures/three_heatmaps.pdf", bbox_inches='tight')
    plt.show()




####### Experiment 3: measurement #######
## get lambda pairs for diffrenet languages
def get_lambda_pairs(filename, num_eg = 20):
    lambdas_0 = []
    lambdas_1 = []

    count =0
    with open(filename, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            data = json.loads(line)
            if count >= num_eg:
                break

            text_0 = [s.strip(" " + data['word0']) for s in data['contexts0']]
            lambdas_0.append(get_embeddings(text_0))

            text_1 = [s.strip(" " + data['word1']) for s in data['contexts1']]
            lambdas_1.append(get_embeddings(text_1))
            
            count += 1

    return torch.cat(lambdas_0), torch.cat(lambdas_1)

## show histogram of lambda^T gamma_W
def hist_measurement(lambda_0, lambda_1, concept, concept_names,
                    base = "English", target = "French", alpha = 0.5):
    fig, axs = plt.subplots(7, 4, figsize=(16, 20))

    axs = axs.flatten()

    for i in range(concept.shape[0]):
        W0 = lambda_0 @ concept[i]
        W1 = lambda_1 @ concept[i]

        axs[i].hist(W0.cpu().numpy(), bins = 25, alpha=alpha, label=base, density=True)
        axs[i].hist(W1.cpu().numpy(), bins = 25, alpha=alpha, label=target,  density=True)
        axs[i].set_yticks([])
        axs[i].set_title(f'{concept_names[i]}')

    handles, labels = axs[0].get_legend_handles_labels()
    axs[concept.shape[0]].legend(handles, labels, loc='center')
    axs[concept.shape[0]].axis('off')

    plt.tight_layout()
    plt.savefig("figures/appendix_measurement_"+ base + "-" + target + ".pdf", bbox_inches='tight')
    plt.show()





####### Experiment 4: intervention #######
## get the difference between the conditional probability of words
def get_logit(embedding, unembedding, base = "king", W = "queen", Z = "King") :
    num = embedding.shape[0]
    logit = torch.zeros(num, 2)
    for i in range(num):
        index_base = tokenizer.encode(base)[1]
        index_W = tokenizer.encode(W)[1]
        index_Z = tokenizer.encode(Z)[1]
        value = unembedding @ embedding[i]
        logit[i, 0] = value[index_W] - value[index_base]
        logit[i, 1] = value[index_Z] - value[index_base]
    return logit

## draw change in the logits
def show_arrows(logit_original, logit_intervened_l, concept_names,
                 base = "king", W = "queen", Z = "King",
                 xlim =[-15, 5], ylim =[-15, 7], fig_name = "gamma"):
    fig, axs = plt.subplots(7, 4, figsize=(16, 20))

    axs = axs.flatten()

    for i in range(len(concept_names)):
        origin = logit_original.numpy()
        vectors_A = logit_intervened_l[i].numpy() - logit_original.numpy()
        
        axs[i].quiver(*origin.T, vectors_A[:, 0], vectors_A[:, 1], color='b', angles='xy', scale_units='xy', scale=1, label='intervened lambda', alpha = 1)
    
        axs[i].set_xlim(xlim)
        axs[i].set_ylim(ylim)
        axs[i].grid(True, linestyle='--', alpha=0.7)
        axs[i].set_title(f'{concept_names[i]}')

    handles, labels = axs[0].get_legend_handles_labels()
    axs[len(concept_names)].legend(handles, labels, loc='center')
    axs[len(concept_names)].set_yticklabels([])
    axs[len(concept_names)].set_xticklabels([])

    plt.xlabel(rf"$\log\frac{{\mathbb{{P}}({W}\mid\lambda)}}{{\mathbb{{P}}({base}\mid \lambda)}}$")
    plt.ylabel(rf"$\log\frac{{\mathbb{{P}}({Z}\mid \lambda)}}{{\mathbb{{P}}({base}\mid \lambda)}}$")

    plt.tight_layout()
    plt.savefig("figures/appendix_intervention_" + fig_name + "_" + base + "_" + W  + "_" + Z + ".pdf", bbox_inches='tight')
    plt.show()

def show_intervention(embedding_batch, unembedding, concept, concept_names,
                        base = "king", W = "queen", Z = "King",
                        alpha = 0.5, xlim =[-15, 15], ylim =[-10, 10], fig_name = "gamma"):
    logit_original = get_logit(embedding_batch , unembedding, base = base, W = W, Z = Z)
    logit_intervened_embedding = []

    for i in range(len(concept_names)):
        intervened_embedding = embedding_batch + alpha * concept[i]
        logit_intervened_embedding.append(get_logit(intervened_embedding, unembedding, base = base, W = W, Z = Z))
    show_arrows(logit_original, logit_intervened_embedding, concept_names, 
                base = base, W = W, Z = Z,
                xlim = xlim, ylim = ylim, fig_name = fig_name)

## show the rank of tokens for a text
def show_rank(text_batch, l_batch, g, concept_g, which_ind, concept_number):
    alphas = torch.linspace(0, 0.4, 5)
    print("Prompt:", text_batch[which_ind])
    print("=" * 40)
    l_king = l_batch[which_ind]

    top_k = 5

    top_tokens = [[] for _ in alphas]
    for i in range(len(alphas)):
        new_lambda = l_king + alphas[i] * concept_g[concept_number]
        value = g @ new_lambda
        norm_values, norm_indices = torch.topk(value, k=top_k, largest=True)
        for j in range(top_k):
            top_tokens[i].append(tokenizer.decode(norm_indices[j]))

    print("  & ", " & ".join([str(alpha) for alpha in alphas.numpy()]))
    for j in range(top_k):
        print(j + 1, "&", " & ".join([token_row[j] for token_row in top_tokens]))





####### Appendix: Sanity check for the estimated causal inner product #######
def sanity_check(g, concept_g, a_i, b_i, c_i, d_i, concept_names,
                 alpha = 0.3, s = 0.8, name_1 = [], ind_1 = [], name_2 = [], ind_2 = []):
    a_g = g @ concept_g[a_i]
    b_g = g @ concept_g[b_i]
    c_g = g @ concept_g[c_i]
    d_g = g @ concept_g[d_i]

    fig, axs = plt.subplots(1, 2, figsize=(12,6))
    
    axs[0].scatter(a_g.cpu().numpy(), b_g.cpu().numpy(), alpha = alpha, s = s)
    for _, label in enumerate(name_1):
        axs[0].text(a_g[ind_1[_]], b_g[ind_1[_]], label, fontsize = 12)
    axs[0].set_xlabel(r"$\bar{\lambda}_W^\top \gamma$")
    axs[0].set_ylabel(r"$\bar{\lambda}_Z^\top \gamma$")
    axs[0].set_title(f"W: {concept_names[a_i]}, Z: {concept_names[b_i]}", x = 0.48, y=1.02)
    
    axs[1].scatter(c_g.cpu().numpy(), d_g.cpu().numpy(), alpha = alpha, s = s)
    for _, label in enumerate(name_2):
        axs[1].text(c_g[ind_2[_]], d_g[ind_2[_]], label, fontsize = 12)
    axs[1].set_xlabel(r"$\bar{\lambda}_W^\top \gamma$")
    axs[1].set_ylabel(r"$\bar{\lambda}_Z^\top \gamma$")
    axs[1].set_title(f"W: {concept_names[c_i]}, Z: {concept_names[d_i]}", x = 0.48, y=1.02)

    axs[0].axhline(0, color='gray', linestyle='--', alpha = 0.6)
    axs[0].axvline(0, color='gray', linestyle='--', alpha = 0.6)
    axs[1].axhline(0, color='gray', linestyle='--', alpha = 0.6)
    axs[1].axvline(0, color='gray', linestyle='--', alpha = 0.6)

    plt.tight_layout()
    plt.savefig("figures/sanity_check.png", dpi=300, bbox_inches='tight')
    plt.show()

