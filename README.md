# linear_rep_geometry
In our paper, [K. Park, Y. J. Choe, and V. Veitch. (2023). The Linear Representation Hypothesis and the Geometry of Large Language Models](https://arxiv.org/abs/2311.03658), we formalize the linear representation hypothesis of large language models, and define a *causal inner product* that respects the semantic structure of language.

We confirm our theory with LLaMA-2 representations and this repo provides the code for the experiments.

## Data
In `word_pairs`, each `[___ - ___].txt` has counterfactual pairs of words for each concepts. We use them to estimate the unembedding representations.

In `paired_contexts`, each `__-__.jsonl` has context samples from Wikipedia. We use them for the measurement experiment (`2_measurement.ipynb`).

## Requirement
You need to install Python packages `transformers`, `torch`, `numpy`, `seaborn`, `matplotlib`, `json`, `tqdm` to run the codes.

You need some GPUs to implement the code efficiently.

Make a directory `matrices` and run `store_matrices.py` first before you run other jupyter notebooks.

## Experiments
- `1_subspace.ipynb` (Section 4.1): We compare the projection of differences between counterfactual pairs (vs random pairs) onto their corresponding concept direction
- `2_measurement.ipynb` (Section 4.2): We confirm that the concept direction acts as a linear probe
- `3_intervention.ipynb` (Section 4.3): We confirm that the embedding representation changes the target concept, without changing off-target concepts.
- `4_heatmap.ipynb` (Section 4.4): We compare the orthogonality between the unembedding representations for causally separable concepts based on the causal inner product (vs Euclidean and random inner product)
- `5_check_independence.ipynb` (Appendix B): We validate Assumption 1.