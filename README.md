# linear_rep_geometry
In our paper, [K. Park, Y. J. Choe, and V. Veitch. (2023). The Linear Representation Hypothesis and the Geometry of Large Language Models](https://arxiv.org/abs/2311.03658){:target="_blank"}, we formalize the linear representation hypothesis of large language models, and define a *causal inner product* that respects the semantic structure of language.

We confirm our theory with LLaMA-2 representations and this repo provides the code for the experiments.

## Data
In [`word_pairs`](word_pairs){:target="_blank"}, each `[___ - ___].txt` has counterfactual pairs of words for each concept. We use them to estimate the unembedding representations.

In [`paired_contexts`](paired_contexts){:target="_blank"}, each `__-__.jsonl` has context samples from Wikipedia in different languages. We use them for the measurement experiment ([`2_measurement.ipynb`](2_measurement.ipynb){:target="_blank"}).

## Requirement
You need to install Python packages `transformers`, `torch`, `numpy`, `seaborn`, `matplotlib`, `json`, and `tqdm` to run the codes. Also, you need some GPUs to implement the code efficiently.

Make a directory `matrices` and run [`store_matrices.py`](store_matrices.py){:target="_blank"} first before you run other jupyter notebooks.

## Experiments
- [**`1_subspace.ipynb`**](1_subspace.ipynb){:target="_blank"} (Section 4.1): We compare the projection of differences between counterfactual pairs (vs random pairs) onto their corresponding concept direction
- [**`2_measurement.ipynb`**](2_measurement.ipynb){:target="_blank"} (Section 4.2): We confirm that the concept direction acts as a linear probe
- [**`3_intervention.ipynb`**](3_intervention.ipynb){:target="_blank"} (Section 4.3): We confirm that the embedding representation changes the target concept, without changing off-target concepts.
- [**`4_heatmap.ipynb`**](4_heatmap.ipynb){:target="_blank"} (Section 4.4): We compare the orthogonality between the unembedding representations for causally separable concepts based on the causal inner product (vs Euclidean and random inner product)
- [**`5_check_independence.ipynb`**](5_check_independence.ipynb){:target="_blank"} (Appendix B): We validate Assumption 1.