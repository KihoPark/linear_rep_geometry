# linear_rep_geometry
In our paper, we formalize the linear representation hypothesis of large language models, and define a *causal inner product* that respects the semantic structure of language.

We confirm our theory with LLaMA-2 representations and this repo provides the code for the experiments.

## Data
In [`word_pairs`](word_pairs), each `[___ - ___].txt` has counterfactual pairs of words for each concept. We use them to estimate the unembedding representations.

In [`paired_contexts`](paired_contexts), each `__-__.jsonl` has context samples from Wikipedia in different languages. We use them for the measurement experiment ([`3_measurement.ipynb`](3_measurement.ipynb)).

## Requirement
You need to install Python packages `transformers`, `torch`, `numpy`, `seaborn`, `matplotlib`, `json`, and `tqdm` to run the codes. Also, you need some GPUs to implement the code efficiently.

Make a directory `matrices` and run [`store_matrices.py`](store_matrices.py) first before you run other jupyter notebooks.

## Experiments
- [**`1_subspace.ipynb`**](1_subspace.ipynb): We compare the projection of differences between counterfactual pairs (vs random pairs) onto their corresponding concept direction
- [**`2_heatmap.ipynb`**](2_heatmap.ipynb): We compare the orthogonality between the unembedding representations for causally separable concepts based on the causal inner product (vs Euclidean and random inner product)
- [**`3_measurement.ipynb`**](3_measurement.ipynb): We confirm that the concept direction acts as a linear probe
- [**`4_intervention.ipynb`**](4_intervention.ipynb): We confirm that the embedding representation changes the target concept, without changing off-target concepts.
- [**`5_sanity_check.ipynb`**](5_sanity_check.ipynb): We verify that the causal inner product we find satisfies the uncorrelatedness condition in Assumption 3.3.