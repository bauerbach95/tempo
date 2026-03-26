# TEMPO: Bayesian Estimation of Cell-Intrinsic Clocks from Single-Cell RNA-Seq Data

TEMPO is a Bayesian algorithm that estimates the cell-intrinsic circadian phase of each cell in a single-cell RNA-seq (scRNA-seq) dataset. It uses prior knowledge of core circadian clock genes to infer each cell's position in the 24-hour circadian cycle, and can also identify *de novo* cycling genes beyond the known clock.

**Paper:** [Auerbach et al., *Nature Communications* (2022)](https://www.nature.com/articles/s41467-022-34185-w)

## Table of Contents
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Inputs](#inputs)
- [Configuration Parameters](#configuration-parameters)
- [Outputs](#outputs)
- [Working with Results](#working-with-results)
- [Optimizing Performance](#optimizing-performance)
- [Reference Data](#reference-data)
- [Citation](#citation)
- [Tutorials](#tutorials)

---

## How It Works

TEMPO uses a two-step variational inference algorithm:

1. **Step 1 (Cell Phase Estimation):** Using known core clock genes and prior knowledge about their peak expression times (acrophases), TEMPO estimates each cell's circadian phase as a posterior distribution over a discrete grid.

2. **Step 2 (De Novo Cycler Identification):** Using the cell phase estimates from Step 1, TEMPO fits harmonic expression models to all highly variable genes and identifies new cycling genes beyond the core clock.

These steps can iterate (controlled by `max_num_alg_steps`), with newly identified cyclers feeding back into phase estimation. TEMPO uses a Bayes factor to assess whether the data contains a real circadian signal before proceeding.

---

## Installation

### Requirements
- **OS:** macOS (10.14+), Linux (CentOS 7+ / Ubuntu)
- **RAM:** >= 8 GB
- **Python:** >= 3.8
- **Conda:** [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda

### Steps

```bash
git clone https://github.com/bauerbach95/tempo
cd tempo
source install.sh
```

This creates a conda environment called `tempo` and installs all dependencies. Installation should take less than 5 minutes.

**Apple M1/M2 users:** Use the M1-compatible environment file instead:
```bash
conda env create -f m1_tempo.yml -n tempo
conda activate tempo
source install_power_spherical.sh
python setup.py install
```

### Verify installation

```bash
conda activate tempo
python run_test.py
```

A successful test prints `SUCCESSFULLY FINISHED` and should take less than 5 minutes.

---

## Quick Start

### Option 1: Config file (recommended for reproducibility)

```bash
conda activate tempo
python -m tempo.run_tempo \
  -f path/to/adata.h5ad \
  -o path/to/output_folder \
  -c path/to/config.txt
```

The config file is a plain text file containing a Python dictionary. Any parameters not specified use defaults. See [Configuration Parameters](#configuration-parameters) for details. Example config:

```python
{
  "gene_acrophase_prior_path": "data/clock_aorta_acrophase_prior.csv",
  "core_clock_gene_path": "data/core_clock_genes.txt",
  "reference_gene": "Arntl",
  "min_gene_prop": 1e-5,
  "use_nb": True,
  "num_phase_grid_points": 24
}
```

### Option 2: Python API

```python
import anndata
import tempo
from tempo import unsupervised_alg

adata = anndata.read_h5ad("path/to/adata.h5ad")

tempo.unsupervised_alg.run(
    adata=adata,
    folder_out="path/to/output_folder",
    gene_acrophase_prior_path="data/clock_aorta_acrophase_prior.csv",
    core_clock_gene_path="data/core_clock_genes.txt",
    reference_gene="Arntl",
    min_gene_prop=1e-5
)
```

### Running on the test data

```bash
cd tempo
python -m tempo.run_tempo \
  -f test_data/adata.h5ad \
  -o test_output \
  -c test_data/config.txt
```

---

## Inputs

### Required Inputs

#### 1. AnnData object (`.h5ad`)

A [cells x genes] AnnData object containing **raw (unnormalized) transcript counts**. Gene names in `.var_names` must match those in your core clock gene list and acrophase prior file.

#### 2. Core clock gene list (`.txt`)

A plain text file with one gene name per line. These are the core circadian clock genes used in Step 1. TEMPO ships a default mouse list at `data/core_clock_genes.txt` containing 28 genes:

```
Arntl
Clock
Cry1
Cry2
Per1
Per2
Per3
Nr1d1
Nr1d2
Dbp
...
```

Only genes that appear in both this list and your AnnData's `.var_names` will be used.

#### 3. Gene acrophase prior file (`.csv`)

A CSV with prior knowledge about when each clock gene peaks. Columns:

| Column | Description |
|--------|-------------|
| `gene` | Gene name (must match `.var_names` in AnnData) |
| `prior_acrophase_loc` | Peak expression time in **radians** (0 to 2pi) |
| `prior_acrophase_95_interval` | Width of the 95% prior interval in **radians** (smaller = more certain) |

**Converting hours to radians:** `radians = hours * (2 * pi / 24)`. For example, a gene peaking at CT12 has an acrophase of pi (~3.14).

TEMPO ships tissue-specific priors:
- `data/clock_aorta_acrophase_prior.csv` (18 genes, for aorta tissue)
- `data/clock_liver_acrophase_prior.csv` (19 genes, for liver tissue)

Example (aorta):
```
gene,prior_acrophase_loc,prior_acrophase_95_interval
Arntl,0.0,0.2618
Clock,0.0,0.2618
Cry1,4.7124,0.2618
Per2,3.6652,0.2618
Nr1d1,2.0944,0.2618
...
```

#### 4. Reference gene

One core clock gene to anchor the phase (set its prior acrophase to 0 by convention). Default: `Arntl`. This must appear in both your core clock gene list and acrophase prior file.

#### 5. Output folder path

Path where TEMPO writes all results. Created automatically if it doesn't exist.

### Optional Inputs

#### Cell phase prior file (`.csv`)

If you have external timing information (e.g., time-of-death metadata), supply it as a CSV with columns:

| Column | Description |
|--------|-------------|
| `barcode` | Cell barcode (must match `.obs_names` in AnnData) |
| `prior_theta_euclid_cos` | Cosine of the prior phase location |
| `prior_theta_euclid_sin` | Sine of the prior phase location |
| `prior_theta_95_interval` | Width of the 95% prior interval in radians (pi = noninformative) |

If not provided, set `use_noninformative_phase_prior=True` (the default) to use flat priors for all cells.

### Preparing Inputs for Your Own Data

**If you have mouse data for aorta or liver tissue:**
Use the shipped reference files directly:
```python
core_clock_gene_path = "data/core_clock_genes.txt"
gene_acrophase_prior_path = "data/clock_aorta_acrophase_prior.csv"  # or clock_liver_acrophase_prior.csv
reference_gene = "Arntl"
```

**If you have mouse data for a different tissue:**
1. Use `data/core_clock_genes.txt` as your clock gene list (these are broadly conserved).
2. Create a gene acrophase prior CSV using literature values for peak expression times in your tissue. Convert hours (CT) to radians: `radians = hours * 2 * pi / 24`. Set the 95% interval to ~0.26 (1 hour) if you are confident, or ~0.52 (2 hours) if uncertain.
3. Set `reference_gene = "Arntl"`.

**If you have human data:**
Core clock genes are conserved across mammals but gene names may differ (e.g., `ARNTL` vs `Arntl`). Ensure your gene list and acrophase priors use the same gene naming convention as your AnnData object.

---

## Configuration Parameters

### Key Parameters

These are the parameters most users will want to consider adjusting:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gene_acrophase_prior_path` | str | *required* | Path to gene acrophase prior CSV |
| `core_clock_gene_path` | str | *required* | Path to core clock gene list |
| `reference_gene` | str | `"Arntl"` | Core clock gene used to anchor the phase |
| `min_gene_prop` | float | `1e-5` | Minimum proportion of transcripts for a gene to be included |
| `use_nb` | bool | `True` | Use negative binomial likelihood (recommended). If False, uses Poisson |
| `num_phase_grid_points` | int | `24` | Grid points for phase posterior. Use 24 for <10k cells, 12 for >10k cells. Minimum 6 |
| `vi_max_epochs` | int | `300` | Maximum optimization epochs per step |
| `vi_batch_size` | int | `3000` | Cell batch size for optimization. Increase for large datasets |
| `max_num_alg_steps` | int | `3` | Maximum iterations of Steps 1-2. Recommended: 1-3 |
| `use_clock_input_only` | bool | `False` | If True, only run Step 1 with clock genes (skip de novo detection) |
| `use_clock_output_only` | bool | `True` | If True, only use clock genes for the ELBO in Step 1 |

### Amplitude Parameters

| Parameter | Type | Default | Recommended | Description |
|-----------|------|---------|-------------|-------------|
| `min_amp` | float | `0.0` | 0-0.1 | Minimum gene amplitude |
| `max_amp` | float | `~3.26` | 2-3 | Maximum gene amplitude |
| `init_amp_loc_val` | float | `0.5` | [min_amp, max_amp] | Initial variational amplitude location |
| `init_amp_scale_val` | float | `3` | 1-100 | Initial amplitude certainty (higher = more certain) |
| `prior_amp_alpha_val` | float | `1` | 1 | Beta prior alpha for amplitude |
| `prior_amp_beta_val` | float | `1` | 1 | Beta prior beta for amplitude |

### Mesor Parameters

| Parameter | Type | Default | Recommended | Description |
|-----------|------|---------|-------------|-------------|
| `init_mesor_scale_val` | float | `0.3` | 0.1-2.0 | Initial variational mesor scale |
| `prior_mesor_scale_val` | float | `0.5` | 0.1-2.0 | Prior mesor scale |

### Acrophase Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `known_cycler_init_shift_95_interval` | float | `pi/12` | Init acrophase 95% interval for known cyclers (radians) |
| `unknown_cycler_init_shift_95_interval` | float | `pi/12` | Init acrophase 95% interval for unknown cyclers (radians) |
| `known_cycler_prior_shift_95_interval` | float | `pi/6` | Prior acrophase 95% interval for known cyclers. Values in the gene acrophase prior CSV take precedence |

### Cycling Probability (Q) Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `init_clock_Q_prob_alpha` | float | `90` | Init Beta alpha for clock gene cycling probability |
| `init_clock_Q_prob_beta` | float | `10` | Init Beta beta for clock gene cycling probability |
| `init_non_clock_Q_prob_alpha` | float | `1` | Init Beta alpha for non-clock gene cycling probability |
| `init_non_clock_Q_prob_beta` | float | `9` | Init Beta beta for non-clock gene cycling probability |
| `prior_clock_Q_prob_alpha` | float | `90` | Prior Beta alpha for clock gene cycling probability |
| `prior_clock_Q_prob_beta` | float | `10` | Prior Beta beta for clock gene cycling probability |
| `prior_non_clock_Q_prob_alpha` | float | `1` | Prior Beta alpha for non-clock gene cycling probability |
| `prior_non_clock_Q_prob_beta` | float | `1` | Prior Beta beta for non-clock gene cycling probability |

### Learning Rate Parameters

All default to `0.1`. Recommended range: `1e-3` to `1e-1`.

| Parameter | Description |
|-----------|-------------|
| `mu_loc_lr` | Mesor location |
| `mu_log_scale_lr` | Mesor scale |
| `A_log_alpha_lr` | Amplitude alpha |
| `A_log_beta_lr` | Amplitude beta |
| `phi_euclid_loc_lr` | Acrophase location |
| `phi_log_scale_lr` | Acrophase scale |
| `Q_prob_log_alpha_lr` | Cycling probability alpha |
| `Q_prob_log_beta_lr` | Cycling probability beta |

### Optimization Parameters

| Parameter | Type | Default | Recommended | Description |
|-----------|------|---------|-------------|-------------|
| `vi_print_epoch_loss` | bool | `True` | -- | Print ELBO at each epoch |
| `vi_improvement_window` | int | `10` | 3-10 | Epoch window for convergence check |
| `vi_convergence_criterion` | float | `1e-3` | 1e-4 to 1e-1 | ELBO improvement threshold for convergence |
| `vi_lr_scheduler_patience` | int | `10` | 3-10 | Epochs before reducing learning rate |
| `vi_lr_scheduler_factor` | float | `0.1` | 0.01-0.1 | Learning rate reduction factor |

### Monte Carlo Sampling Parameters

All default to `3`. Recommended range: 1-3.

| Parameter | Description |
|-----------|-------------|
| `num_phase_est_cell_samples` | Cell phase samples in Step 1 |
| `num_phase_est_gene_samples` | Gene parameter samples in Step 1 |
| `num_harmonic_est_cell_samples` | Cell phase samples in Step 2 |
| `num_harmonic_est_gene_samples` | Gene parameter samples in Step 2 |

### De Novo Cycler Detection Parameters

| Parameter | Type | Default | Recommended | Description |
|-----------|------|---------|-------------|-------------|
| `hv_std_residual_threshold` | float | `0.5` | 0-3 | Pearson residual threshold for highly variable gene selection |
| `frac_pos_cycler_samples_threshold` | float | `0.95` | 0.8-0.99 | MAP threshold for cycling probability to call a de novo cycler |
| `A_loc_pearson_residual_threshold` | float | `1.0` | 1-5 | Amplitude Pearson residual threshold for de novo cyclers |
| `confident_cell_interval_size_threshold` | float | `12.0` | -- | Max 95% interval size (hours) for a cell to be used in Step 2 |

### Negative Binomial Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mean_disp_init_coef` | list | `[-4, -0.2]` | Initial log-proportion to log-dispersion polynomial coefficients |
| `est_mean_disp_relationship` | bool | `True` | Whether to optimize the dispersion relationship |
| `mean_disp_max_num_genes_per_bin` | int | `50` | Max genes per bin for dispersion estimation (recommended: 10-50) |

### Other Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_noninformative_phase_prior` | bool | `True` | Use flat cell phase priors (if True, cell_phase_prior_path priors still take precedence) |
| `opt_phase_est_gene_params` | bool | `True` | Optimize gene distributions in Step 1 (if False, uses priors directly) |
| `init_variational_dist_to_prior` | bool | `False` | Reset gene distributions to priors at each step |
| `log10_bf_tempo_vs_null_threshold` | float | `~0.176` | Log10 Bayes factor threshold to continue (default: log10(1.5)) |
| `use_de_novo_cycler_detection` | bool | `True` | Whether to run Step 2 |
| `test_mode` | bool | `False` | Enable PyTorch profilers (slows computation) |

---

## Outputs

TEMPO writes all results to the user-specified output folder with this structure:

```
output_folder/
|-- config.txt                          # Parameters used for this run
|-- run_time.txt                        # Total wall-clock time
|-- evidence/
|   |-- null_log_evidence_vec.txt       # Random (null) clock evidence
|   |-- step_1_clock_evidence.txt       # Clock evidence at iteration 1
|   |-- step_2_clock_evidence.txt       # Clock evidence at iteration 2 (if applicable)
|-- mean_disp_param/
|   |-- log_mean_log_disp_poly_coef_0.txt  # Dispersion relationship coefficients
|-- tempo_results/
    |-- 0/                              # Raw results for iteration 0
    |   |-- cell_phase_estimation/
    |   |   |-- cell_posterior.tsv       # Cell phase posteriors
    |   |   |-- cell_posterior_init.tsv  # Cell phase posteriors before gene param optimization
    |   |   |-- cell_prior.tsv          # Cell priors used
    |   |   |-- gene_prior_and_posterior.tsv
    |   |   |-- gene_prior_and_posterior_init.tsv
    |   |   |-- loss.txt, kl_loss.txt, ll_loss.txt
    |   |-- de_novo_cycler_id/
    |       |-- gene_prior_and_posterior.tsv
    |       |-- loss.txt, kl_loss.txt, ll_loss.txt
    |-- 1/                              # Iteration 1 (if applicable)
    |   |-- ...
    |-- opt/                            # Final optimal results
        |-- cell_posterior.tsv
        |-- cycler_gene_prior_and_posterior.tsv
        |-- flat_gene_prior_and_posterior.tsv
```

### Key Output Files

#### `tempo_results/opt/cell_posterior.tsv`

The main cell-level result. Each row is a cell, each column (`bin_0` through `bin_{N-1}`) is the posterior density at that phase grid point. Phase grid points are evenly spaced from 0 to 2pi.

| barcode | bin_0 | bin_1 | ... | bin_23 |
|---------|-------|-------|-----|--------|
| cell_1  | 0.001 | 0.003 | ... | 0.001  |

#### `tempo_results/opt/cycler_gene_prior_and_posterior.tsv`

Gene-level results for all cycling genes (core clock + de novo). Key columns:

| Column | Description |
|--------|-------------|
| `mu_loc` | Posterior mesor (log-scale mean expression) |
| `A_loc` | Posterior amplitude (effect size of oscillation) |
| `phi_loc` | Posterior acrophase in radians (peak expression phase) |
| `Q_prob_loc` | Posterior probability of non-zero amplitude (cycling probability) |
| `phi_euclid_cos`, `phi_euclid_sin` | Euclidean coordinates of posterior acrophase |
| `phi_scale` | Posterior acrophase concentration (higher = more certain) |

#### `evidence/step_{p}_clock_evidence.txt`

The log10 Bayes factor comparing TEMPO's model to a null (random phases) model at iteration p. Higher values indicate stronger circadian signal. Values above `log10(1.5) ~ 0.176` (the default threshold) indicate the data has detectable circadian structure.

---

## Working with Results

### Loading cell phase posteriors and getting MAP phases

```python
import pandas as pd
import numpy as np
import torch
from tempo import cell_posterior

# Load cell posterior
cell_posterior_df = pd.read_table(
    "output_folder/tempo_results/opt/cell_posterior.tsv",
    sep='\t', index_col='barcode'
)

# Create posterior distribution object
cell_posterior_obj = cell_posterior.ThetaPosteriorDist(
    torch.Tensor(np.array(cell_posterior_df))
)

# Get MAP (maximum a posteriori) phase for each cell
map_phases = cell_posterior_obj.map_phase  # tensor of phases in radians [0, 2*pi)
```

### Computing confidence intervals

```python
# Get 95% credible intervals
confidence_intervals = cell_posterior_obj.compute_confidence_interval(confidence=0.95)
# Returns: [num_cells x num_grid_points] boolean array

# Get interval sizes in hours (0-24)
interval_sizes = np.sum(confidence_intervals, axis=1)
```

### Visualizing a single cell's posterior

```python
import matplotlib.pyplot as plt

cell_index = 0
phase_grid = cell_posterior_obj.phase_grid.numpy()
density = cell_posterior_obj.theta_posterior_likelihood[cell_index, :].numpy()

plt.scatter(phase_grid, density)
plt.xlabel("Phase (radians)")
plt.ylabel("Posterior density")
plt.title(f"Cell {cell_posterior_df.index[cell_index]}")
plt.show()
```

### Polar plot of cell phases

```python
map_phases_np = map_phases.numpy()

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.hist(map_phases_np, bins=48, density=True)
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)  # clockwise

# Add hour labels
hour_ticks = np.linspace(0, 2 * np.pi, 24, endpoint=False)
ax.set_xticks(hour_ticks)
ax.set_xticklabels([f"CT{h}" for h in range(24)])
ax.set_title("Cell Phase Distribution")
plt.show()
```

### Identifying de novo cycling genes

```python
# Load cycling gene results
gene_df = pd.read_table(
    "output_folder/tempo_results/opt/cycler_gene_prior_and_posterior.tsv",
    sep='\t', index_col='gene'
)

# All cycling genes (core clock + de novo)
print("Cycling genes:", gene_df.index.tolist())

# Gene acrophases (peak expression phases)
print(gene_df[['phi_loc', 'A_loc', 'Q_prob_loc']])

# Convert acrophase from radians to circadian hours
gene_df['peak_hour'] = gene_df['phi_loc'] * 24 / (2 * np.pi)
```

---

## Optimizing Performance

### Getting a good Bayes factor / clock evidence

The Bayes factor (`evidence/step_*_clock_evidence.txt`) indicates how strong the circadian signal is. If TEMPO's Bayes factor does not exceed the threshold, the algorithm halts early. Tips for improving it:

1. **Use raw counts.** Do not normalize or log-transform your data. TEMPO models raw transcript counts internally.

2. **Include enough core clock genes.** More known cycling genes give TEMPO more signal. Ensure the genes in your clock list and acrophase prior file are actually expressed in your data (check `.var_names`).

3. **Provide accurate acrophase priors.** If your tissue's clock gene peak times differ from the shipped priors, create custom priors from literature values. Inaccurate priors can hurt performance.

4. **Use the right reference files for your tissue.** Liver and aorta have different clock gene expression patterns. Use tissue-specific priors when available.

5. **Have enough cells.** More cells give more statistical power. Datasets with fewer than ~500 cells may have weak signal.

6. **Adjust grid resolution.** For small datasets (<10k cells), use `num_phase_grid_points=24`. For large datasets, `12` is sufficient and faster.

### Computational efficiency

- Reduce `num_phase_grid_points` from 24 to 12 for datasets >10k cells
- Increase `vi_batch_size` for large datasets
- Set `confident_cell_interval_size_threshold` to a smaller value (e.g., 6) to exclude uncertain cells from Step 2
- Set `use_clock_input_only=True` if you only need cell phases (skip de novo detection)

---

## Reference Data

The `data/` directory contains reference files for various tissues and species:

### Core Clock Gene Lists

| File | Description |
|------|-------------|
| `core_clock_genes.txt` | 28 mouse core clock genes (default, broadly applicable) |
| `core_clock_genes_alt.txt` | Alternative mouse core clock gene list |
| `core_clock_genes_without_rorc.txt` | Mouse clock genes excluding Rorc |

### Tissue-Specific Acrophase Priors

| File | Tissue | Genes |
|------|--------|-------|
| `clock_aorta_acrophase_prior.csv` | Mouse aorta | 18 genes |
| `clock_liver_acrophase_prior.csv` | Mouse liver | 19 genes |
| `core_clock_and_ubiq_acrophase_prior.csv` | Core clock + ubiquitous cyclers | -- |

### Cell Cycle Gene References

| File | Description |
|------|-------------|
| `cell_cycle_mouse_genes.txt` | Mouse cell cycle genes |
| `cell_cycle_mouse_acrophases.csv` | Cell cycle gene acrophases |
| `regev_cell_cycle_genes.txt` | Regev lab cell cycle gene list |

### Tissue-Specific Cycler Lists

| File | Description |
|------|-------------|
| `liver_cyclers.csv` | Known liver cycling genes |
| `neuron_cyclers.txt` | Known neuron cycling genes |
| `BHTC_cyclers.csv` | Brain, heart, thymus, and colon cycling genes |
| `Human_UbiquityCyclers.csv` | Human ubiquitous cycling genes |

---

## Citation

If you use TEMPO, please cite:

> Auerbach, B.J., Oh, J., Gee, S.E. *et al.* TEMPO: unsupervised Bayesian estimation of cell-intrinsic clocks from single-cell RNA-seq data. *Nat Commun* **13**, 6319 (2022). https://doi.org/10.1038/s41467-022-34185-w

---

## Tutorials

Detailed tutorial notebooks are in the `tutorial/` folder:

| Notebook | Description |
|----------|-------------|
| [tutorial_running_tempo.ipynb](tutorial/tutorial_running_tempo.ipynb) | How to run TEMPO (config file and Python API) |
| [tutorial_tempo_inputs.ipynb](tutorial/tutorial_tempo_inputs.ipynb) | Detailed description of all input parameters |
| [tutorial_tempo_outputs.ipynb](tutorial/tutorial_tempo_outputs.ipynb) | Understanding and working with TEMPO outputs |
| [tutorial_real_data.ipynb](tutorial/tutorial_real_data.ipynb) | End-to-end example using real mouse aorta SMC data from the paper |
