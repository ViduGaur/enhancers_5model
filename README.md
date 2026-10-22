Cross-species enhancer classification with k-mer features

Do k-mer enhancer classifiers trained on human DNA transfer to mouse?

We classify DNA sequences as **enhancer** vs. **non-regulatory background** using normalized k-mer frequencies (k = 3–6) and five classical models from scikit-learn: Logistic Regression, KNN, Linear SVM, Random Forest and HistGradientBoosting. Models are tuned with cross-validation on human data only, then tested on mouse without ever tuning on it. The gap between human and mouse ROC-AUC measures how well each model family generalizes across species.

**Data:** [`genomic-benchmarks`](https://github.com/ML-Bioinformatics/genomic-benchmarks) `human_enhancers_ensembl` (train and test) and `dummy_mouse_enhancers_ensembl` (test only). The raw human data has heavy train/test leakage (38% of test rows also appear in train), so the pipeline deduplicates, removes leaked sequences, filters N-heavy mouse sequences and cuts mouse sequences into 270 bp windows.

**Key findings**
- Logistic regression with 5-mers is best within human (AUC 0.815) but drops the most on mouse (gap 0.18).
- Random forest and KNN lose only about 0.03–0.04 AUC, and random forest is the best mouse classifier.
- Longer k-mers help on human but hurt transfer for the linear models.

## Structure

| Path | Purpose |
|---|---|
| `00_verify.py` | Downloads data and runs sanity checks |
| `01_clean.py` | Deduplication, leakage removal, mouse filtering and windowing |
| `run_experiments.py` | CV tuning, human and mouse evaluation, shuffled-label controls |
| `analysis.py` | Result tables, figures, generalization gap, top k-mers |
| `src/` | Config, data loading, cleaning, k-mer features, models, metrics |

## Run

```bash
python -m venv .venv
.venv/bin/pip install numpy pandas scikit-learn matplotlib seaborn genomic-benchmarks tqdm joblib
.venv/bin/python 00_verify.py        # downloads data into ./data
