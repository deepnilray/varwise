# Real Gaia/ZTF training archive — 27 October 2026

This experiment actually fitted a classifier on **110** real ZTF light curves, then opened **28** sky-disjoint validation curves for the first time. The model achieved **AUC 0.568**, virtually identical to the **0.569** shuffled-label control: **failed validation**.

The exact feature CSV files and successful light-curve cache are preserved as downloadable originals. They are **not currently committed as raw data to this repository**; the binary-file workspace is unavailable, and parsed CSV previews are not lossless copies.

- [Original training_features.csv](https://chatgpt.com/library?file_id=libfile_b8e6604241b88191b11cdfa9d826db7b&account_id=personal)
- [Original sealed_validation_features.csv](https://chatgpt.com/library?file_id=libfile_c751ca3b954c819191c9aa9cde503082&account_id=personal)
- [Successful Gaia/ZTF lookup cache ZIP](https://chatgpt.com/library?file_id=libfile_cfceced574d08191be96cfb4080e97da&account_id=personal)
- [Invalid research-only model joblib](https://chatgpt.com/library?file_id=libfile_a31bc1304e688191a6d1b8aa402abb64&account_id=personal)

Some Library links may require the project owner's ChatGPT account.

## Reproduce the research-only fitted model

1. Place the original feature CSVs and `validation_report.json` in `verification/strict_training_real_20261027/`.
2. Install `requirements.txt`.
3. Run:

```bash
python verification/export_research_only_model.py
```

The export script re-fits on **training rows only**, checks the old holdout strictly **after** fitting, reproduces the failed AUC, and asserts that the discovery loader refuses the model.

**Do not tune on the now-open validation set and call it independent. Use new sealed sky regions for future models.**
