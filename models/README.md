# Models — quarantined experiments only

**No model is approved for astronomical discovery.** The real 27 October 2026 model was fitted on 110 ZTF light curves and failed an independent 28-object holdout (ROC AUC **0.568** versus **0.569** for the shuffled control).

A fitted research artifact is preserved here: [model_RESEARCH_ONLY_FAILED_VALIDATION.joblib](https://chatgpt.com/library?file_id=libfile_a31bc1304e688191a6d1b8aa402abb64&account_id=personal).

That binary is **not currently committed to GitHub**. It carries `validated_for_discovery=False`; the survey's model loader must reject it. Do not rename or pass it off as a validated detector.

A different model must be trained from better controls and evaluated on entirely **new, sealed sky regions** before use.
