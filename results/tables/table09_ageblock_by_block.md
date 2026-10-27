**Table 9. Leave-one-age-block-out performance by block (12 h purge; block ranges and window counts from the fold manifest). 'mean of block values' is the unweighted mean of the five per-block metrics; 'pooled over all held-out windows' is one metric computed over all windows (R2 against their pooled mean age); the two are different quantities**

| block | age range (days) | n windows | boundary type | MAE (days) | RMSE (days) | R2 |
|---|---|---|---|---|---|---|
| 1 | 31.333-69.708 | 839 | backward extrapolation | 33.65 | 42.91 | -15.79 |
| 2 | 69.708-109.792 | 842 | internal gap | 21.53 | 24.69 | -3.51 |
| 3 | 109.792-137.667 | 835 | internal gap | 15.01 | 18.49 | -4.35 |
| 4 | 137.667-160.458 | 840 | internal gap | 13.04 | 17.56 | -5.91 |
| 5 | 160.458-181.708 | 842 | forward extrapolation | 19.36 | 24.55 | -15.71 |
| mean of block values (all 5 blocks) |  | 4,198 |  | 20.52 | 25.64 | -9.06 |
| pooled over all held-out windows (all 5 blocks) |  | 4,198 |  | 20.52 | 27.22 | 0.60 |
| mean of block values (excluding block 1) |  | 3,359 |  | 17.24 | 21.32 | -7.37 |
| pooled over all held-out windows (excluding block 1) |  | 3,359 |  | 17.24 | 21.59 | 0.53 |
