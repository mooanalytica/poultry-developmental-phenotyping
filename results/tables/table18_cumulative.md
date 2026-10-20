**Table 18. Cumulative effect of each change on leave-one-room-out MAE (days), age-block (12 h purge) and seasonal MAE; Steps 0-2 use the row-based slope, Step 3 the time-based slope**

| stage | model | LORO MAE room2 | LORO MAE room6 | LORO MAE room7 | LORO MAE macro | LORO R2 macro | age-block pooled MAE | age-block excl. block 1 | season summer->fall MAE | season fall->summer MAE |
|---|---|---|---|---|---|---|---|---|---|---|
| Step 0 (centred filter, row slope, anchor purge) | fused | 14.97 | 17.00 | 11.16 | 14.38 | 0.792 | 17.47 | 15.02 | 62.18 | 41.35 |
| Step 0 (centred filter, row slope, anchor purge) | morph_only | 16.63 | 15.53 | 11.57 | 14.58 | 0.794 |  |  |  |  |
| Step 1 (past-only filter, row slope, anchor purge) | fused | 15.06 | 17.76 | 11.11 | 14.64 | 0.782 | 17.50 | 15.22 | 62.59 | 40.47 |
| Step 1 (past-only filter, row slope, anchor purge) | morph_only | 18.43 | 14.84 | 11.20 | 14.82 | 0.784 |  |  |  |  |
| Step 2 (+ 12 h window purge, age-block and seasonal) | fused | 15.06 | 17.76 | 11.11 | 14.64 | 0.782 | 18.09 | 15.73 | 61.91 | 42.58 |
| Step 2 (+ 12 h window purge, age-block and seasonal) | morph_only | 18.43 | 14.84 | 11.20 | 14.82 | 0.784 |  |  |  |  |
| Step 3 = primary (time-based slope, calendar-bounded) | fused | 18.02 | 23.64 | 14.08 | 18.58 | 0.627 | 20.52 | 17.24 | 63.71 | 47.38 |
| Step 3 = primary (time-based slope, calendar-bounded) | morph_only | 19.86 | 17.72 | 13.70 | 17.09 | 0.690 |  |  |  |  |
| V2 (audio first-week reference), labelled extension | fused | 16.16 | 23.76 | 13.63 | 17.85 | 0.653 | 20.29 | 17.00 | 63.14 | 47.01 |
| V3 (body-size-only comparator), labelled extension | V3 | 25.99 | 20.65 | 14.21 | 20.28 | 0.594 | 22.28 | 19.24 | 62.02 | 46.48 |
