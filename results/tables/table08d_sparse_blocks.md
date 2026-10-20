**Table 8d. Non-overlapping 7-day blocks (4-of-7-day rule): block counts, AUC and sensitivity; growth cells are not estimable**

| setting | blocks (pos) | detail | AUC | sensitivity | achieved specificity | estimable |
|---|---|---|---|---|---|---|
| all_sensor | growth (<=70 d) | k=5 d | rule=same_hour | 8 | n_blocks_neg=8; per room 1/3/4; estimable=False | 0.984 | 0.500 | 0.750 | not estimable (<10 blocks) |
| all_sensor | growth (<=70 d) | k=10 d | rule=same_hour | 7 | n_blocks_neg=7; per room 1/2/4; estimable=False | 1.000 | 1.000 | 0.857 | not estimable (<10 blocks) |
| all_sensor | growth (<=70 d) | k=15 d | rule=same_hour | 4 | n_blocks_neg=4; per room 0/1/3; estimable=False | 1.000 | 1.000 | 1.000 | not estimable (<10 blocks) |
| all_sensor | growth (<=70 d) | k=20 d | rule=same_hour | 3 | n_blocks_neg=3; per room 1/0/2; estimable=False | 1.000 | 1.000 | 0.333 | not estimable (<10 blocks) |
| all_sensor | plateau (>70 d) | k=5 d | rule=same_hour | 34 | n_blocks_neg=34; per room 11/8/15; estimable=True | 0.542 | 0.118 | 0.882 | yes |
| all_sensor | plateau (>70 d) | k=10 d | rule=same_hour | 29 | n_blocks_neg=29; per room 9/6/14; estimable=True | 0.564 | 0.138 | 0.897 | yes |
| all_sensor | plateau (>70 d) | k=15 d | rule=same_hour | 32 | n_blocks_neg=32; per room 9/8/15; estimable=True | 0.636 | 0.188 | 0.906 | yes |
| all_sensor | plateau (>70 d) | k=20 d | rule=same_hour | 31 | n_blocks_neg=31; per room 10/5/16; estimable=True | 0.702 | 0.290 | 0.839 | yes |
| all_sensor | growth (<=70 d) | k=5 d | rule=nearest_pm1h | 8 | n_blocks_neg=8; per room 1/3/4; estimable=False | 0.984 | 0.500 | 0.750 | not estimable (<10 blocks) |
| all_sensor | growth (<=70 d) | k=10 d | rule=nearest_pm1h | 7 | n_blocks_neg=7; per room 1/2/4; estimable=False | 1.000 | 1.000 | 0.857 | not estimable (<10 blocks) |
| all_sensor | growth (<=70 d) | k=15 d | rule=nearest_pm1h | 4 | n_blocks_neg=4; per room 0/1/3; estimable=False | 1.000 | 1.000 | 1.000 | not estimable (<10 blocks) |
| all_sensor | growth (<=70 d) | k=20 d | rule=nearest_pm1h | 3 | n_blocks_neg=3; per room 1/0/2; estimable=False | 1.000 | 1.000 | 0.333 | not estimable (<10 blocks) |
| all_sensor | plateau (>70 d) | k=5 d | rule=nearest_pm1h | 34 | n_blocks_neg=34; per room 11/8/15; estimable=True | 0.540 | 0.118 | 0.882 | yes |
| all_sensor | plateau (>70 d) | k=10 d | rule=nearest_pm1h | 29 | n_blocks_neg=29; per room 9/6/14; estimable=True | 0.559 | 0.138 | 0.897 | yes |
| all_sensor | plateau (>70 d) | k=15 d | rule=nearest_pm1h | 32 | n_blocks_neg=32; per room 9/8/15; estimable=True | 0.638 | 0.188 | 0.906 | yes |
| all_sensor | plateau (>70 d) | k=20 d | rule=nearest_pm1h | 31 | n_blocks_neg=31; per room 10/5/16; estimable=True | 0.701 | 0.290 | 0.839 | yes |
