**Table 11. Modality-availability regimes, named by the blocks present; models A (primary complete-case model applied unchanged), B (trained on all hours), C (regime-specific). Unscored regimes have fewer than 30 anchors per held-out room**

| regime (blocks present) | hours | rooms and age coverage | anchors room2 | anchors room6 | anchors room7 | scored (>= 30 anchors in a held-out room) | MAE macro A | MAE macro B | MAE macro C |
|---|---|---|---|---|---|---|---|---|---|
| audio  (absent: tracking, morphometry, optical flow) | 3,971 | rooms: room2, room6, room7; age coverage 1-181 d | 1,671 | 1,106 | 1,194 | yes | 39.03 | 31.96 | 43.89 |
| audio + tracking + morphometry + optical flow  (all four blocks) | 3,090 | rooms: room2, room6, room7; age coverage 31-182 d | 1,006 | 824 | 1,260 | yes | 19.16 | 16.03 | 13.70 |
| audio + optical flow  (absent: tracking, morphometry) | 1,156 | rooms: room2, room6, room7; age coverage 29-181 d | 385 | 296 | 475 | yes | 41.44 | 27.19 | 21.16 |
| audio + tracking + morphometry  (absent: optical flow) | 1,108 | rooms: room2, room6, room7; age coverage 32-181 d | 366 | 309 | 433 | yes | 14.49 | 14.02 | 12.00 |
| tracking + morphometry + optical flow  (absent: audio) | 469 | rooms: room2, room6; age coverage 31-182 d | 104 | 365 | 0 | yes | 26.37 | 21.63 | 19.42 |
| optical flow  (absent: audio, tracking, morphometry) | 218 | rooms: room2, room6; age coverage 29-181 d | 53 | 165 | 0 | yes | 48.26 | 28.69 | 21.69 |
| tracking + morphometry  (absent: audio, optical flow) | 195 | rooms: room2, room6, room7; age coverage 32-182 d | 43 | 131 | 21 | yes | 27.01 | 19.02 | 22.76 |
| audio + tracking + optical flow  (absent: morphometry) | 89 | rooms: room2, room6, room7; age coverage 29-78 d | 40 | 16 | 33 | yes | 65.27 | 9.51 | 10.99 |
| audio + tracking  (absent: morphometry, optical flow) | 52 | rooms: room2, room6, room7; age coverage 29-112 d | 23 | 11 | 18 | no: unscored |  |  |  |
| tracking + optical flow  (absent: audio, morphometry) | 42 | rooms: room2, room6; age coverage 29-31 d | 20 | 22 | 0 | no: unscored |  |  |  |
| tracking  (absent: audio, morphometry, optical flow) | 22 | rooms: room2, room6, room7; age coverage 29-112 d | 7 | 11 | 4 | no: unscored |  |  |  |
