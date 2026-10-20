**Table 1b. Key dates and reference periods**

| item | room | date_or_range | age_days | note |
|---|---|---|---|---|
| placement (age 0) | all | 2025-06-05 | 0.00 | day_zero; age = days since placement |
| sensor start: audio | room2 | 2025-06-06 06:00:00 | 1.25 | first observed hour |
| sensor start: tracking (YOLO) | room2 | 2025-07-03 16:00:00 | 28.67 | first observed hour |
| sensor start: optical flow | room2 | 2025-07-03 13:00:00 | 28.54 | first observed hour |
| sensor start: morphometry (before QC filter) | room2 | 2025-07-03 16:00:00 | 28.67 | first observed hour |
| sensor start: morphometry (after QC filter) | room2 | 2025-07-06 09:00:00 | 31.38 | first observed hour |
| morphometry first-week reference period (body-size reference) | room2 | 2025-07-03 16:00:00 to 2025-07-10 16:00:00 | 28.67 | median over hours with age <= first morphometry hour + 7 d; includes hours later removed by the QC burn-in |
| audio first-week reference period (V2 only) | room2 | 2025-06-06 06:00:00 to 2025-06-13 06:00:00 | 1.25 | first 7 days of the room's audio; not used by the primary pipeline |
| day-28 baseline: age of the first morphometry hour (start of the body-size reference) | room2 | 2025-07-03 16:00:00 | 28.67 | the reference week covers ages ~28.7-35.7 d |
| analysed age range (complete-case hours, primary) | room2 | 31.38 to 181.50 d |  | 1372 hours |
| last observed hour (any modality) | room2 | 2025-12-03 21:00:00 | 181.88 |  |
| sensor start: audio | room6 | 2025-06-05 13:00:00 | 0.54 | first observed hour |
| sensor start: tracking (YOLO) | room6 | 2025-07-03 17:00:00 | 28.71 | first observed hour |
| sensor start: optical flow | room6 | 2025-07-03 14:00:00 | 28.58 | first observed hour |
| sensor start: morphometry (before QC filter) | room6 | 2025-07-03 17:00:00 | 28.71 | first observed hour |
| sensor start: morphometry (after QC filter) | room6 | 2025-07-06 09:00:00 | 31.38 | first observed hour |
| morphometry first-week reference period (body-size reference) | room6 | 2025-07-03 17:00:00 to 2025-07-10 17:00:00 | 28.71 | median over hours with age <= first morphometry hour + 7 d; includes hours later removed by the QC burn-in |
| audio first-week reference period (V2 only) | room6 | 2025-06-05 13:00:00 to 2025-06-12 13:00:00 | 0.54 | first 7 days of the room's audio; not used by the primary pipeline |
| day-28 baseline: age of the first morphometry hour (start of the body-size reference) | room6 | 2025-07-03 17:00:00 | 28.71 | the reference week covers ages ~28.7-35.7 d |
| analysed age range (complete-case hours, primary) | room6 | 44.42 to 178.58 d |  | 1133 hours |
| last observed hour (any modality) | room6 | 2025-12-03 21:00:00 | 181.88 |  |
| sensor start: audio | room7 | 2025-07-03 14:00:00 | 28.58 | first observed hour |
| sensor start: tracking (YOLO) | room7 | 2025-07-03 17:00:00 | 28.71 | first observed hour |
| sensor start: optical flow | room7 | 2025-07-03 14:00:00 | 28.58 | first observed hour |
| sensor start: morphometry (before QC filter) | room7 | 2025-07-03 17:00:00 | 28.71 | first observed hour |
| sensor start: morphometry (after QC filter) | room7 | 2025-07-06 08:00:00 | 31.33 | first observed hour |
| morphometry first-week reference period (body-size reference) | room7 | 2025-07-03 17:00:00 to 2025-07-10 17:00:00 | 28.71 | median over hours with age <= first morphometry hour + 7 d; includes hours later removed by the QC burn-in |
| audio first-week reference period (V2 only) | room7 | 2025-07-03 14:00:00 to 2025-07-10 14:00:00 | 28.58 | first 7 days of the room's audio; not used by the primary pipeline |
| day-28 baseline: age of the first morphometry hour (start of the body-size reference) | room7 | 2025-07-03 17:00:00 | 28.71 | the reference week covers ages ~28.7-35.7 d |
| analysed age range (complete-case hours, primary) | room7 | 31.33 to 181.71 d |  | 1693 hours |
| last observed hour (any modality) | room7 | 2025-12-03 21:00:00 | 181.88 |  |
| analysed age range (complete-case hours, all rooms) | all | 31.33 to 181.71 d |  | 4198 hours |
| season cut | all | 2025-09-01 00:00 (age 88 d): summer < 88 d, fall >= 88 d | 88.00 | used by the seasonal folds |
| plateau boundary | all | 70 d (growth <= 70 d, plateau > 70 d) | 70.00 | used by growth/plateau reporting and detection |
