# Spraak1
ASR Project 1

Implementing different GMM's for a given phoneset (40 classes) in the TIMIT database


Model - GMM Model
Syntax - Delta_NumMixtures_(with/with out)EC

Delta:
0 — MFCC
1 — MFCC_Delta
2 — MFCC_Delta_Delta

NumMixtures:
Number of mixtures used in different GMM’s
[2, 4, 8, 16, 32, 64, 128, 256]

EC:
EnergyCoefficient corresponding to the feature vectors


Different Cases Considered:

CASE 1 :
For delta - [0, 1, 2] and 64 mixtures with energy coefficient

CASE 2 :
For delta - [0, 1, 2] and 64 mixtures with out energy coefficient

CASE 3 :
For delta - [0] and mixtures -[2, 4, 8, 16, 32, 64, 128, 256] with out energy coefficient


Accuracy & Error Rates :
FLA - Frame Level Accuracy
PER - Phoneme Error Rate


       Model                   FLA      PER

Case 1/0_064_EC.pkl    =====  14.35    87.63
Case 1/1_064_EC.pkl    =====  18.73    85.47
Case 1/2_064_EC.pkl    =====  19.33    84.53
Case 2/0_064_noEC.pkl  =====  13.26    85.59
Case 2/1_064_noEC.pkl  =====  17.16    85.47
Case 2/2_064_noEC.pkl  =====  17.05    85.43
Case 3/0_002_noEC.pkl  =====  11.14    86.25
Case 3/0_004_noEC.pkl  =====  13.81    86.44
Case 3/0_008_noEC.pkl  =====  13.60    86.68
Case 3/0_016_noEC.pkl  =====  13.81    86.83
Case 3/0_032_noEC.pkl  =====  13.76    86.97
Case 3/0_064_noEC.pkl  =====  13.25    87.11
Case 3/0_128_noEC.pkl  =====  12.62    87.29
Case 3/0_256_noEC.pkl  =====  12.12    87.48

