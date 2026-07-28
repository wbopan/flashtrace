# Formal visual evaluation v2 results

Protocol frozen 2026-07-24. Formal samples exclude the strict and native pilots; pilot estimates are never pooled with these results.

The earlier breadth-first n=20 previews overlap the formal fixed-seed freeze by 14 Wiki-VISA and 16 VizWiz-LF samples. For those rows only, deterministic GPU records were reused after exact image/question, frozen-response, token-ID, and model-revision identity checks; every formal estimate and bootstrap denominator is recomputed from the complete n=120/n=100 frozen sets.

## E1/E2 gate funnels

### Wiki-VISA

| gate stage | passed | eliminated at stage | not evaluated at stage |
|---|---:|---:|---:|
| candidate_manifest | 940 | 0 | 0 |
| prior_pilot_sample_exclusion | 931 | 9 | 0 |
| thinking_closed | 815 | 116 | 0 |
| generated_teacher_forced_ids_match | 797 | 18 | 0 |
| model_record_available | 797 | 0 | 0 |
| generation_stable | 797 | 0 | 0 |
| whole_output_correct | 324 | 473 | 0 |
| positive_blur_logprob_drop | 225 | 99 | 0 |
| generation_ablation_changes_output | 180 | 45 | 0 |
| final_strict_eligible | 180 | 0 | 0 |
| unique_image_and_fixed_seed_freeze | 120 | 60 | 0 |

| gate marginal | passed | failed | not evaluated |
|---|---:|---:|---:|
| thinking_closed | 815 | 116 | 0 |
| generated_teacher_forced_ids_match | 797 | 18 | 116 |
| generation_stable | 797 | 0 | 134 |
| whole_output_correct | 324 | 473 | 134 |
| positive_blur_logprob_drop | 605 | 192 | 134 |
| generation_ablation_changes_output | 180 | 751 | 0 |

### VizWiz-LF

| gate stage | passed | eliminated at stage | not evaluated at stage |
|---|---:|---:|---:|
| candidate_manifest | 200 | 0 | 0 |
| prior_pilot_sample_exclusion | 197 | 3 | 0 |
| thinking_closed | 193 | 4 | 0 |
| generated_teacher_forced_ids_match | 191 | 2 | 0 |
| model_record_available | 191 | 0 | 0 |
| thinking_within_token_limit | 191 | 0 | 0 |
| output_meets_min_tokens | 179 | 12 | 0 |
| output_non_refusal | 144 | 35 | 0 |
| generation_stable | 144 | 0 | 0 |
| positive_blur_logprob_drop | 144 | 0 | 0 |
| generation_ablation_changes_output | 144 | 0 | 0 |
| final_strict_eligible | 144 | 0 | 0 |
| unique_image_and_fixed_seed_freeze | 100 | 44 | 0 |

| gate marginal | passed | failed | not evaluated |
|---|---:|---:|---:|
| thinking_closed | 193 | 4 | 0 |
| generated_teacher_forced_ids_match | 191 | 2 | 4 |
| thinking_within_token_limit | 191 | 0 | 6 |
| output_meets_min_tokens | 179 | 12 | 6 |
| output_non_refusal | 156 | 35 | 6 |
| generation_stable | 191 | 0 | 6 |
| positive_blur_logprob_drop | 191 | 0 | 6 |
| generation_ablation_changes_output | 144 | 53 | 0 |

## E3: Wiki-VISA localization

Common paired samples: 120; paired bootstrap draws: 50000. Energy and R@5 are primary.

| method | Energy | Rank AUC | R@5 | R@20 |
|---|---:|---:|---:|---:|
| Random | 0.0385 [0.0337, 0.0437] | 0.5006 [0.4920, 0.5092] | 0.0521 [0.0458, 0.0586] | 0.2053 [0.1947, 0.2161] |
| Center prior | 0.0330 [0.0256, 0.0412] | 0.4960 [0.4597, 0.5326] | 0.0495 [0.0323, 0.0677] | 0.1725 [0.1314, 0.2158] |
| Visual LOO | 0.1192 [0.0978, 0.1425] | 0.7349 [0.7016, 0.7667] | 0.2240 [0.1920, 0.2563] | 0.6346 [0.5804, 0.6877] |
| Visual IG | 0.0459 [0.0398, 0.0528] | 0.5202 [0.5088, 0.5324] | 0.0879 [0.0734, 0.1057] | 0.2780 [0.2597, 0.2978] |
| AttnLRP | 0.2844 [0.2548, 0.3145] | 0.6358 [0.6141, 0.6576] | 0.3004 [0.2671, 0.3348] | 0.5205 [0.4880, 0.5532] |
| FlashTrace (exact, K=1) | 0.2048 [0.1844, 0.2258] | 0.7897 [0.7678, 0.8106] | 0.3114 [0.2759, 0.3480] | 0.6133 [0.5736, 0.6528] |
| IFR-span (K=0) | 0.1718 [0.1532, 0.1912] | 0.7255 [0.7010, 0.7495] | 0.2208 [0.1908, 0.2528] | 0.5047 [0.4651, 0.5451] |
| FlashTrace all-generation | 0.1952 [0.1752, 0.2159] | 0.7756 [0.7536, 0.7971] | 0.2827 [0.2492, 0.3171] | 0.5886 [0.5485, 0.6286] |

### Primary paired differences: FlashTrace minus baseline

#### Energy

| baseline | favorable delta [95% CI] | W/T/L |
|---|---:|---:|
| Random | 0.1663 [0.1472, 0.1861] | 117/0/3 |
| Center prior | 0.1718 [0.1520, 0.1921] | 119/0/1 |
| Visual LOO | 0.0856 [0.0636, 0.1072] | 97/0/23 |
| Visual IG | 0.1589 [0.1394, 0.1789] | 116/0/4 |
| AttnLRP | -0.0795 [-0.0946, -0.0647] | 22/0/98 |
| IFR-span (K=0) | 0.0330 [0.0273, 0.0389] | 105/0/15 |
| FlashTrace all-generation | 0.0096 [0.0075, 0.0119] | 103/0/17 |

#### R@5

| baseline | favorable delta [95% CI] | W/T/L |
|---|---:|---:|
| Random | 0.2593 [0.2237, 0.2961] | 118/1/1 |
| Center prior | 0.2619 [0.2220, 0.3028] | 109/0/11 |
| Visual LOO | 0.0873 [0.0450, 0.1304] | 78/0/42 |
| Visual IG | 0.2235 [0.1900, 0.2584] | 110/4/6 |
| AttnLRP | 0.0110 [-0.0079, 0.0307] | 61/4/55 |
| IFR-span (K=0) | 0.0906 [0.0749, 0.1076] | 106/10/4 |
| FlashTrace all-generation | 0.0287 [0.0228, 0.0350] | 84/27/9 |

### Wiki strata

| stratum | n | method | Energy | R@5 |
|---|---:|---|---:|---:|
| first_page_passage | 40 | Random | 0.0319 [0.0272, 0.0372] | 0.0552 [0.0440, 0.0673] |
| first_page_passage | 40 | Center prior | 0.0093 [0.0060, 0.0138] | 0.0000 [0.0000, 0.0000] |
| first_page_passage | 40 | Visual LOO | 0.0944 [0.0755, 0.1142] | 0.2419 [0.1860, 0.2964] |
| first_page_passage | 40 | Visual IG | 0.0407 [0.0339, 0.0482] | 0.0866 [0.0698, 0.1034] |
| first_page_passage | 40 | AttnLRP | 0.2554 [0.2024, 0.3104] | 0.3308 [0.2684, 0.3962] |
| first_page_passage | 40 | FlashTrace (exact, K=1) | 0.1717 [0.1431, 0.2021] | 0.2863 [0.2304, 0.3457] |
| first_page_passage | 40 | IFR-span (K=0) | 0.1546 [0.1279, 0.1833] | 0.2184 [0.1737, 0.2670] |
| first_page_passage | 40 | FlashTrace all-generation | 0.1646 [0.1364, 0.1949] | 0.2614 [0.2096, 0.3162] |
| later_page_passage | 40 | Random | 0.0363 [0.0303, 0.0427] | 0.0512 [0.0406, 0.0622] |
| later_page_passage | 40 | Center prior | 0.0515 [0.0403, 0.0630] | 0.0813 [0.0477, 0.1172] |
| later_page_passage | 40 | Visual LOO | 0.0826 [0.0641, 0.1030] | 0.1653 [0.1228, 0.2069] |
| later_page_passage | 40 | Visual IG | 0.0431 [0.0348, 0.0520] | 0.0785 [0.0637, 0.0942] |
| later_page_passage | 40 | AttnLRP | 0.2924 [0.2432, 0.3437] | 0.2816 [0.2284, 0.3367] |
| later_page_passage | 40 | FlashTrace (exact, K=1) | 0.2107 [0.1755, 0.2479] | 0.3328 [0.2696, 0.3995] |
| later_page_passage | 40 | IFR-span (K=0) | 0.1593 [0.1267, 0.1944] | 0.2124 [0.1566, 0.2746] |
| later_page_passage | 40 | FlashTrace all-generation | 0.1956 [0.1606, 0.2335] | 0.2926 [0.2329, 0.3573] |
| non_passage | 40 | Random | 0.0473 [0.0355, 0.0597] | 0.0499 [0.0391, 0.0609] |
| non_passage | 40 | Center prior | 0.0382 [0.0225, 0.0575] | 0.0672 [0.0336, 0.1042] |
| non_passage | 40 | Visual LOO | 0.1805 [0.1260, 0.2388] | 0.2649 [0.2026, 0.3285] |
| non_passage | 40 | Visual IG | 0.0539 [0.0396, 0.0714] | 0.0984 [0.0622, 0.1463] |
| non_passage | 40 | AttnLRP | 0.3053 [0.2554, 0.3556] | 0.2889 [0.2353, 0.3494] |
| non_passage | 40 | FlashTrace (exact, K=1) | 0.2321 [0.1950, 0.2702] | 0.3150 [0.2527, 0.3815] |
| non_passage | 40 | IFR-span (K=0) | 0.2014 [0.1675, 0.2370] | 0.2316 [0.1814, 0.2892] |
| non_passage | 40 | FlashTrace all-generation | 0.2253 [0.1891, 0.2632] | 0.2940 [0.2357, 0.3563] |

### Supplemental localization endpoints

These endpoints are computed from the same whole-patch, tie-aware maps and paired n=120 intersection; they are not primary endpoints.

| method | Pointing Game | Top-area IoU | R@1 | R@10 |
|---|---:|---:|---:|---:|
| Random | 0.0583 [0.0167, 0.1000] | 0.0200 [0.0165, 0.0236] | 0.0122 [0.0093, 0.0154] | 0.1034 [0.0950, 0.1120] |
| Center prior | 0.0625 [0.0292, 0.1042] | 0.0171 [0.0097, 0.0257] | 0.0102 [0.0049, 0.0162] | 0.0882 [0.0612, 0.1168] |
| Visual LOO | 0.7000 [0.6167, 0.7833] | 0.1044 [0.0836, 0.1272] | 0.2240 [0.1918, 0.2561] | 0.4246 [0.3735, 0.4767] |
| Visual IG | 0.0333 [0.0083, 0.0667] | 0.0272 [0.0225, 0.0323] | 0.0187 [0.0124, 0.0267] | 0.1661 [0.1483, 0.1861] |
| AttnLRP | 0.5833 [0.4917, 0.6667] | 0.1324 [0.1179, 0.1477] | 0.1319 [0.1100, 0.1565] | 0.4039 [0.3696, 0.4388] |
| FlashTrace (exact, K=1) | 0.6333 [0.5417, 0.7167] | 0.1282 [0.1153, 0.1416] | 0.1244 [0.1050, 0.1452] | 0.4439 [0.4041, 0.4846] |
| IFR-span (K=0) | 0.5917 [0.5000, 0.6750] | 0.0913 [0.0807, 0.1025] | 0.0952 [0.0785, 0.1139] | 0.3401 [0.3034, 0.3779] |
| FlashTrace all-generation | 0.6250 [0.5333, 0.7083] | 0.1150 [0.1034, 0.1271] | 0.1136 [0.0948, 0.1340] | 0.4079 [0.3701, 0.4468] |

## E4: VizWiz-LF frozen-response faithfulness

Common paired samples: 100; paired bootstrap draws: 50000. Deletion AUC is the primary endpoint.

| method | Deletion AUC ↓ | Insertion AUC ↑ | Visual-MAS ↓ |
|---|---:|---:|---:|
| Random | 0.4010 [0.3602, 0.4419] | 0.4547 [0.4158, 0.4940] | 0.5403 [0.4976, 0.5846] |
| Center prior | 0.4015 [0.3596, 0.4435] | 0.6298 [0.5924, 0.6661] | 0.5469 [0.4910, 0.6028] |
| Visual LOO | 0.3358 [0.2938, 0.3791] | 0.6712 [0.6363, 0.7040] | 0.4757 [0.4230, 0.5288] |
| Visual IG | 0.4116 [0.3723, 0.4514] | 0.4426 [0.4006, 0.4847] | 0.5549 [0.5024, 0.6071] |
| AttnLRP | 0.3662 [0.3234, 0.4096] | 0.5306 [0.4891, 0.5714] | 0.5125 [0.4530, 0.5716] |
| FlashTrace (exact, K=1) | 0.3599 [0.3207, 0.3993] | 0.5317 [0.4928, 0.5707] | 0.4938 [0.4433, 0.5448] |
| IFR-span (K=0) | 0.3599 [0.3208, 0.3990] | 0.5327 [0.4926, 0.5726] | 0.4935 [0.4431, 0.5444] |
| FlashTrace all-generation | 0.3614 [0.3223, 0.4006] | 0.5314 [0.4915, 0.5701] | 0.4964 [0.4466, 0.5475] |

Visual LOO is retained in the complete eight-method appendix as a cost-insensitive perturbation diagnostic. The practical main comparison and interpretation use Visual IG, AttnLRP, and FlashTrace; Center remains an explicit spatial-prior check.

### FlashTrace favorable deletion-AUC differences

| baseline | favorable delta [95% CI] | W/T/L |
|---|---:|---:|
| Random | 0.0411 [0.0186, 0.0647] | 64/3/33 |
| Center prior | 0.0416 [0.0110, 0.0718] | 60/2/38 |
| Visual LOO | -0.0241 [-0.0479, 0.0004] | 38/4/58 |
| Visual IG | 0.0517 [0.0271, 0.0772] | 61/2/37 |
| AttnLRP | 0.0063 [-0.0123, 0.0262] | 42/6/52 |
| IFR-span (K=0) | 0.0000 [-0.0056, 0.0055] | 44/15/41 |
| FlashTrace all-generation | 0.0015 [-0.0030, 0.0062] | 39/22/39 |

### Signed-order vs positive-only sensitivity

| method | signed deletion AUC | positive-only deletion AUC | shift |
|---|---:|---:|---:|
| Random | 0.4010 | 0.4010 | +0.0000 |
| Center prior | 0.4015 | 0.4015 | +0.0000 |
| Visual LOO | 0.3358 | 0.3284 | -0.0074 |
| Visual IG | 0.4116 | 0.4172 | +0.0056 |
| AttnLRP | 0.3662 | 0.3617 | -0.0045 |
| FlashTrace (exact, K=1) | 0.3599 | 0.3599 | +0.0000 |
| IFR-span (K=0) | 0.3599 | 0.3599 | +0.0000 |
| FlashTrace all-generation | 0.3614 | 0.3614 | +0.0000 |

## E5: Wiki-VISA frozen-response faithfulness

Common paired samples: 120; paired bootstrap draws: 50000. Deletion AUC is the primary endpoint.

| method | Deletion AUC ↓ | Insertion AUC ↑ | Visual-MAS ↓ |
|---|---:|---:|---:|
| Random | 0.4414 [0.3975, 0.4857] | 0.5423 [0.4936, 0.5900] | 0.5681 [0.5275, 0.6095] |
| Center prior | 0.5151 [0.4684, 0.5614] | 0.4399 [0.3910, 0.4897] | 0.6036 [0.5560, 0.6510] |
| Visual LOO | 0.2266 [0.1868, 0.2684] | 0.7954 [0.7653, 0.8238] | 0.3237 [0.2734, 0.3760] |
| Visual IG | 0.4402 [0.3938, 0.4863] | 0.5453 [0.5010, 0.5886] | 0.5313 [0.4780, 0.5842] |
| AttnLRP | 0.3927 [0.3382, 0.4474] | 0.6439 [0.5908, 0.6951] | 0.4964 [0.4306, 0.5621] |
| FlashTrace (exact, K=1) | 0.3303 [0.2848, 0.3768] | 0.6950 [0.6539, 0.7347] | 0.4331 [0.3814, 0.4853] |
| IFR-span (K=0) | 0.3464 [0.2999, 0.3934] | 0.6793 [0.6370, 0.7197] | 0.4516 [0.3976, 0.5060] |
| FlashTrace all-generation | 0.3353 [0.2889, 0.3828] | 0.6932 [0.6507, 0.7337] | 0.4395 [0.3854, 0.4936] |

### FlashTrace favorable deletion-AUC differences

| baseline | favorable delta [95% CI] | W/T/L |
|---|---:|---:|
| Random | 0.1111 [0.0704, 0.1524] | 77/9/34 |
| Center prior | 0.1848 [0.1358, 0.2335] | 82/8/30 |
| Visual LOO | -0.1037 [-0.1362, -0.0725] | 25/26/69 |
| Visual IG | 0.1099 [0.0747, 0.1455] | 82/11/27 |
| AttnLRP | 0.0624 [0.0302, 0.0962] | 56/21/43 |
| IFR-span (K=0) | 0.0161 [0.0039, 0.0283] | 55/34/31 |
| FlashTrace all-generation | 0.0050 [-0.0036, 0.0136] | 40/44/36 |

### Signed-order vs positive-only sensitivity

| method | signed deletion AUC | positive-only deletion AUC | shift |
|---|---:|---:|---:|
| Random | 0.4414 | 0.4414 | +0.0000 |
| Center prior | 0.5151 | 0.5151 | +0.0000 |
| Visual LOO | 0.2266 | 0.2342 | +0.0076 |
| Visual IG | 0.4402 | 0.4771 | +0.0369 |
| AttnLRP | 0.3927 | 0.3933 | +0.0006 |
| FlashTrace (exact, K=1) | 0.3303 | 0.3303 | +0.0000 |
| IFR-span (K=0) | 0.3464 | 0.3464 | +0.0000 |
| FlashTrace all-generation | 0.3353 | 0.3353 | +0.0000 |

## A1–A4: recursion and geometry diagnostics

| dataset | n | exact/all-generation cosine | direct positive mass | recursive positive mass | recursive absolute mass |
|---|---:|---:|---:|---:|---:|
| Wiki-VISA | 120 | 0.9921 [0.9907, 0.9932] | 0.5500 [0.5411, 0.5589] | 0.4500 [0.4411, 0.4589] | 0.4500 [0.4411, 0.4589] |
| VizWiz-LF | 100 | 0.9958 [0.9951, 0.9964] | 0.6757 [0.6642, 0.6875] | 0.3243 [0.3124, 0.3358] | 0.3243 [0.3125, 0.3358] |

### Native-evidence centroid distance to image center

| Wiki-VISA stratum | GT centroid distance [95% CI] |
|---|---:|
| first_page_passage | 0.4213 [0.4052, 0.4359] |
| later_page_passage | 0.2022 [0.1689, 0.2365] |
| non_passage | 0.3082 [0.2552, 0.3591] |

### Heatmap geometry and sign diagnostics

| dataset | method | border mass | top-row mass | heatmap centroid distance | negative cells |
|---|---|---:|---:|---:|---:|
| Wiki-VISA | Random | 0.1209 [0.1199, 0.1220] | 0.0311 [0.0306, 0.0316] | 0.0064 [0.0059, 0.0070] | 0.0000 [0.0000, 0.0000] |
| Wiki-VISA | Center prior | 0.0050 [0.0050, 0.0050] | 0.0012 [0.0012, 0.0012] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| Wiki-VISA | Visual LOO | 0.8270 [0.7942, 0.8579] | 0.4420 [0.3783, 0.5061] | 0.3575 [0.3275, 0.3875] | 0.3542 [0.3146, 0.3943] |
| Wiki-VISA | Visual IG | 0.1147 [0.1080, 0.1218] | 0.0212 [0.0187, 0.0241] | 0.0717 [0.0649, 0.0787] | 0.4981 [0.4935, 0.5025] |
| Wiki-VISA | AttnLRP | 0.1331 [0.1242, 0.1422] | 0.0629 [0.0563, 0.0699] | 0.2515 [0.2338, 0.2692] | 0.4419 [0.4331, 0.4507] |
| Wiki-VISA | FlashTrace (exact, K=1) | 0.1108 [0.1036, 0.1186] | 0.0350 [0.0318, 0.0386] | 0.1597 [0.1474, 0.1722] | 0.0000 [0.0000, 0.0000] |
| Wiki-VISA | IFR-span (K=0) | 0.1271 [0.1179, 0.1370] | 0.0450 [0.0405, 0.0500] | 0.1626 [0.1500, 0.1754] | 0.0000 [0.0000, 0.0000] |
| Wiki-VISA | FlashTrace all-generation | 0.1187 [0.1108, 0.1274] | 0.0398 [0.0360, 0.0441] | 0.1614 [0.1491, 0.1739] | 0.0000 [0.0000, 0.0000] |
| VizWiz-LF | Random | 0.1210 [0.1198, 0.1221] | 0.0315 [0.0308, 0.0321] | 0.0071 [0.0064, 0.0079] | 0.0000 [0.0000, 0.0000] |
| VizWiz-LF | Center prior | 0.0050 [0.0050, 0.0050] | 0.0012 [0.0012, 0.0012] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| VizWiz-LF | Visual LOO | 0.6245 [0.5761, 0.6721] | 0.1857 [0.1511, 0.2233] | 0.2032 [0.1762, 0.2324] | 0.2512 [0.2037, 0.3019] |
| VizWiz-LF | Visual IG | 0.0935 [0.0833, 0.1043] | 0.0228 [0.0198, 0.0262] | 0.0937 [0.0824, 0.1057] | 0.4947 [0.4900, 0.4993] |
| VizWiz-LF | AttnLRP | 0.1677 [0.1521, 0.1850] | 0.1118 [0.1004, 0.1245] | 0.1511 [0.1375, 0.1652] | 0.2423 [0.2310, 0.2538] |
| VizWiz-LF | FlashTrace (exact, K=1) | 0.1449 [0.1331, 0.1575] | 0.0813 [0.0745, 0.0884] | 0.1077 [0.0978, 0.1179] | 0.0000 [0.0000, 0.0000] |
| VizWiz-LF | IFR-span (K=0) | 0.1493 [0.1368, 0.1625] | 0.0867 [0.0790, 0.0945] | 0.1095 [0.0996, 0.1195] | 0.0000 [0.0000, 0.0000] |
| VizWiz-LF | FlashTrace all-generation | 0.1483 [0.1362, 0.1613] | 0.0868 [0.0793, 0.0944] | 0.1092 [0.0991, 0.1195] | 0.0000 [0.0000, 0.0000] |

VizWiz-LF has no native evidence mask, so a ground-truth evidence centroid is not defined; its A3 report is restricted to heatmap centroids and border/top-row mass. Wiki-VISA additionally reports native-box centroid distance by stratum.

### One-hop recursion gain by THINKING length

Localization deltas are K=1 minus K=0; deletion is oriented so positive values favor K=1.

| dataset | bucket | Δ Energy | Δ R@5 | favorable Δ deletion AUC |
|---|---|---:|---:|---:|
| Wiki-VISA | short | 0.0336 [0.0261, 0.0411] | 0.0676 [0.0482, 0.0883] | 0.0053 [-0.0197, 0.0294] |
| Wiki-VISA | medium | 0.0318 [0.0215, 0.0430] | 0.0919 [0.0652, 0.1209] | 0.0185 [0.0036, 0.0346] |
| Wiki-VISA | long | 0.0336 [0.0225, 0.0453] | 0.1122 [0.0807, 0.1476] | 0.0245 [0.0034, 0.0471] |
| VizWiz-LF | short | -- | -- | 0.0045 [-0.0044, 0.0139] |
| VizWiz-LF | medium | -- | -- | -0.0015 [-0.0096, 0.0059] |
| VizWiz-LF | long | -- | -- | -0.0030 [-0.0146, 0.0074] |

## Spatial resolution disclosure

Native attribution grids are method outputs before nearest-patch resampling. Faithfulness layouts are shared by every method for a given image and contain approximately 64 perturbation regions.

| dataset | method | native attribution grid shapes | faithfulness layouts |
|---|---|---|---|
| Wiki-VISA | Random | 32x32 (n=120) | 16x4 (n=120) |
| Wiki-VISA | Center prior | 32x32 (n=120) | 16x4 (n=120) |
| Wiki-VISA | Visual LOO | 4x4 (n=120) | 16x4 (n=120) |
| Wiki-VISA | Visual IG | 88x22 (n=120) | 16x4 (n=120) |
| Wiki-VISA | AttnLRP | 88x22 (n=120) | 16x4 (n=120) |
| Wiki-VISA | FlashTrace (exact, K=1) | 88x22 (n=120) | 16x4 (n=120) |
| Wiki-VISA | IFR-span (K=0) | 88x22 (n=120) | 16x4 (n=120) |
| Wiki-VISA | FlashTrace all-generation | 88x22 (n=120) | 16x4 (n=120) |
| VizWiz-LF | Random | 32x32 (n=100) | 7x9 (n=7), 8x8 (n=1), 9x7 (n=92) |
| VizWiz-LF | Center prior | 32x32 (n=100) | 7x9 (n=7), 8x8 (n=1), 9x7 (n=92) |
| VizWiz-LF | Visual LOO | 4x4 (n=100) | 7x9 (n=7), 8x8 (n=1), 9x7 (n=92) |
| VizWiz-LF | Visual IG | 14x14 (n=1), 17x13 (n=10), 20x15 (n=1), 22x30 (n=4), 26x19 (n=2), 32x24 (n=3), 38x51 (n=3), 40x30 (n=33), 51x38 (n=43) | 7x9 (n=7), 8x8 (n=1), 9x7 (n=92) |
| VizWiz-LF | AttnLRP | 14x14 (n=1), 17x13 (n=10), 20x15 (n=1), 22x30 (n=4), 26x19 (n=2), 32x24 (n=3), 38x51 (n=3), 40x30 (n=33), 51x38 (n=43) | 7x9 (n=7), 8x8 (n=1), 9x7 (n=92) |
| VizWiz-LF | FlashTrace (exact, K=1) | 14x14 (n=1), 17x13 (n=10), 20x15 (n=1), 22x30 (n=4), 26x19 (n=2), 32x24 (n=3), 38x51 (n=3), 40x30 (n=33), 51x38 (n=43) | 7x9 (n=7), 8x8 (n=1), 9x7 (n=92) |
| VizWiz-LF | IFR-span (K=0) | 14x14 (n=1), 17x13 (n=10), 20x15 (n=1), 22x30 (n=4), 26x19 (n=2), 32x24 (n=3), 38x51 (n=3), 40x30 (n=33), 51x38 (n=43) | 7x9 (n=7), 8x8 (n=1), 9x7 (n=92) |
| VizWiz-LF | FlashTrace all-generation | 14x14 (n=1), 17x13 (n=10), 20x15 (n=1), 22x30 (n=4), 26x19 (n=2), 32x24 (n=3), 38x51 (n=3), 40x30 (n=33), 51x38 (n=43) | 7x9 (n=7), 8x8 (n=1), 9x7 (n=92) |

IFR-span, Visual IG, AttnLRP, FlashTrace, and FlashTrace all-generation share the same native model-token grid within each sample. Random/Center use a 32x32 synthetic grid and Visual LOO uses a coarse 4x4 perturbation grid. Nearest-neighbor resampling does not create sub-patch attribution detail.

## Observed visual compute

Times are per successful sample-method on the formal common intersection. Attribution VRAM is incremental peak allocation; faithfulness time covers the matched 64-region/10-step perturbations.

| dataset | method | attribution seconds | incremental peak VRAM GiB | faithfulness seconds |
|---|---|---:|---:|---:|
| Wiki-VISA | Random | 0.005 | 0.000 | 23.037 |
| Wiki-VISA | Center prior | 0.003 | 0.000 | 23.034 |
| Wiki-VISA | Visual LOO | 21.213 | 9.204 | 42.440 |
| Wiki-VISA | Visual IG | 28.058 | 15.930 | 45.949 |
| Wiki-VISA | AttnLRP | 1.914 | 13.258 | 45.916 |
| Wiki-VISA | FlashTrace (exact, K=1) | 2.055 | 13.280 | 23.024 |
| Wiki-VISA | IFR-span (K=0) | 1.587 | 13.280 | 23.039 |
| Wiki-VISA | FlashTrace all-generation | 2.044 | 13.280 | 22.995 |
| VizWiz-LF | Random | 0.000 | 0.000 | 15.435 |
| VizWiz-LF | Center prior | 0.000 | 0.000 | 15.438 |
| VizWiz-LF | Visual LOO | 13.868 | 5.630 | 24.687 |
| VizWiz-LF | Visual IG | 19.119 | 9.875 | 30.857 |
| VizWiz-LF | AttnLRP | 1.435 | 8.803 | 30.870 |
| VizWiz-LF | FlashTrace (exact, K=1) | 1.547 | 10.797 | 15.451 |
| VizWiz-LF | IFR-span (K=0) | 1.147 | 10.797 | 15.456 |
| VizWiz-LF | FlashTrace all-generation | 1.555 | 10.797 | 15.478 |

## A8: VizWiz semantic correctness sensitivity

Labels: {'fully': 58, 'partial': 22, 'wrong': 20}. Independent human audit: 10/10. Fully-correct subset size: 58.

| method | Deletion AUC ↓ | Insertion AUC ↑ | Visual-MAS ↓ |
|---|---:|---:|---:|
| Random | 0.3806 [0.3280, 0.4324] | 0.4289 [0.3781, 0.4796] | 0.5268 [0.4732, 0.5825] |
| Center prior | 0.3860 [0.3325, 0.4405] | 0.6243 [0.5769, 0.6687] | 0.5285 [0.4555, 0.6018] |
| Visual LOO | 0.3282 [0.2739, 0.3839] | 0.6548 [0.6066, 0.6998] | 0.4593 [0.3922, 0.5285] |
| Visual IG | 0.3797 [0.3266, 0.4333] | 0.4333 [0.3764, 0.4899] | 0.5093 [0.4378, 0.5801] |
| AttnLRP | 0.3306 [0.2758, 0.3867] | 0.5167 [0.4645, 0.5668] | 0.4620 [0.3856, 0.5401] |
| FlashTrace (exact, K=1) | 0.3322 [0.2826, 0.3828] | 0.5178 [0.4648, 0.5686] | 0.4599 [0.3970, 0.5258] |
| IFR-span (K=0) | 0.3340 [0.2849, 0.3847] | 0.5133 [0.4601, 0.5646] | 0.4613 [0.3989, 0.5269] |
| FlashTrace all-generation | 0.3377 [0.2883, 0.3878] | 0.5146 [0.4615, 0.5667] | 0.4679 [0.4040, 0.5345] |

### Fully-correct FlashTrace favorable deletion-AUC differences

| baseline | favorable delta [95% CI] | W/T/L |
|---|---:|---:|
| Random | 0.0484 [0.0226, 0.0756] | 40/2/16 |
| Center prior | 0.0538 [0.0164, 0.0910] | 35/2/21 |
| Visual LOO | -0.0040 [-0.0349, 0.0279] | 25/3/30 |
| Visual IG | 0.0475 [0.0163, 0.0786] | 36/2/20 |
| AttnLRP | -0.0016 [-0.0230, 0.0197] | 23/5/30 |
| IFR-span (K=0) | 0.0018 [-0.0047, 0.0087] | 24/10/24 |
| FlashTrace all-generation | 0.0055 [-0.0000, 0.0120] | 23/16/19 |

## Independent frozen-sample protocol audits

Wiki-VISA (12/120): image dependence {'borderline': 1, 'supported': 11}; THINKING quality {'good': 11, 'mixed': 1}.

VizWiz-LF (10/100): image dependence {'borderline': 1, 'supported': 8, 'unsupported': 1}; THINKING quality {'good': 6, 'mixed': 3, 'poor': 1}.

These reviews are caveat-only and did not change frozen IDs.

## Scope and limitations

- Wiki-VISA boxes mark supporting HTML elements, not exhaustive word-level evidence.
- VizWiz-LF evaluates faithfulness to prompted long-form model responses; answer correctness is a sensitivity label, not a gate.
- Strict stability and image-dependence gates improve internal validity while reducing representativeness; the complete funnels above expose that selection.
- Center prior remains in every method panel. Claims are limited to one frozen VLM and one recursive hop.
- CLEVR-XAI n=20 and VISTAQA n=10 remain separate diagnostics; see `A6_LEGACY_DIAGNOSTICS.md`.
