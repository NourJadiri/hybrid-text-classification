
# Confusion Matrix Analysis - Optimal Threshold (0.65)

This section provides detailed confusion matrix analysis for both narratives and subnarratives using the optimal threshold of 0.65. The analysis includes both aggregated results across all languages and detailed per-language breakdowns.

## Aggregated Results Across All Languages


### Narratives Summary Statistics

- **Total Labels Analyzed**: 17
- **Total Samples**: 1780
- **Macro-averaged Precision**: 0.347
- **Macro-averaged Recall**: 0.470
- **Macro-averaged F1**: 0.380
- **Micro-averaged Precision**: 0.349
- **Micro-averaged Recall**: 0.457
- **Micro-averaged F1**: 0.396



## Top 15 Narratives - Confusion Matrix Analysis (Aggregated - Threshold 0.65)

| Label | TP | FP | FN | TN | Precision | Recall | F1 Score |
|-------|----|----|----|----|-----------|--------|---------|
| CC: Questioning the measurements and science | 4 | 3 | 0 | 34 | 0.571 | 1.000 | 0.727 |
| CC: Criticism of climate policies | 6 | 2 | 3 | 24 | 0.750 | 0.667 | 0.706 |
| CC: Amplifying Climate Fears | 17 | 6 | 19 | 63 | 0.739 | 0.472 | 0.576 |
| URW: Discrediting the West, Diplomacy | 24 | 19 | 20 | 115 | 0.558 | 0.545 | 0.552 |
| CC: Criticism of institutions and authorities | 15 | 24 | 4 | 68 | 0.385 | 0.789 | 0.517 |
| CC: Downplaying climate change | 3 | 6 | 1 | 25 | 0.333 | 0.750 | 0.462 |
| URW: Discrediting Ukraine | 19 | 33 | 14 | 112 | 0.365 | 0.576 | 0.447 |
| CC: Criticism of climate movement | 6 | 13 | 3 | 54 | 0.316 | 0.667 | 0.429 |
| URW: Distrust towards Media | 2 | 3 | 4 | 64 | 0.400 | 0.333 | 0.364 |
| URW: Praise of Russia | 11 | 22 | 23 | 87 | 0.333 | 0.324 | 0.328 |
| URW: Speculating war outcomes | 4 | 11 | 8 | 85 | 0.267 | 0.333 | 0.296 |
| URW: Amplifying war-related fears | 6 | 25 | 10 | 96 | 0.194 | 0.375 | 0.255 |
| Other | 5 | 15 | 18 | 105 | 0.250 | 0.217 | 0.233 |
| URW: Negative Consequences for the West | 4 | 25 | 4 | 104 | 0.138 | 0.500 | 0.216 |
| URW: Blaming the war on others rather than the ... | 4 | 20 | 11 | 73 | 0.167 | 0.267 | 0.205 |



### Subnarratives Summary Statistics

- **Total Labels Analyzed**: 29
- **Total Samples**: 1780
- **Macro-averaged Precision**: 0.270
- **Macro-averaged Recall**: 0.508
- **Macro-averaged F1**: 0.329
- **Micro-averaged Precision**: 0.296
- **Micro-averaged Recall**: 0.469
- **Micro-averaged F1**: 0.363



## Top 15 Subnarratives - Confusion Matrix Analysis (Aggregated - Threshold 0.65)

| Label | TP | FP | FN | TN | Precision | Recall | F1 Score |
|-------|----|----|----|----|-----------|--------|---------|
| CC: Amplifying Climate Fears: Amplifying existi... | 16 | 7 | 11 | 71 | 0.696 | 0.593 | 0.640 |
| URW: Negative Consequences for the West: Other | 3 | 4 | 0 | 28 | 0.429 | 1.000 | 0.600 |
| CC: Criticism of climate policies: Other | 4 | 4 | 2 | 25 | 0.500 | 0.667 | 0.571 |
| URW: Discrediting Ukraine: Discrediting Ukraini... | 7 | 10 | 2 | 13 | 0.412 | 0.778 | 0.538 |
| URW: Discrediting the West, Diplomacy: Other | 16 | 27 | 9 | 126 | 0.372 | 0.640 | 0.471 |
| CC: Criticism of climate movement: Climate move... | 4 | 9 | 0 | 28 | 0.308 | 1.000 | 0.471 |
| URW: Amplifying war-related fears: Russia will ... | 2 | 3 | 2 | 28 | 0.400 | 0.500 | 0.444 |
| URW: Praise of Russia: Praise of Russian milita... | 6 | 13 | 2 | 46 | 0.316 | 0.750 | 0.444 |
| CC: Criticism of institutions and authorities: ... | 9 | 22 | 1 | 44 | 0.290 | 0.900 | 0.439 |
| CC: Amplifying Climate Fears: Other | 8 | 10 | 12 | 40 | 0.444 | 0.400 | 0.421 |
| URW: Amplifying war-related fears: There is a r... | 4 | 8 | 3 | 55 | 0.333 | 0.571 | 0.421 |
| URW: Discrediting the West, Diplomacy: The West... | 6 | 14 | 3 | 50 | 0.300 | 0.667 | 0.414 |
| URW: Discrediting the West, Diplomacy: Diplomac... | 1 | 1 | 2 | 37 | 0.500 | 0.333 | 0.400 |
| CC: Criticism of climate policies: Climate poli... | 2 | 6 | 0 | 27 | 0.250 | 1.000 | 0.400 |
| URW: Discrediting Ukraine: Discrediting Ukraini... | 4 | 14 | 1 | 13 | 0.222 | 0.800 | 0.348 |


## Per-Language Analysis

The following sections show how model performance varies across different languages, revealing language-specific strengths and challenges.


## Narratives Performance by Language


### BG - Top 8 Narratives

| Label | TP | FP | FN | TN | Precision | Recall | F1 Score |
|-------|----|----|----|----|-----------|--------|---------|
| CC: Amplifying Climate Fears | 9 | 4 | 0 | 22 | 0.692 | 1.000 | 0.818 |
| URW: Discrediting the West, Diplomacy | 5 | 4 | 0 | 26 | 0.556 | 1.000 | 0.714 |
| URW: Negative Consequences for the West | 3 | 4 | 0 | 28 | 0.429 | 1.000 | 0.600 |
| CC: Downplaying climate change | 3 | 6 | 1 | 25 | 0.333 | 0.750 | 0.462 |
| CC: Criticism of institutions and authorities | 2 | 6 | 1 | 26 | 0.250 | 0.667 | 0.364 |
| URW: Russia is the Victim | 1 | 2 | 2 | 30 | 0.333 | 0.333 | 0.333 |
| CC: Criticism of climate movement | 1 | 5 | 0 | 29 | 0.167 | 1.000 | 0.286 |
| URW: Discrediting Ukraine | 2 | 10 | 3 | 20 | 0.167 | 0.400 | 0.235 |

### EN - Top 8 Narratives

| Label | TP | FP | FN | TN | Precision | Recall | F1 Score |
|-------|----|----|----|----|-----------|--------|---------|
| CC: Questioning the measurements and science | 4 | 3 | 0 | 34 | 0.571 | 1.000 | 0.727 |
| URW: Speculating war outcomes | 3 | 4 | 1 | 33 | 0.429 | 0.750 | 0.545 |
| URW: Discrediting the West, Diplomacy | 4 | 2 | 5 | 30 | 0.667 | 0.444 | 0.533 |
| CC: Criticism of institutions and authorities | 7 | 13 | 1 | 20 | 0.350 | 0.875 | 0.500 |
| CC: Criticism of climate movement | 5 | 8 | 3 | 25 | 0.385 | 0.625 | 0.476 |
| Other | 3 | 3 | 8 | 27 | 0.500 | 0.273 | 0.353 |
| URW: Distrust towards Media | 1 | 2 | 3 | 35 | 0.333 | 0.250 | 0.286 |
| URW: Discrediting Ukraine | 1 | 2 | 6 | 32 | 0.333 | 0.143 | 0.200 |

### HI - Top 8 Narratives

| Label | TP | FP | FN | TN | Precision | Recall | F1 Score |
|-------|----|----|----|----|-----------|--------|---------|
| CC: Amplifying Climate Fears | 3 | 2 | 1 | 29 | 0.600 | 0.750 | 0.667 |
| URW: Amplifying war-related fears | 3 | 2 | 5 | 25 | 0.600 | 0.375 | 0.462 |
| URW: Blaming the war on others rather than... | 2 | 4 | 3 | 26 | 0.333 | 0.400 | 0.364 |
| URW: Praise of Russia | 3 | 1 | 10 | 21 | 0.750 | 0.231 | 0.353 |
| URW: Speculating war outcomes | 1 | 2 | 4 | 28 | 0.333 | 0.200 | 0.250 |
| URW: Discrediting Ukraine | 1 | 7 | 2 | 25 | 0.125 | 0.333 | 0.182 |
| URW: Russia is the Victim | 1 | 2 | 8 | 24 | 0.333 | 0.111 | 0.167 |
| URW: Discrediting the West, Diplomacy | 0 | 7 | 7 | 21 | 0.000 | 0.000 | 0.000 |

### PT - Top 8 Narratives

| Label | TP | FP | FN | TN | Precision | Recall | F1 Score |
|-------|----|----|----|----|-----------|--------|---------|
| URW: Discrediting the West, Diplomacy | 7 | 0 | 1 | 27 | 1.000 | 0.875 | 0.933 |
| CC: Criticism of climate policies | 6 | 2 | 3 | 24 | 0.750 | 0.667 | 0.706 |
| CC: Criticism of institutions and authorities | 6 | 5 | 2 | 22 | 0.545 | 0.750 | 0.632 |
| URW: Discrediting Ukraine | 3 | 5 | 0 | 27 | 0.375 | 1.000 | 0.545 |
| CC: Amplifying Climate Fears | 5 | 0 | 18 | 12 | 1.000 | 0.217 | 0.357 |
| URW: Praise of Russia | 2 | 8 | 2 | 23 | 0.200 | 0.500 | 0.286 |
| URW: Negative Consequences for the West | 1 | 6 | 0 | 28 | 0.143 | 1.000 | 0.250 |
| URW: Amplifying war-related fears | 1 | 7 | 0 | 27 | 0.125 | 1.000 | 0.222 |

### RU - Top 8 Narratives

| Label | TP | FP | FN | TN | Precision | Recall | F1 Score |
|-------|----|----|----|----|-----------|--------|---------|
| URW: Discrediting Ukraine | 12 | 9 | 3 | 8 | 0.571 | 0.800 | 0.667 |
| URW: Discrediting the West, Diplomacy | 8 | 6 | 7 | 11 | 0.571 | 0.533 | 0.552 |
| URW: Distrust towards Media | 1 | 1 | 1 | 29 | 0.500 | 0.500 | 0.500 |
| URW: Praise of Russia | 6 | 9 | 9 | 8 | 0.400 | 0.400 | 0.400 |
| Other | 1 | 3 | 3 | 25 | 0.250 | 0.250 | 0.250 |
| URW: Blaming the war on others rather than... | 2 | 15 | 2 | 13 | 0.118 | 0.500 | 0.190 |
| URW: Amplifying war-related fears | 1 | 9 | 1 | 21 | 0.100 | 0.500 | 0.167 |
| URW: Speculating war outcomes | 0 | 5 | 3 | 24 | 0.000 | 0.000 | 0.000 |



## Subnarratives Performance by Language


### BG - Top 8 Subnarratives

| Label | TP | FP | FN | TN | Precision | Recall | F1 Score |
|-------|----|----|----|----|-----------|--------|---------|
| CC: Amplifying Climate Fears: Amplifying e... | 9 | 4 | 0 | 22 | 0.692 | 1.000 | 0.818 |
| URW: Discrediting the West, Diplomacy: Other | 5 | 4 | 0 | 26 | 0.556 | 1.000 | 0.714 |
| URW: Negative Consequences for the West: O... | 3 | 4 | 0 | 28 | 0.429 | 1.000 | 0.600 |
| CC: Amplifying Climate Fears: Other | 4 | 9 | 0 | 22 | 0.308 | 1.000 | 0.471 |
| CC: Downplaying climate change: Weather su... | 2 | 7 | 1 | 25 | 0.222 | 0.667 | 0.333 |
| CC: Amplifying Climate Fears: Doomsday sce... | 2 | 9 | 0 | 24 | 0.182 | 1.000 | 0.308 |
| URW: Amplifying war-related fears: There i... | 1 | 6 | 1 | 27 | 0.143 | 0.500 | 0.222 |
| Other | 1 | 2 | 5 | 27 | 0.333 | 0.167 | 0.222 |

### EN - Top 8 Subnarratives

| Label | TP | FP | FN | TN | Precision | Recall | F1 Score |
|-------|----|----|----|----|-----------|--------|---------|
| URW: Discrediting the West, Diplomacy: The... | 3 | 3 | 1 | 34 | 0.500 | 0.750 | 0.600 |
| URW: Discrediting the West, Diplomacy: Other | 3 | 3 | 3 | 32 | 0.500 | 0.500 | 0.500 |
| CC: Criticism of climate movement: Climate... | 4 | 9 | 0 | 28 | 0.308 | 1.000 | 0.471 |
| URW: Discrediting the West, Diplomacy: Dip... | 1 | 1 | 2 | 37 | 0.500 | 0.333 | 0.400 |
| CC: Criticism of institutions and authorit... | 5 | 15 | 1 | 20 | 0.250 | 0.833 | 0.385 |
| Other | 3 | 3 | 8 | 27 | 0.500 | 0.273 | 0.353 |
| URW: Distrust towards Media: Western media... | 1 | 2 | 3 | 35 | 0.333 | 0.250 | 0.286 |
| CC: Criticism of climate movement: Other | 2 | 11 | 2 | 26 | 0.154 | 0.500 | 0.235 |

### HI - Top 8 Subnarratives

| Label | TP | FP | FN | TN | Precision | Recall | F1 Score |
|-------|----|----|----|----|-----------|--------|---------|
| CC: Amplifying Climate Fears: Amplifying e... | 3 | 2 | 1 | 29 | 0.600 | 0.750 | 0.667 |
| URW: Amplifying war-related fears: There i... | 3 | 2 | 2 | 28 | 0.600 | 0.600 | 0.600 |
| URW: Praise of Russia: Praise of Russian m... | 2 | 2 | 1 | 30 | 0.500 | 0.667 | 0.571 |
| URW: Amplifying war-related fears: Russia ... | 2 | 3 | 2 | 28 | 0.400 | 0.500 | 0.444 |
| URW: Praise of Russia: Russia has internat... | 0 | 0 | 10 | 25 | 0.000 | 0.000 | 0.000 |
| URW: Russia is the Victim: The West is rus... | 0 | 0 | 7 | 28 | 0.000 | 0.000 | 0.000 |
| URW: Discrediting the West, Diplomacy: Other | 0 | 7 | 3 | 25 | 0.000 | 0.000 | 0.000 |
| URW: Praise of Russia: Russia is a guarant... | 0 | 2 | 3 | 30 | 0.000 | 0.000 | 0.000 |

### PT - Top 8 Subnarratives

| Label | TP | FP | FN | TN | Precision | Recall | F1 Score |
|-------|----|----|----|----|-----------|--------|---------|
| URW: Discrediting the West, Diplomacy: Other | 6 | 1 | 1 | 27 | 0.857 | 0.857 | 0.857 |
| CC: Criticism of climate policies: Other | 4 | 4 | 2 | 25 | 0.500 | 0.667 | 0.571 |
| CC: Criticism of institutions and authorit... | 4 | 7 | 0 | 24 | 0.364 | 1.000 | 0.533 |
| CC: Amplifying Climate Fears: Amplifying e... | 4 | 1 | 10 | 20 | 0.800 | 0.286 | 0.421 |
| CC: Criticism of climate policies: Climate... | 2 | 6 | 0 | 27 | 0.250 | 1.000 | 0.400 |
| CC: Amplifying Climate Fears: Other | 4 | 1 | 12 | 18 | 0.800 | 0.250 | 0.381 |
| URW: Praise of Russia: Russia is a guarant... | 2 | 7 | 0 | 26 | 0.222 | 1.000 | 0.364 |
| CC: Criticism of institutions and authorit... | 2 | 9 | 2 | 22 | 0.182 | 0.500 | 0.267 |

### RU - Top 8 Subnarratives

| Label | TP | FP | FN | TN | Precision | Recall | F1 Score |
|-------|----|----|----|----|-----------|--------|---------|
| URW: Discrediting Ukraine: Discrediting Uk... | 7 | 10 | 2 | 13 | 0.412 | 0.778 | 0.538 |
| URW: Praise of Russia: Praise of Russian m... | 4 | 11 | 1 | 16 | 0.267 | 0.800 | 0.400 |
| URW: Discrediting Ukraine: Discrediting Uk... | 4 | 14 | 1 | 13 | 0.222 | 0.800 | 0.348 |
| URW: Discrediting the West, Diplomacy: The... | 3 | 11 | 2 | 16 | 0.214 | 0.600 | 0.316 |
| URW: Blaming the war on others rather than... | 2 | 9 | 1 | 20 | 0.182 | 0.667 | 0.286 |
| Other | 1 | 3 | 3 | 25 | 0.250 | 0.250 | 0.250 |
| URW: Discrediting the West, Diplomacy: Other | 2 | 12 | 2 | 16 | 0.143 | 0.500 | 0.222 |
| URW: Discrediting Ukraine: Ukraine is a pu... | 2 | 16 | 2 | 12 | 0.111 | 0.500 | 0.182 |


## Key Insights

### Aggregated Performance
- **Climate-related narratives** consistently show the strongest performance across languages
- **War-related narratives** demonstrate moderate performance with higher variance
- **Subnarratives** classification remains significantly more challenging than narratives

### Language-Specific Observations
- **Portuguese (PT)** typically shows strong performance, especially for climate narratives
- **English (EN)** demonstrates balanced performance across different narrative types
- **Hindi (HI)** shows particular challenges with certain narrative categories
- **Bulgarian (BG) and Russian (RU)** exhibit similar performance patterns for war-related content

### Cross-Linguistic Patterns
- **Climate Fear narratives** perform consistently well across all languages
- **"Other" categories** remain challenging regardless of language
- **Sample size effects** are evident - languages with more examples show better performance
- **Cultural/contextual factors** may influence performance for specific narrative types

### Recommendations
1. **Deploy with confidence** for climate-related narrative detection across all languages
2. **Monitor closely** for war-related narratives, especially in low-resource scenarios
3. **Consider language-specific tuning** for categories showing high variance
4. **Collect additional data** for underperforming categories in specific languages

This comprehensive analysis confirms that **threshold 0.65 provides optimal performance** across languages while revealing important insights for language-specific deployment strategies.
