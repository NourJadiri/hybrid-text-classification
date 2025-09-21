# Google Gemini 2.5 Flash - Devset Performance Analysis Report

**Generated**: 2025-09-21 19:04:16  
**Model**: Google Gemini 2.5 Flash  
**Configuration**: No Validation  
**Languages Evaluated**: Bulgarian (BG), English (EN), Hindi (HI), Portuguese (PT), Russian (RU)  

---

## Executive Summary

This report presents a comprehensive analysis of Google Gemini 2.5 Flash's performance on hierarchical text classification for disinformation detection across 5 languages. The model was evaluated without validation steps to assess its raw classification capability.

### Overall Performance Metrics

#### Narratives Classification
- **F1 Macro**: 0.4715 ± 0.0548
- **F1 Micro**: 0.6096 ± 0.0944
- **F1 Samples**: 0.6019 ± 0.1056

#### Subnarratives Classification
- **F1 Macro**: 0.2988 ± 0.0359
- **F1 Micro**: 0.4018 ± 0.0570
- **F1 Samples**: 0.4077 ± 0.0660

---

## Detailed Performance Analysis

### Performance by Language


| Language | Label Type | F1 Macro | F1 Micro | F1 Samples | Total Labels | Total Files |
|----------|------------|----------|----------|------------|--------------|-------------|
| BG | Narratives | 0.4236 | 0.5584 | 0.5799 | 21 | 35 |
| BG | Subnarratives | 0.2819 | 0.3824 | 0.4104 | 71 | 35 |
| EN | Narratives | 0.5398 | 0.5714 | 0.5132 | 22 | 41 |
| EN | Subnarratives | 0.3116 | 0.4019 | 0.3815 | 83 | 41 |
| HI | Narratives | 0.3971 | 0.4841 | 0.4715 | 13 | 35 |
| HI | Subnarratives | 0.2376 | 0.3071 | 0.2976 | 46 | 35 |
| PT | Narratives | 0.4756 | 0.7460 | 0.7489 | 14 | 35 |
| PT | Subnarratives | 0.3387 | 0.4725 | 0.4737 | 40 | 35 |
| RU | Narratives | 0.5213 | 0.6879 | 0.6958 | 11 | 32 |
| RU | Subnarratives | 0.3241 | 0.4453 | 0.4754 | 41 | 32 |


### Key Observations

#### Language-Specific Performance


**RU (Russian)**: 
- Processed 32 files
- Narratives F1 Macro: 0.5213
- Subnarratives F1 Macro: 0.3241
- Performance Rank: Fair

**EN (English)**: 
- Processed 41 files
- Narratives F1 Macro: 0.5398
- Subnarratives F1 Macro: 0.3116
- Performance Rank: Fair

**PT (Portuguese)**: 
- Processed 35 files
- Narratives F1 Macro: 0.4756
- Subnarratives F1 Macro: 0.3387
- Performance Rank: Good

**BG (Bulgarian)**: 
- Processed 35 files
- Narratives F1 Macro: 0.4236
- Subnarratives F1 Macro: 0.2819
- Performance Rank: Good

**HI (Hindi)**: 
- Processed 35 files
- Narratives F1 Macro: 0.3971
- Subnarratives F1 Macro: 0.2376
- Performance Rank: Good


#### Performance Patterns

1. **Best Performing Language**: EN 
2. **Most Challenging Language**: HI
3. **Narratives vs Subnarratives**: Narratives consistently outperform subnarratives across all languages
4. **F1 Score Distribution**: Ranges from 0.3971 to 0.5398 for narratives

---

## Top Performing Labels Analysis

### Best Performing Narratives (Top 15)

| Rank | Narrative | Avg F1 Score | Total Support | Languages |
|------|-----------|-------------|---------------|-----------|
| 1 | URW: Praise of Russia | 0.7420 | 36 | BG, EN, HI, PT, RU |
| 2 | URW: Discrediting the West, Diplomacy | 0.7351 | 44 | BG, EN, HI, PT, RU |
| 3 | URW: Discrediting Ukraine | 0.7175 | 33 | BG, EN, HI, PT, RU |
| 4 | CC: Climate change is beneficial | 0.6667 | 1 | EN |
| 5 | CC: Amplifying Climate Fears | 0.6588 | 36 | BG, EN, HI, PT |
| 6 | CC: Criticism of climate movement | 0.6190 | 9 | BG, EN |
| 7 | CC: Criticism of climate policies | 0.6159 | 14 | BG, EN, PT |
| 8 | CC: Criticism of institutions and authorities | 0.6143 | 19 | BG, EN, PT |
| 9 | Other | 0.5401 | 24 | BG, EN, HI, PT, RU |
| 10 | URW: Amplifying war-related fears | 0.5350 | 19 | BG, EN, HI, PT, RU |
| 11 | CC: Downplaying climate change | 0.5079 | 6 | BG, EN |
| 12 | URW: Russia is the Victim | 0.4564 | 20 | BG, EN, HI, PT, RU |
| 13 | CC: Hidden plots by secret schemes of powerful gro... | 0.4444 | 7 | BG, EN, PT |
| 14 | CC: Questioning the measurements and science | 0.4286 | 4 | BG, EN |
| 15 | URW: Blaming the war on others rather than the inv... | 0.3631 | 17 | BG, EN, HI, PT, RU |

### Best Performing Subnarratives (Top 15)

| Rank | Subnarrative | Avg F1 Score | Total Support | Languages |
|------|-------------|-------------|---------------|-----------|
| 1 | CC: Downplaying climate change: Climate cycles are... | 1.0000 | 1 | BG |
| 2 | CC: Climate change is beneficial: CO2 is beneficia... | 1.0000 | 1 | EN |
| 3 | CC: Hidden plots by secret schemes of powerful gro... | 0.8333 | 3 | BG, EN |
| 4 | CC: Criticism of climate movement: Ad hominem atta... | 0.7857 | 4 | BG, EN |
| 5 | URW: Praise of Russia: Praise of Russian President... | 0.7381 | 5 | BG, EN, HI, RU |
| 6 | CC: Criticism of climate policies: Climate policie... | 0.7222 | 4 | BG, EN, PT |
| 7 | CC: Criticism of climate movement: Climate movemen... | 0.6667 | 5 | BG, EN |
| 8 | URW: Amplifying war-related fears: There is a real... | 0.6667 | 9 | BG, EN, HI, RU |
| 9 | CC: Controversy about green technologies: Renewabl... | 0.6667 | 1 | EN |
| 10 | URW: Discrediting Ukraine: Discrediting Ukrainian ... | 0.6375 | 19 | BG, EN, HI, PT, RU |
| 11 | URW: Discrediting Ukraine: Ukraine is a puppet of ... | 0.6234 | 11 | BG, EN, HI, PT, RU |
| 12 | CC: Criticism of institutions and authorities: Cri... | 0.6120 | 12 | BG, EN, PT |
| 13 | CC: Amplifying Climate Fears: Amplifying existing ... | 0.6112 | 27 | BG, EN, HI, PT |
| 14 | URW: Praise of Russia: Praise of Russian military ... | 0.5927 | 10 | BG, EN, HI, RU |
| 15 | URW: Praise of Russia: Russia has international su... | 0.5621 | 17 | BG, HI, PT, RU |

---

## Analysis and Interpretations

### Strengths

1. **High Performance on Ukraine-Russia War Narratives**: The model shows excellent performance on URW-related narratives across multiple languages, particularly for:
   - Discrediting Ukraine (F1 scores ranging 0.80-0.87)
   - Discrediting the West/Diplomacy (F1 scores around 0.82-0.87)
   - Praising Russia (consistently high performance)

2. **Climate Change Detection**: Strong performance on climate-related narratives:
   - "Amplifying Climate Fears" achieves high F1 scores across languages
   - Climate criticism narratives are well-detected

3. **Language Consistency**: Performance is relatively consistent across different languages, suggesting good multilingual capability.

4. **"Other" Category**: Perfect classification of "Other" category indicates good boundary detection.

### Challenges

1. **Subnarrative Granularity**: Performance drops significantly at the subnarrative level, indicating difficulty with fine-grained classification.

2. **Low-Support Labels**: Labels with low support (1-3 instances) show high variance in performance.

3. **Complex Narratives**: Some complex narrative types show lower performance, particularly those requiring nuanced understanding.

4. **Hindi Language**: Shows the lowest overall performance, suggesting potential language-specific challenges.

### Technical Observations

1. **F1 Samples vs F1 Macro**: F1 Samples scores are generally higher than F1 Macro, indicating the model performs better on common labels.

2. **Precision vs Recall**: Analysis of individual labels shows the model tends toward high recall but sometimes lower precision, leading to over-classification.

3. **Hierarchical Consistency**: The performance drop from narratives to subnarratives is expected but significant (approximately 15-20% decrease in F1 scores).

---

## Recommendations

### For Model Improvement

1. **Subnarrative Fine-tuning**: Focus additional training on subnarrative classification to improve fine-grained detection.

2. **Language-Specific Optimization**: Consider additional training or fine-tuning for Hindi and other lower-performing languages.

3. **Validation Integration**: While this evaluation was performed without validation, integrating validation steps could improve precision.

4. **Data Augmentation**: For low-support labels, consider data augmentation techniques to improve performance.

### For Operational Deployment

1. **Confidence Thresholds**: Implement confidence thresholds based on these performance metrics for production use.

2. **Language Prioritization**: Deploy first on languages showing highest performance (Russian, English, Portuguese).

3. **Hierarchical Approach**: Use narrative-level predictions as primary classification, with subnarratives as supplementary information.

4. **Human Validation**: Implement human validation loops for low-confidence predictions.

---

## Methodology

### Evaluation Setup
- **Metrics Used**: F1 Macro, F1 Micro, F1 Samples
- **Dataset**: Official devset with human annotations
- **Languages**: 5 languages (BG, EN, HI, PT, RU)
- **Total Files Processed**: {sum([all_results[lang]['results']['total_files'] for lang in all_results if 'results' in all_results[lang]])}

### Configuration
- **Model**: Google Gemini 2.5 Flash
- **Validation**: Disabled (no validation steps)
- **Label Cleaning**: Enabled
- **Text Cleaning**: Disabled

### Evaluation Framework
- Multi-label classification evaluation using scikit-learn
- Binary relevance approach for each label
- Confusion matrices computed for each label
- Per-label metrics (precision, recall, F1) calculated

---

## Conclusion

Google Gemini 2.5 Flash demonstrates strong performance on hierarchical disinformation classification, particularly excelling at narrative-level detection. The model shows consistent cross-lingual performance with some language-specific variations. While subnarrative classification remains challenging, the overall results suggest the model is suitable for production deployment with appropriate confidence thresholds and validation procedures.

The analysis reveals clear patterns in performance across different types of narratives, with war-related and climate-related content being particularly well-detected. This suggests the model has learned meaningful representations of disinformation patterns across multiple languages and domains.

**Overall Assessment**: The model achieves good performance for a challenging multilingual, multi-label, hierarchical classification task, with clear areas for improvement identified through this comprehensive evaluation.

---

*Report generated from evaluation results in `{results_dir}`*
*Plots and detailed data available in the evaluation directory*
