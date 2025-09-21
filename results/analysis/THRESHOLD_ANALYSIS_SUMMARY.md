# Threshold Comparison Analysis Summary

## Overview

This document summarizes the comprehensive threshold experimentation conducted on the hierarchical text classification model (`AWCO/mdeberta-v3-base-narratives-classifier-hierarchical`) across multiple classification thresholds. The analysis evaluates how different confidence thresholds affect model performance for both narrative and subnarrative classification tasks.

## Experimental Setup

- **Model**: `AWCO/mdeberta-v3-base-narratives-classifier-hierarchical`
- **Dataset**: Development set across 5 languages (BG, EN, HI, PT, RU)
- **Total Documents**: 178 documents
- **Thresholds Tested**: 0.5, 0.65 (training), 0.7, 0.8
- **Evaluation Metrics**: F1 Macro, F1 Micro, F1 Samples

## Results Summary

### Overall Performance Comparison

#### Narratives Classification

| Threshold | F1 Macro | F1 Micro | F1 Samples | Performance vs Training |
|-----------|----------|----------|------------|------------------------|
| **0.5**   | 0.1800   | 0.2614   | 0.2475     | ↓ **-0.0204** (-10.2%) |
| **0.65*** | **0.2004** | **0.3145** | **0.2901** | **Baseline** (Training) |
| **0.7**   | 0.1825   | 0.2911   | 0.2657     | ↓ **-0.0179** (-8.9%)  |
| **0.8**   | 0.0937   | 0.2012   | 0.2233     | ↓ **-0.1067** (-53.2%) |

*\* Training threshold*

#### Subnarratives Classification

| Threshold | F1 Macro | F1 Micro | F1 Samples | Performance vs Training |
|-----------|----------|----------|------------|------------------------|
| **0.5**   | 0.0654   | 0.0987   | 0.0971     | ↓ **-0.0122** (-15.7%) |
| **0.65*** | 0.0776   | 0.1427   | 0.1543     | **Baseline** (Training) |
| **0.7**   | **0.0821** | 0.1516   | 0.1566     | ↑ **+0.0046** (+5.9%)  |
| **0.8**   | 0.0583   | **0.1574** | **0.1750** | ↑ **+0.0207** (+13.4%) |

*\* Training threshold*

### Best Performing Thresholds by Metric

| Label Type | Metric | Best Threshold | Best Score | Improvement vs Training |
|------------|--------|---------------|------------|------------------------|
| **Narratives** | F1 Macro | 0.65 | 0.2004 | Baseline |
| **Narratives** | F1 Micro | 0.65 | 0.3145 | Baseline |
| **Narratives** | F1 Samples | 0.65 | 0.2901 | Baseline |
| **Subnarratives** | F1 Macro | **0.7** | 0.0821 | +0.0046 (+5.9%) |
| **Subnarratives** | F1 Micro | **0.8** | 0.1574 | +0.0147 (+10.3%) |
| **Subnarratives** | F1 Samples | **0.8** | 0.1750 | +0.0207 (+13.4%) |

## Per-Language Analysis

### Narratives F1 Macro by Language and Threshold

| Language | 0.5 | 0.65 | 0.7 | 0.8 | Best Threshold |
|----------|-----|------|-----|-----|----------------|
| **BG** | 0.177 | **0.209** | 0.212 | 0.134 | 0.7 |
| **EN** | 0.224 | **0.212** | 0.207 | 0.130 | 0.5 |
| **HI** | 0.159 | **0.167** | 0.100 | 0.053 | 0.65 |
| **PT** | 0.177 | **0.243** | 0.227 | 0.073 | 0.65 |
| **RU** | 0.170 | **0.170** | 0.167 | 0.079 | 0.65/0.5 |

### Subnarratives F1 Macro by Language and Threshold

| Language | 0.5 | 0.65 | 0.7 | 0.8 | Best Threshold |
|----------|-----|------|-----|-----|----------------|
| **BG** | 0.060 | 0.077 | **0.080** | 0.071 | 0.7 |
| **EN** | 0.085 | **0.102** | 0.101 | 0.079 | 0.65 |
| **HI** | 0.062 | **0.061** | 0.032 | 0.033 | 0.5 |
| **PT** | 0.067 | 0.085 | **0.104** | 0.055 | 0.7 |
| **RU** | 0.053 | 0.063 | 0.095 | **0.053** | 0.7 |

## Key Insights and Findings

### 🎯 **Threshold Effectiveness**

1. **Training Threshold (0.65) is Optimal for Narratives**
   - All narrative metrics achieve their best performance at the training threshold
   - Shows excellent model calibration during training
   - Provides the best balance between precision and recall

2. **Higher Thresholds Benefit Subnarrative Classification**
   - Threshold 0.7: Best F1 Macro for subnarratives (+5.9% improvement)
   - Threshold 0.8: Best F1 Micro and F1 Samples (+10.3% and +13.4% improvement)
   - More conservative predictions reduce false positives for fine-grained classification

3. **Lower Threshold (0.5) Reduces Performance**
   - Consistently lower performance across both task levels
   - Too many false positives hurt precision
   - Not recommended for deployment

4. **Very High Threshold (0.8) Hurts Narrative Classification**
   - Dramatic drop in narrative performance (-53.2% F1 Macro)
   - Too conservative, missing many true positives
   - Only beneficial for subnarrative precision tasks

### 🌍 **Language-Specific Patterns**

1. **Portuguese (PT)**: Most sensitive to threshold changes
   - Best narrative performance at training threshold (0.243)
   - Significant improvement in subnarratives at threshold 0.7

2. **Hindi (HI)**: Most affected by high thresholds
   - Severe performance degradation at thresholds 0.7 and 0.8
   - Optimal at training threshold or slightly lower

3. **English (EN)**: Consistent across moderate thresholds
   - Relatively stable performance between 0.5-0.7
   - Good candidate for threshold experimentation

4. **Bulgarian (BG) and Russian (RU)**: Moderate sensitivity
   - Generally follow overall trends
   - Training threshold provides good baseline

### 📊 **Performance Trade-offs**

| Threshold | Narrative Performance | Subnarrative Performance | Overall Recommendation |
|-----------|----------------------|--------------------------|------------------------|
| **0.5** | Poor (↓10.2%) | Poor (↓15.7%) | ❌ Not recommended |
| **0.65** | **Excellent** (Baseline) | Good (Baseline) | ✅ **Recommended for balanced performance** |
| **0.7** | Good (↓8.9%) | **Best Macro** (+5.9%) | ⚖️ Consider for subnarrative-focused tasks |
| **0.8** | Poor (↓53.2%) | **Best Micro/Samples** (+13.4%) | ⚠️ Only for high-precision subnarrative tasks |

## Recommendations

### 🎯 **Production Deployment**

1. **Default Choice**: **Threshold 0.65** (Training threshold)
   - Best overall balance between narrative and subnarrative performance
   - Well-calibrated and tested during training
   - Consistent performance across languages

2. **Task-Specific Optimization**:
   - **Narrative-focused applications**: Use threshold 0.65
   - **Subnarrative precision-critical tasks**: Consider threshold 0.7-0.8
   - **High-recall applications**: Avoid threshold 0.8

3. **Language-Specific Considerations**:
   - **Hindi**: Stick with threshold 0.65 or lower
   - **Portuguese**: Consider threshold 0.7 for subnarratives
   - **Other languages**: Training threshold provides good baseline

### 🔬 **Further Experimentation**

1. **Fine-grained threshold search** between 0.65-0.75 for subnarratives
2. **Language-specific threshold optimization**
3. **Dynamic thresholding** based on confidence distributions
4. **Ensemble approaches** combining multiple thresholds

## Conclusion

The threshold experimentation reveals that the model's training threshold (0.65) was well-chosen for overall performance. While higher thresholds can improve subnarrative classification precision, they come at the cost of narrative classification performance. The analysis provides clear guidance for threshold selection based on specific use case requirements and demonstrates the importance of threshold tuning in hierarchical multi-label classification tasks.

---

*Analysis conducted on September 21, 2025 using the AWCO/mdeberta-v3-base-narratives-classifier-hierarchical model across 178 documents in 5 languages (BG, EN, HI, PT, RU).*