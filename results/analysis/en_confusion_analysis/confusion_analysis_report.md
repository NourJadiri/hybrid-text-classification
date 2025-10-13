# Multi-Label Confusion Analysis Report

## Label-wise Performance Statistics

| Label | True Positives | False Positives | False Negatives | Total True | Total Predicted | Precision | Recall | F1 |
|-------|----------------|-----------------|-----------------|------------|-----------------|-----------|-----------|-----|
| CC | 15 | 6 | 2 | 17 | 21 | 0.714 | 0.882 | 0.789 |
| Other | 4 | 2 | 7 | 11 | 6 | 0.667 | 0.364 | 0.471 |
| URW | 12 | 2 | 1 | 13 | 14 | 0.857 | 0.923 | 0.889 |

## Most Common Label Confusion Patterns

Shows cases where a true label is present but a different label is predicted.

| Rank | True Label | Wrongly Predicted Label | Count | Description |
|------|------------|------------------------|-------|-------------|
| 1 | Other | CC | 10 | Model predicted 'CC' when 'Other' was true |
| 2 | Other | URW | 4 | Model predicted 'URW' when 'Other' was true |
| 3 | CC | Other | 4 | Model predicted 'Other' when 'CC' was true |
| 4 | URW | CC | 2 | Model predicted 'CC' when 'URW' was true |

## Label-Specific Confusion Analysis


### CC

When **CC** is the true label, it is often confused with:

- **Other**: 4 times

### Other

When **Other** is the true label, it is often confused with:

- **CC**: 10 times
- **URW**: 4 times

### URW

When **URW** is the true label, it is often confused with:

- **CC**: 2 times
