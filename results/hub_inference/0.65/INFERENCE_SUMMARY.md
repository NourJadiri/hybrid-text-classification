# Inference Results Summary

## Model Used
- **Model Name**: AWCO/mdeberta-v3-base-narratives-classifier-hierarchical
- **Model Type**: Multi-head DeBERTa for Hierarchical Classification
- **Threshold**: 0.650 (loaded from training configuration)

## Inference Statistics

| Language | Documents Processed | 'Other' Classifications | Classified Documents |
|----------|-------------------|----------------------|-------------------|
| BG       | 35                | 3                    | 32                |
| EN       | 41                | 6                    | 35                |
| HI       | 35                | 7                    | 28                |
| PT       | 35                | 7                    | 28                |
| RU       | 32                | 4                    | 28                |
| **Total**| **178**           | **27**               | **151**           |

## Classification Performance
- **Overall Classification Rate**: 84.8% (151/178 documents received specific classifications)
- **Other Rate**: 15.2% (27/178 documents classified as "Other")

## Output Format
Each prediction file contains tab-separated values with the following columns:
1. **Filename**: Original document filename
2. **Narratives**: Parent-level classifications (semicolon-separated)
3. **Sub-narratives**: Child-level classifications (semicolon-separated)

## Sample Classifications

### Climate Change (CC) Examples:
- Documents showing multiple CC narratives like "Criticism of climate movement", "Downplaying climate change", "Hidden plots by secret schemes"
- Rich sub-narrative predictions with specific aspects like "Climate policies have negative impact on the economy"

### Ukraine-Russia War (URW) Examples:
- Complex narratives including "Discrediting Ukraine", "Praise of Russia", "Speculating war outcomes"
- Detailed sub-narratives like "Ukraine is associated with nazism", "Russia is a guarantor of peace and prosperity"

## Technical Details
- **Device**: CUDA (GPU acceleration)
- **Inference Speed**: ~5-6 documents per second
- **Model Size**: 1.11GB
- **Downloaded from**: HuggingFace Hub cache

## Files Generated
- `BG_predictions.tsv` - Bulgarian documents (35 predictions)
- `EN_predictions.tsv` - English documents (41 predictions) 
- `HI_predictions.tsv` - Hindi documents (35 predictions)
- `PT_predictions.tsv` - Portuguese documents (35 predictions)
- `RU_predictions.tsv` - Russian documents (32 predictions)

All results are saved in: `results/hub_inference/`