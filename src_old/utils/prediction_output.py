def write_predictions_to_txt(predictor, articles_df, output_file):
    """
    Runs batch prediction and writes results to a .txt file (tab-separated, no header).
    Each line: article_id, narratives, subnarratives
    Args:
        predictor: The NarrativePredictor instance
        articles_df: DataFrame with columns 'article_id' and 'text'
        output_file: Path to output .txt file
    """
    texts = articles_df['text'].tolist()
    article_ids = articles_df['article_id'].tolist()
    predictions = predictor.predict_batch(texts)
    with open(output_file, 'w', encoding='utf-8') as f:
        for article_id, pred in zip(article_ids, predictions):
            narratives = pred.get('narratives', [])
            subnarratives = pred.get('subnarratives', [])
            narr_str = ';'.join(str(n) for n in narratives) if isinstance(narratives, list) else ''
            subnarr_str = ';'.join(str(s) for s in subnarratives) if isinstance(subnarratives, list) else ''
            f.write(f"{article_id}\t{narr_str}\t{subnarr_str}\n")
