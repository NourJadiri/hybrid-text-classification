from sklearn.preprocessing import MultiLabelBinarizer


def binarize_labels(labels, all_labels):
    """
    Convert a list of label ids to a binary vector using sklearn's MultiLabelBinarizer.
    The output vector is ordered according to sorted(all_labels).
    """
    mlb = MultiLabelBinarizer(classes=sorted(all_labels))
    # Fit and transform expects a list of label lists
    return mlb.fit(sorted(all_labels)).transform([labels])[0]


if __name__ == "__main__":
    # Example usage
    all_labels = ['cat', 'dog', 'mouse']
    labels = ['dog', 'cat']
    binary_vector = binarize_labels(labels, all_labels)
    print(f"Labels: {labels}")
    print(f"All labels: {all_labels}")
    print(f"Binarized vector: {binary_vector}")
