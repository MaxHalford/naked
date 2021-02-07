def tfidf_vectorizer(x, lowercase, norm, vocabulary_, idf_):

    import re

    if lowercase:
        x = x.lower()

    # Tokenize
    x = re.findall(r"(?u)\b\w\w+\b", x)
    x = [xi for xi in x if len(xi) > 1]

    # Count term frequencies
    from collections import Counter
    tf = Counter(x)
    total = sum(tf.values())

    # Compute the TF-IDF of each tokenized term
    tfidf = [0] * len(vocabulary_)
    for term, freq in tf.items():
        try:
            index = vocabulary_[term]
        except KeyError:
            continue
        tfidf[index] = freq * idf_[index] / total

    # Apply normalization
    if norm == 'l2':
        norm_val = sum(xi ** 2 for xi in tfidf) ** .5

    return [v / norm_val for v in tfidf]
