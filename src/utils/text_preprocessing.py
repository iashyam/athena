from collections import Counter

def build_vocabulary(sentences, min_count=2):

    word_counts = Counter()

    for sentence in sentences:
        words = sentence.lower().split()
        word_counts.update(words)

    vocab = {"PAD": 0, "UNK": 1}
    for word, count in word_counts.items():
        if count >= min_count:
            vocab[word] = len(vocab)

    return vocab