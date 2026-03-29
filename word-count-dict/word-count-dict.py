def word_count_dict(sentences):
    freq = {}
    for sentence in sentences:
        for word in sentence:
            if word in freq:
                freq[word] += 1
            else:
                freq[word] = 1

    return freq
    pass