import extract_data
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

def most_common_ngram(data : pd.DataFrame, c_name : str, ngram_range : tuple, top : int = 10) -> None:
    if c_name == 'non_toxic':
        non_toxic = ((data['toxic'] == 0) & (data['severe_toxic'] == 0) & (data['obscene'] == 0) & (
                    data['threat'] == 0) &
                     (data['insult'] == 0) & (data['identity_hate'] == 0))
        corpus = data[non_toxic]['comment_text']
    elif c_name in data.columns[2:]:
        corpus = data[data[c_name] == 1]['comment_text']
    else:
        print("Invalid column name:", c_name)
        return

    vec = CountVectorizer(ngram_range=ngram_range, max_features=top).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    temp = words_freq[:top]
    temp = pd.DataFrame(temp)
    temp.rename(columns={0: "Words", 1: "Count"}, inplace=True)
    print(f"Top {top} {ngram_range[0]}-grams to {ngram_range[0]}-grams in {c_name}:")
    print(temp.head(top))

def most_common_all(data : pd.DataFrame, ngram_range: tuple, top : int = 10) -> None:
    for i in data.columns[2:]:
        most_common_ngram(data, i, ngram_range, top)
        print('-----------------------------')
    most_common_ngram(data, 'non_toxic', ngram_range, top)

def comment_length(data : pd.DataFrame, c_name : str):
    if c_name == 'non_toxic':
        non_toxic = ((data['toxic'] == 0) & (data['severe_toxic'] == 0) & (data['obscene'] == 0) & (
                data['threat'] == 0) &
                     (data['insult'] == 0) & (data['identity_hate'] == 0))
        corpus = data[non_toxic]['comment_text']
    elif c_name in data.columns[2:]:
        corpus = data[data[c_name] == 1]['comment_text']
    else:
        print("Invalid column name:", c_name)
        return

    data['len'] = corpus.apply(lambda i: len(i.split()))
    print(f"Numbers of words in {c_name}")
    print(data['len'].describe())

def comment_length_all(data : pd.DataFrame) -> None:
    for i in data.columns[2:]:
        comment_length(data, i)
        print('-----------------------------')
    comment_length(data, 'non_toxic')

if __name__ == '__main__':
    train = extract_data.extract('train.csv')
    #train = extract_data.clean(train)
    #most_common_all(train, (2, 2))
    comment_length_all(train)