import pandas as pd
import matplotlib.pyplot as plt
# import nltk
from nltk.corpus import stopwords

# nltk.download('stopwords')
global_path = "../data/"

def clean(data : pd.DataFrame) -> pd.DataFrame:
    def clean_text(text : str) -> str:
        stop = set(stopwords.words('english'))
        text = ' '.join(word for word in text.split() if word not in stop)
        return text

    data['comment_text'] = data['comment_text'].map(lambda i : clean_text(i))
    return data

def extract(filename : str,) -> pd.DataFrame:
    # Just grabs the data from a specified file
    global global_path
    return pd.read_csv(global_path + filename)

def print_csv(data : pd.DataFrame, n_rows : int = 10) -> None:
    # Prints out 10 rows if not specified otherwise
    pd.set_option('display.max_columns', 10)
    pd.set_option('display.width', 1000)
    print(data.head(n_rows))

def plot_frequency(data : pd.DataFrame) -> None:
    # Plots both the number of each type of toxic comment and the non-toxic to toxic comment graphs
    frequency = {}
    for i in data.columns[2:]:
        frequency[i] = len(data[data[i] == 1])
    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(15, 5)
    bars = ax[0].bar(frequency.keys(), frequency.values())
    ax[0].set_xlabel("Category of comment")
    ax[0].set_ylabel("# of appearances")
    ax[0].bar_label(bars, label = [count for count in frequency.values()])

    non_toxic = ((data['toxic'] == 0) & (data['severe_toxic'] == 0) & (data['obscene'] == 0) & (data['threat'] == 0) &
                 (data['insult'] == 0) & (data['identity_hate'] == 0))
    frequency_general = {"non_toxic": len(data[non_toxic]), "toxic": len(data) - len(data[non_toxic])}
    bar1 = ax[1].bar(frequency_general.keys(), frequency_general.values())
    ax[1].set_xlabel("General category of comment")
    ax[1].set_ylabel("# of appearances")
    ax[1].bar_label(bar1, label=[count for count in frequency_general.values()])
    plt.show()

if __name__ == "__main__":
    train = extract("train.csv")
    #print_csv(train)
    #plot_frequency(train)
    #train = clean(train)