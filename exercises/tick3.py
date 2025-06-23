from utils.sentiment_detection import clean_plot, chart_plot, best_fit
from typing import List, Tuple, Callable
from collections import Counter
import os
from math import log

def estimate_zipf(token_frequencies_log: List[Tuple[float, float]], token_frequencies: List[Tuple[int, int]]) \
        -> Callable:
    """
    Use the provided least squares algorithm to estimate a line of best fit in the log-log plot of rank against
    frequency. Weight each word by its frequency to avoid distortion in favour of less common words. Use this to
    create a function which given a rank can output an expected frequency.

    @param token_frequencies_log: list of tuples of log rank and log frequency for each word
    @param token_frequencies: list of tuples of rank to frequency for each word used for weighting
    @return: a function estimating a word's frequency from its rank
    """
    m,c = best_fit(token_frequencies_log, token_frequencies)
    def ef(rank):
        return 2.718281828 **(m * log(rank) + c)
    return ef


def count_token_frequencies(dataset_path: str) -> List[Tuple[str, int]]:
    """
    For each of the words in the dataset, calculate its frequency within the dataset.

    @param dataset_path: a path to a folder with a list of  reviews
    @returns: a list of the frequency for each word in the form [(word, frequency), (word, frequency) ...], sorted by
        frequency in descending order
    """
    temp = {}
    for file in os.listdir(dataset_path):
        file_path = f"{dataset_path}\{file}"
        with open(file_path, encoding="utf-8") as f:
            txt = f.readlines()
        print(file_path)
        for line in txt:
            l = line.strip().split(" ")
            cl = Counter(l)
            for i in cl:
                if i in temp:
                    temp[i] += cl[i]
                else:
                    temp[i] = cl[i]
    ret = []
    for i, c in temp.items():
        ret.append((i, c))
    ret.sort(key=lambda x: x[1], reverse=True)
    return ret


def draw_frequency_ranks(frequencies: List[Tuple[str, int]]) -> None:
    """
    Use the provided chart plotting program to plot the most common 10000 word ranks against their frequencies.

    @param frequencies: A sorted list of tuples with the first element being the word and second its frequency
    """
    data = []
    n = len(frequencies)
    for i in range(n):
        data.append((i+1, frequencies[i][1]))
    chart_plot(data, "frequency vs rank", "rank", "frequency")


def draw_selected_words_ranks(frequencies: List[Tuple[str, int]]) -> None:
    """
    Use the chart plotting program to plot your 10 selected words' word frequencies (from Task 1) against their
    ranks. Plot the Task 1 words on the frequency-rank plot as a separate series on the same plot (i.e., tell the
    plotter to draw it as an additional graph on top of the graph from above function).

    @param frequencies: A sorted list of tuples with the first element being the word and second its frequency
    """
    swords = ["amazing","wonderful","good","great","awesome","bland","dull","dry","bad","lacking"]
    data = []
    frequencies = [[i[0],i[1]] for i in frequencies]
    n = len(frequencies)
    for i in range(n):
        frequencies[i][1] = [frequencies[i][1],i+1]
    frequencies = dict(frequencies)
    for i in swords:
        data.append((frequencies[i][1],frequencies[i][0]))
    chart_plot(data, "frequency vs rank", "rank", "frequency")


def draw_zipf(frequencies: List[Tuple[str, int]]) -> None:
    """
    Use the chart plotting program to plot the logs of your first 10000 word frequencies against the logs of their
    ranks. Also use your estimation function to plot a line of best fit.

    @param frequencies: A sorted list of tuples with the first element being the word and second its frequency
    """
    data = []
    unlogdata = []
    for i in range(10000):
        data.append((log(i+1), log(frequencies[i][1])))
        unlogdata.append((i+1, frequencies[i][1]))
    chart_plot(data, "log of vs log rank", "log rank", "log frequency")
    cdata = []
    m,c = best_fit(data,unlogdata)
    print(-1*m,2.718281828**c)
    for i in range(0,9,2):
        cdata.append((i,m*i+c))
    chart_plot(cdata, "log of vs log rank", "log rank", "log frequency")

    esti = estimate_zipf(data,unlogdata)
    swords = ["amazing", "wonderful", "good", "great", "awesome", "bland", "dull", "dry", "bad", "lacking"]
    frequencies = [[i[0], i[1]] for i in frequencies]
    n = len(frequencies)
    for i in range(n):
        frequencies[i][1] = [frequencies[i][1], i + 1]
    frequencies = dict(frequencies)
    for i in swords:
        print(i, frequencies[i][1], esti(frequencies[i][1]), frequencies[i][0])





def compute_type_count(dataset_path: str) -> List[Tuple[int, int]]:
    """
     Go through the words in the dataset; record the number of unique words against the total number of words for total
     numbers of words that are powers of 2 (starting at 2^0, until the end of the data-set)

     @param dataset_path: a path to a folder with a list of  reviews
     @returns: the number of types for every n tokens with n being 2^k in the form [(#tokens, #types), ...]
    """
    mem = set()
    total = 0
    count = 1
    datapoint = []
    for file in os.listdir(dataset_path):
        file_path = f"{dataset_path}\{file}"
        with open(file_path, encoding="utf-8") as f:
            txt = f.readlines()
        for line in txt:
            l = line.strip().split(" ")
            for i in l:
                if i not in mem:
                    total += 1
                    mem.add(i)
                    if total == count:
                        count *= 2
                        datapoint.append((total, len(mem)))
                else:
                    total += 1
                    if total == count:
                        count *= 2
                        datapoint.append((total, len(mem)))

    if total != (count/2):
        datapoint.append((total, len(mem)))
    return datapoint


def draw_heap(type_counts: List[Tuple[int, int]]) -> None:
    """
    Use the provided chart plotting program to plot the logs of the number of unique words against the logs of the
    number of total words.

    @param type_counts: the number of types for every n tokens with n being 2^k in the form [(#tokens, #types), ...]
    """
    data = []
    for i in type_counts:
        data.append((log(i[0]),log(i[1])))
    chart_plot(data, "log unique vs log word", "log word", "log unique")



def main():
    """
    Code to check your work locally (run this from the root directory, 'mlrd/')
    """
    frequencies = count_token_frequencies(os.path.join('data', 'sentiment_detection', 'reviews_large', 'reviews'))

    draw_frequency_ranks(frequencies)
    draw_selected_words_ranks(frequencies)

    clean_plot()
    draw_zipf(frequencies)

    clean_plot()
    tokens = compute_type_count(os.path.join('data', 'sentiment_detection', 'reviews_large', 'reviews'))
    draw_heap(tokens)


if __name__ == '__main__':
    main()