import os
from typing import List, Dict, Tuple
from exercises.classifier import accuracy, predict_sentiment, read_lexicon
from exercises.naive_bayes import calculate_class_log_probabilities, calculate_smoothed_log_probabilities, predict_sentiment_nbc
from utils.sentiment_detection import read_tokens, load_reviews, split_data
import math

def read_lexicon_magnitude(filename: str) -> Dict[str, Tuple[int, str]]:
    """
    Read the lexicon from a given path.

    @param filename: path to file
    @return: dictionary from word to a tuple of sentiment (1, -1) and magnitude ('strong', 'weak').
    """
    with open(filename, 'r') as f:
        txt = f.readlines()
    ret = {}
    for line in txt:
        l = line.strip().split(" ")
        if l[2] == "polarity=negative":
            ret[l[0][l[0].index("=")+1:]] = (-1, l[1][l[1].index("=")+1:])
        else:
            ret[l[0][l[0].index("=")+1:]] = (1, l[1][l[1].index("=")+1:])
    return ret

def predict_sentiment_magnitude(review: List[str], lexicon: Dict[str, Tuple[int, str]]) -> int:
    """
    Modify the simple classifier from Tick1 to include the information about the magnitude of a sentiment. Given a list
    of tokens from a tokenized review and a lexicon containing both sentiment and magnitude of a word, determine whether
    the sentiment of each review in the test set is positive or negative based on whether there are more positive or
    negative words. A word with a strong intensity should be weighted *four* times as high for the evaluator.

    @param review: list of tokens from tokenized review
    @param lexicon: dictionary from word to a tuple of sentiment (1, -1) and magnitude ('strong', 'weak').
    @return: calculated sentiment for each review (+1 or -1 for positive or negative sentiments respectively).
    """
    score = 0
    for i in review:
        if i in lexicon:
            if lexicon[i][1] == 'strong':
                score += 4*lexicon[i][0]
            else:
                score += lexicon[i][0]
    if score >= 0:
        return 1
    return -1


def sign_test(actual_sentiments: List[int], classification_a: List[int], classification_b: List[int]) -> float:
    """
    Implement the two-sided sign test algorithm to determine if one classifier is significantly better or worse than
    another. The sign for a result should be determined by which classifier is more correct and the ceiling of the least
    common sign total should be used to calculate the probability.

    @param actual_sentiments: list of correct sentiment for each review
    @param classification_a: list of sentiment prediction from classifier A
    @param classification_b: list of sentiment prediction from classifier B
    @return: p-value of the two-sided sign test.
    """
    plus = 0
    minus = 0
    null = 0
    n = len(actual_sentiments)
    for i in range(n):
        if classification_a[i] == classification_b[i]:
            null += 1
        elif classification_a[i] == actual_sentiments[i]:
            plus += 1
        else:
            minus += 1
    n = 2*math.ceil(null/2) + plus + minus
    k = math.ceil(null/2) + min(plus, minus)
    p = 0
    def comb(n, r):
        return int(math.factorial(n)/(math.factorial(n-r)*math.factorial(r)))
    for i in range(k+1):
        p += comb(n, i) * 0.5 ** n

    return 2*p


def main():
    """
    Code to check your work locally (run this from the root directory, 'mlrd/')
    """
    review_data = load_reviews(os.path.join('data', 'sentiment_detection', 'reviews'))
    training_data, validation_data = split_data(review_data, seed=0)

    train_tokenized_data = [{'text': read_tokens(fn['filename']), 'sentiment': fn['sentiment']} for fn in training_data]
    dev_tokenized_data = [read_tokens(fn['filename']) for fn in validation_data]
    validation_sentiments = [x['sentiment'] for x in validation_data]

    lexicon_magnitude = read_lexicon_magnitude(os.path.join('data', 'sentiment_detection', 'sentiment_lexicon'))
    lexicon = read_lexicon(os.path.join('data', 'sentiment_detection', 'sentiment_lexicon'))

    preds_magnitude = []
    preds_simple = []
    for review in dev_tokenized_data:
        pred = predict_sentiment_magnitude(review, lexicon_magnitude)
        preds_magnitude.append(pred)
        pred_simple = predict_sentiment(review, lexicon)
        preds_simple.append(pred_simple)

    acc_magnitude = accuracy(preds_magnitude, validation_sentiments)
    acc_simple = accuracy(preds_simple, validation_sentiments)

    print(f"Your accuracy using simple classifier: {acc_simple}")
    print(f"Your accuracy using magnitude classifier: {acc_magnitude}")

    class_priors = calculate_class_log_probabilities(train_tokenized_data)
    smoothed_log_probabilities = calculate_smoothed_log_probabilities(train_tokenized_data)

    preds_nb = []
    for review in dev_tokenized_data:
        pred = predict_sentiment_nbc(review, smoothed_log_probabilities, class_priors)
        preds_nb.append(pred)

    acc_nb = accuracy(preds_nb, validation_sentiments)
    print(f"Your accuracy using Naive Bayes classifier: {acc_nb}\n")

    p_value_magnitude_simple = sign_test(validation_sentiments, preds_simple, preds_magnitude)
    print(f"The p-value of the two-sided sign test for classifier_a \"{'classifier simple'}\" and classifier_b \"{'classifier magnitude'}\": {p_value_magnitude_simple}")

    p_value_magnitude_nb = sign_test(validation_sentiments, preds_nb, preds_magnitude)
    print(f"The p-value of the two-sided sign test for classifier_a \"{'classifier magnitude'}\" and classifier_b \"{'naive bayes classifier'}\": {p_value_magnitude_nb}")



if __name__ == '__main__':
    main()
