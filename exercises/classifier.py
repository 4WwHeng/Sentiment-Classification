from typing import List, Dict
import os
import nltk
from utils.sentiment_detection import read_tokens, load_reviews, split_data


def read_lexicon(filename: str) -> Dict[str, int]:
    """
    Read the lexicon from a given path.

    @param filename: path to file
    @return: dictionary from word to sentiment (+1 or -1 for positive or negative sentiments respectively).
    """
    with open(filename, 'r') as f:
        txt = f.readlines()
    ret = {}
    for line in txt:
        l = line.strip().split(" ")
        if l[2] == "polarity=negative":
            ret[l[0][l[0].index("=")+1:]] = -1
        else:
            ret[l[0][l[0].index("=")+1:]] = 1
    return ret

def predict_sentiment(review: List[str], lexicon: Dict[str, int]) -> int:
    """
    Given a list of tokens from a tokenized review and a lexicon, determine whether the sentiment of each review in the
    test set is positive or negative based on whether there are more positive or negative words.

    @param review: list of tokens from tokenized review
    @param lexicon: dictionary from word to sentiment (+1 or -1 for positive or negative sentiments respectively).
    @return: calculated sentiment for each review (+1 or -1 for positive or negative sentiments respectively).
    """
    score = 0
    for i in review:
        if i in lexicon:
            score += lexicon[i]
    if score >= 0:
        return 1
    return -1


def accuracy(pred: List[int], true: List[int]) -> float:
    """
    Calculate the proportion of predicted sentiments that were correct.

    @param pred: list of calculated sentiment for each review
    @param true: list of correct sentiment for each review
    @return: the overall accuracy of the predictions
    """
    n = len(pred)
    correct = 0
    for i in range(n):
        if pred[i] == true[i]:
            correct += 1
    return correct/n

def predict_sentiment_improved(review: List[str], lexicon: Dict[str, int]) -> int:
    """
    Use the training data to improve your classifier, perhaps by choosing an offset for the classifier cutoff which
    works better than 0.

    @param review: list of tokens from tokenized review
    @param lexicon: dictionary from word to sentiment (+1 or -1 for positive or negative sentiments respectively).
    @return: calculated sentiment for each review (+1, -1 for positive and negative sentiments, respectively).
    """
    
    score = 0
    for i in review:
        if i in lexicon:
            score += lexicon[i]
    if score >= 8:
        return 1
    return -1

def main():
    """
    Code to check your work locally (run this from the root directory, 'mlrd/')
    """
    review_data = load_reviews(os.path.join('data','sentiment_detection', 'reviews'))

    tokenized_data = [read_tokens(x['filename']) for x in review_data]

    lexicon = read_lexicon(os.path.join('data', 'sentiment_detection', 'sentiment_lexicon'))

    pred1 = [predict_sentiment(t, lexicon) for t in tokenized_data]
    acc1 = accuracy(pred1, [x['sentiment'] for x in review_data])
    print(f"Your accuracy: {acc1}")

    pred2 = [predict_sentiment_improved(t, lexicon) for t in tokenized_data]
    acc2 = accuracy(pred2, [x['sentiment'] for x in review_data])
    print(f"Your improved accuracy: {acc2}")


if __name__ == '__main__':
    main()
