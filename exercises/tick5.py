from typing import List, Dict, Union
import os
from utils.sentiment_detection import read_tokens, load_reviews, print_binary_confusion_matrix
from exercises.tick1 import accuracy
from exercises.tick2 import predict_sentiment_nbc, calculate_smoothed_log_probabilities, \
    calculate_class_log_probabilities
from exercises.tick1 import read_lexicon, predict_sentiment


def generate_random_cross_folds(training_data: List[Dict[str, Union[List[str], int]]], n: int = 10) \
        -> List[List[Dict[str, Union[List[str], int]]]]:
    """
    Split training data into n folds, random.

    @param training_data: list of training instances, where each instance is a dictionary with two fields: 'text' and
        'sentiment'. 'text' is the tokenized review and 'sentiment' is +1 or -1, for positive and negative sentiments.
    @param n: the number of cross-folds
    @return: a list of n folds, where each fold is a list of training instances
    """
    ret = []
    l = len(training_data)
    s = l//n
    for i in range(n):
        ret.append(training_data[s*i:s*(i+1)])
    return ret


def generate_stratified_cross_folds(training_data: List[Dict[str, Union[List[str], int]]], n: int = 10) \
        -> List[List[Dict[str, Union[List[str], int]]]]:
    """
    Split training data into n folds, stratified.

    @param training_data: list of training instances, where each instance is a dictionary with two fields: 'text' and
        'sentiment'. 'text' is the tokenized review and 'sentiment' is +1 or -1, for positive and negative sentiments.
    @param n: the number of cross-folds
    @return: a list of n folds, where each fold is a list of training instances
    """
    pos = []
    neg = []
    l = len(training_data)
    s = l//n
    for data in training_data:
        if data["sentiment"] == 1:
            if not pos:
                pos.append([data])
            elif len(pos[-1]) == s//2:
                pos.append([data])
            else:
                pos[-1].append(data)
        else:
            if not neg:
                neg.append([data])
            elif len(neg[-1]) == s//2:
                neg.append([data])
            else:
                neg[-1].append(data)
    for i in range(n):
        pos[i].extend(neg[i])
    return pos


def cross_validate_nbc(split_training_data: List[List[Dict[str, Union[List[str], int]]]]) -> List[float]:
    """
    Perform an n-fold cross validation, and return the mean accuracy and variance.

    @param split_training_data: a list of n folds, where each fold is a list of training instances, where each instance
        is a dictionary with two fields: 'text' and 'sentiment'. 'text' is the tokenized review and 'sentiment' is +1 or
        -1, for positive and negative sentiments.
    @return: list of accuracy scores for each fold
    """
    l = len(split_training_data)
    ret = []
    for i in range(l):
        test = split_training_data[i][:]
        test_text = [x['text'] for x in test]
        test_sentiments = [x['sentiment'] for x in test]
        train = []
        for j in range(l):
            if j != i:
                if not train:
                    train = split_training_data[j][:]
                else:
                    train.extend(split_training_data[j])
        class_priors = calculate_class_log_probabilities(train)
        smoothed_log_probabilities = calculate_smoothed_log_probabilities(train)
        preds_smoothed = []
        for review in test_text:
            pred = predict_sentiment_nbc(review, smoothed_log_probabilities, class_priors)
            preds_smoothed.append(pred)

        acc_smoothed = accuracy(preds_smoothed, test_sentiments)
        ret.append(acc_smoothed)
    return ret


def cross_validation_accuracy(accuracies: List[float]) -> float:
    """Calculate the mean accuracy across n cross fold accuracies.

    @param accuracies: list of accuracy scores for n cross folds
    @returns: mean accuracy over the cross folds
    """
    n = len(accuracies)
    summ = sum(accuracies)
    return summ/n

def cross_validation_variance(accuracies: List[float]) -> float:
    """Calculate the variance of n cross fold accuracies.

    @param accuracies: list of accuracy scores for n cross folds
    @returns: variance of the cross fold accuracies
    """
    n = len(accuracies)
    mean = sum(accuracies)
    mean /= n
    var = 0
    for i in accuracies:
        var += (i-mean)**2
    return var/n


def confusion_matrix(predicted_sentiments: List[int], actual_sentiments: List[int]) -> List[List[int]]:
    """
    Calculate the number of times (1) the prediction was POS and it was POS [correct], (2) the prediction was POS but
    it was NEG [incorrect], (3) the prediction was NEG and it was POS [incorrect], and (4) the prediction was NEG and it
    was NEG [correct]. Store these values in a list of lists, [[(1), (2)], [(3), (4)]], so they form a confusion matrix:
                     actual:
                     pos     neg
    predicted:  pos  [[(1),  (2)],
                neg   [(3),  (4)]]

    @param actual_sentiments: a list of the true (gold standard) sentiments
    @param predicted_sentiments: a list of the sentiments predicted by a system
    @returns: a confusion matrix
    """
    n = len(predicted_sentiments)
    a = 0
    b = 0
    c = 0
    d = 0
    for i in range(n):
        pre = predicted_sentiments[i]
        act = actual_sentiments[i]
        if pre == act:
            if pre == 1:
                a += 1
            else:
                d += 1
        else:
            if pre == 1:
                b += 1
            else:
                c += 1
    return [[a,b],[c,d]]


def main():
    """
    Code to check your work locally (run this from the root directory, 'mlrd/')
    """
    review_data = load_reviews(os.path.join('data', 'sentiment_detection', 'reviews'))
    tokenized_data = [{'text': read_tokens(fn['filename']), 'sentiment': fn['sentiment']} for fn in review_data]

    # First test cross-fold validation
    folds = generate_random_cross_folds(tokenized_data, n=10)
    accuracies = cross_validate_nbc(folds)
    print(f"Random cross validation accuracies: {accuracies}")
    mean_accuracy = cross_validation_accuracy(accuracies)
    print(f"Random cross validation mean accuracy: {mean_accuracy}")
    variance = cross_validation_variance(accuracies)
    print(f"Random cross validation variance: {variance}\n")

    folds = generate_stratified_cross_folds(tokenized_data, n=10)
    accuracies = cross_validate_nbc(folds)
    print(f"Stratified cross validation accuracies: {accuracies}")
    mean_accuracy = cross_validation_accuracy(accuracies)
    print(f"Stratified cross validation mean accuracy: {mean_accuracy}")
    variance = cross_validation_variance(accuracies)
    print(f"Stratified cross validation variance: {variance}\n")

    # Now evaluate on 2016 and test
    class_priors = calculate_class_log_probabilities(tokenized_data)
    smoothed_log_probabilities = calculate_smoothed_log_probabilities(tokenized_data)

    preds_test = []
    test_data = load_reviews(os.path.join('data', 'sentiment_detection', 'reviews_test'))
    test_tokens = [read_tokens(x['filename']) for x in test_data]
    test_sentiments = [x['sentiment'] for x in test_data]
    for review in test_tokens:
        pred = predict_sentiment_nbc(review, smoothed_log_probabilities, class_priors)
        preds_test.append(pred)

    acc_smoothed = accuracy(preds_test, test_sentiments)
    print(f"Smoothed Naive Bayes accuracy on held-out data: {acc_smoothed}")
    print("Confusion matrix:")
    print_binary_confusion_matrix(confusion_matrix(preds_test, test_sentiments))

    lexicon = read_lexicon(os.path.join('data', 'sentiment_detection', 'sentiment_lexicon'))

    pred1 = [predict_sentiment(t, lexicon) for t in test_tokens]
    acc1 = accuracy(pred1, [x['sentiment'] for x in test_data])
    print(f"Simple classifier accuracy on held-out data: {acc1}")
    print("\n")

    preds_recent = []
    recent_review_data = load_reviews(os.path.join('data', 'sentiment_detection', 'reviews_2016'))
    recent_tokens = [read_tokens(x['filename']) for x in recent_review_data]
    recent_sentiments = [x['sentiment'] for x in recent_review_data]
    for review in recent_tokens:
        pred = predict_sentiment_nbc(review, smoothed_log_probabilities, class_priors)
        preds_recent.append(pred)

    acc_smoothed = accuracy(preds_recent, recent_sentiments)
    print(f"Smoothed Naive Bayes accuracy on 2016 data: {acc_smoothed}")
    print("Confusion matrix:")
    print_binary_confusion_matrix(confusion_matrix(preds_recent, recent_sentiments))

    pred1 = [predict_sentiment(t, lexicon) for t in recent_tokens]
    acc1 = accuracy(pred1, [x['sentiment'] for x in recent_review_data])
    print(f"Simple classifier accuracy on 2016 data: {acc1}")


if __name__ == '__main__':
    main()
