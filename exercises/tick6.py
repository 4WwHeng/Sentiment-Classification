import os
from typing import List, Dict, Union

from utils.sentiment_detection import load_reviews, read_tokens, read_student_review_predictions, print_agreement_table

from exercises.tick5 import generate_random_cross_folds, cross_validation_accuracy
from  math import log

def nuanced_class_log_probabilities(training_data: List[Dict[str, Union[List[str], int]]]) \
        -> Dict[int, float]:
    """
    Calculate the prior class probability P(c) for nuanced sentiments.

    @param training_data: list of training instances, where each instance is a dictionary with two fields: 'text' and
        'sentiment'. 'text' is the tokenized review and 'sentiment' is +1, 0 or -1, for positive, neutral, and negative sentiments.
    @return: dictionary from sentiment to prior probability
    """
    total = 0
    pos = 0
    neu = 0
    neg = 0
    for dic in training_data:
        total += 1
        if dic['sentiment'] == 1:
            pos += 1
        elif dic['sentiment'] == 0:
            neu += 1
        else:
            neg += 1
    ret = {1: log(pos / total), -1: log(neg / total), 0: log(neu / total)}
    return ret


def nuanced_conditional_log_probabilities(training_data: List[Dict[str, Union[List[str], int]]]) \
        -> Dict[int, Dict[str, float]]:
    """
    Calculate the smoothed log likelihood log (P(x|c)) of a word in the vocabulary given a nuanced sentiment.

    @param training_data: list of training instances, where each instance is a dictionary with two fields: 'text' and
        'sentiment'. 'text' is the tokenized review and 'sentiment' is +1, 0 or -1, for positive, neutral, and negative sentiments.
    @return: dictionary from sentiment to Dictionary of tokens with respective log probability
    """
    totalPos = 0
    totalNeg = 0
    totalNeu = 0
    probPos = {}
    probNeg = {}
    probNeu = {}
    for dic in training_data:
        for txt in dic['text']:
            if dic['sentiment'] == 1:
                totalPos += 1
                if txt not in probPos:
                    probPos[txt] = 1
                probPos[txt] += 1
            elif dic['sentiment'] == 0:
                totalNeu += 1
                if txt not in probNeu:
                    probNeu[txt] = 1
                probNeu[txt] += 1
            else:
                totalNeg += 1
                if txt not in probNeg:
                    probNeg[txt] = 1
                probNeg[txt] += 1
    ret = {}
    tpos = {}
    tneg = {}
    tneu = {}
    totalNeg += len(probNeg)
    totalPos += len(probPos)
    totalNeu += len(probNeu)
    for txt in probPos:
        if txt not in probNeg:
            probNeg[txt] = 1
            probPos[txt] += 1
            totalNeg += 1
            totalPos += 1
        if txt not in probNeu:
            probNeu[txt] = 1
            probPos[txt] += 1
            totalNeu += 1
            totalPos += 1
    for txt in probNeg:
        if txt not in probPos:
            probPos[txt] = 1
            probNeg[txt] += 1
            totalPos += 1
            totalNeg += 1
        if txt not in probNeu:
            probNeu[txt] = 1
            probNeg[txt] += 1
            totalNeu += 1
            totalNeg += 1
    for txt in probNeu:
        if txt not in probPos:
            probPos[txt] = 1
            probNeu[txt] += 1
            totalPos += 1
            totalNeu += 1
        if txt not in probNeg:
            probNeg[txt] = 1
            probNeu[txt] += 1
            totalNeu += 1
            totalNeg += 1

    for txt in probPos:
        tpos[txt] = log(probPos[txt] / totalPos)
    ret[1] = tpos
    for txt in probNeg:
        tneg[txt] = log(probNeg[txt] / totalNeg)
    ret[-1] = tneg
    for txt in probNeu:
        tneu[txt] = log(probNeu[txt] / totalNeu)
    ret[0] = tneu
    return ret


def nuanced_accuracy(pred: List[int], true: List[int]) -> float:
    """
    Calculate the proportion of predicted sentiments that were correct.

    @param pred: list of calculated sentiment for each review
    @param true: list of correct sentiment for each review
    @return: the overall accuracy of the predictions
    """
    n = len(pred)
    ret = 0
    for i in range(n):
        if pred[i] == true[i]:
            ret += 1
    return ret/n


def predict_nuanced_sentiment_nbc(review: List[str], log_probabilities: Dict[int, Dict[str, float]],
                                  class_log_probabilities: Dict[int, float]) -> int:
    """
    Use the estimated log probabilities to predict the sentiment of a given review.

    @param review: a single review as a list of tokens
    @param log_probabilities: dictionary from sentiment to Dictionary of tokens with respective log probability
    @param class_log_probabilities: dictionary from sentiment to prior probability
    @return: predicted sentiment [-1, 0, 1] for the given review
    """
    probPos = class_log_probabilities[1]
    probNeg = class_log_probabilities[-1]
    probNeu = class_log_probabilities[0]
    for i in review:
        if i in log_probabilities[1]:
            probPos += log_probabilities[1][i]
        if i in log_probabilities[-1]:
            probNeg += log_probabilities[-1][i]
        if i in log_probabilities[0]:
            probNeu += log_probabilities[0][i]
    if probNeg > probPos:
        if probNeu > probNeg:
            return 0
        else:
            return -1
    else:
        if probNeu > probPos:
            return 0
        else:
            return 1


def calculate_kappa(agreement_table: Dict[int, Dict[int,int]]) -> float:
    """
    Using your agreement table, calculate the kappa value for how much agreement there was; 1 should mean total agreement and -1 should mean total disagreement.

    @param agreement_table:  For each review (1, 2, 3, 4) the number of predictions that predicted each sentiment
    @return: The kappa value, between -1 and 1
    """
    N = len(agreement_table)
    PeN = {-1:0,1:0}

    for id in agreement_table:
        if -1 in agreement_table[id]:
            PeN[-1] += agreement_table[id][-1]
        if 1 in agreement_table[id]:
            PeN[1] += agreement_table[id][1]
        k = agreement_table[id][1] + agreement_table[id][-1]

    Pe = (PeN[-1]/(N*k))**2 + (PeN[1]/(N*k))**2

    Pa = 0
    for id in agreement_table:
        temp = 0
        if -1 in agreement_table[id]:
            temp += agreement_table[id][-1]*(agreement_table[id][-1]-1)
        if 1 in agreement_table[id]:
            temp += agreement_table[id][1]*(agreement_table[id][1]-1)
        Pa += temp/(k*(k-1))
    Pa = Pa/N
    kappa = (Pa-Pe)/(1-Pe)
    return kappa


def get_agreement_table(review_predictions: List[Dict[int, int]]) -> Dict[int, Dict[int,int]]:
    """
    Builds an agreement table from the student predictions.

    @param review_predictions: a list of predictions for each student, the predictions are encoded as dictionaries, with the key being the review id and the value the predicted sentiment
    @return: an agreement table, which for each review contains the number of predictions that predicted each sentiment.
    """
    ret = {}
    for review in review_predictions:
        for id in review:
            if id not in ret:
                ret[id] = {1:0,-1:0}
            ret[id][review[id]] += 1
    print(ret)
    return ret


def main():
    """
    Code to check your work locally (run this from the root directory, 'mlrd/')
    """
    review_data = load_reviews(os.path.join('data', 'sentiment_detection', 'reviews_nuanced'), include_nuance=True)
    tokenized_data = [{'text': read_tokens(fn['filename']), 'sentiment': fn['sentiment']} for fn in review_data]

    split_training_data = generate_random_cross_folds(tokenized_data, n=10)

    n = len(split_training_data)
    accuracies = []
    for i in range(n):
        test = split_training_data[i]
        train_unflattened = split_training_data[:i] + split_training_data[i+1:]
        train = [item for sublist in train_unflattened for item in sublist]

        dev_tokens = [x['text'] for x in test]
        dev_sentiments = [x['sentiment'] for x in test]

        class_priors = nuanced_class_log_probabilities(train)
        nuanced_log_probabilities = nuanced_conditional_log_probabilities(train)
        preds_nuanced = []
        for review in dev_tokens:
            pred = predict_nuanced_sentiment_nbc(review, nuanced_log_probabilities, class_priors)
            preds_nuanced.append(pred)
        acc_nuanced = nuanced_accuracy(preds_nuanced, dev_sentiments)
        accuracies.append(acc_nuanced)

    mean_accuracy = cross_validation_accuracy(accuracies)
    print(f"Your accuracy on the nuanced dataset: {mean_accuracy}\n")

    review_predictions = read_student_review_predictions(os.path.join('data', 'sentiment_detection', 'class_predictions.csv'))

    print('Agreement table for this year.')

    agreement_table = get_agreement_table(review_predictions)
    print_agreement_table(agreement_table)

    fleiss_kappa = calculate_kappa(agreement_table)

    print(f"The cohen kappa score for the review predictions is {fleiss_kappa}.")

    fleiss_kappa = calculate_kappa({x:y for x,y in agreement_table.items() if x in [0, 1]})

    print(f"The cohen kappa score for the review predictions of review 1 and 2 is {fleiss_kappa}.")

    fleiss_kappa = calculate_kappa({x:y for x,y in agreement_table.items() if x in [2, 3]})

    print(f"The cohen kappa score for the review predictions of review 3 and 4 is {fleiss_kappa}.\n")

    review_predictions_four_years = read_student_review_predictions(os.path.join('data', 'sentiment_detection', 'class_predictions_2019_2023.csv'))
    agreement_table_four_years = get_agreement_table(review_predictions_four_years)

    print('Agreement table for the years 2019 to 2023.')
    print_agreement_table(agreement_table_four_years)

    fleiss_kappa = calculate_kappa(agreement_table_four_years)

    print(f"The cohen kappa score for the review predictions from 2019 to 2023 is {fleiss_kappa}.")



if __name__ == '__main__':
    main()
