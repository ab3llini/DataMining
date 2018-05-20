from sklearn.metrics import r2_score


def evaluate(effective, predicted):
    return r2_score(effective, predicted);
