import math

''' Compute the MSE loss

Input:
    targets: list of target values (float)
    preds: list of predicted values (float)

Output:
    mean squared error (float)

Precondition:
    len(targets) == len(preds)
'''
def mse(targets, preds):
    N = len(targets)
    assert len(preds) == N
    total = 0
    for i in range(N):
        total += (targets[i]-preds[i])**2

    return total / N


''' Compute the RMSE loss

Input:
    targets: list of target values (float)
    preds: list of predicted values (float)

Output:
    root mean squared error (float)

Precondition:
    len(targets) == len(preds)

'''
def rmse(targets, preds):
    return math.sqrt(mse(targets, preds))


''' Compute the MAE loss

Input:
    targets: list of target values (float)
    preds: list of predicted values (float)

Output:
    mean absolute error (float)

Precondition:
    len(targets) == len(preds)
'''
def mae(targets, preds):
    N = len(targets)
    assert len(preds) == N
    total = 0
    for i in range(N):
        total += abs(targets[i]-preds[i])

    return total / N

