from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def get_r2(train_x, train_y, test_x, test_y):
    reg = LinearRegression().fit(train_x, train_y)
    return reg.score(test_x, test_y)
