import lucas_dataset
from sklearn.neural_network import MLPRegressor
import pickle
from sklearn.metrics import r2_score, mean_squared_error

def train():
    ds = lucas_dataset.LucasDataset(is_train=True)
    x = ds.get_x()
    y = ds.get_y()
    clf = MLPRegressor()
    clf.fit(x,y)
    print("Train done")
    pickle.dump(clf, open("models/mlp","wb"))


def test():
    reg = pickle.load(open('models/mlp', 'rb'))
    ds = lucas_dataset.LucasDataset(is_train=False)
    x = ds.get_x()
    y = ds.get_y()
    y_hat = reg.predict(x)
    print(r2_score(y, y_hat))
    print(mean_squared_error(y, y_hat))

    for i in range(3):
        a_y = ds.unscale(y[i])
        a_y_hat = ds.unscale(y_hat[i])
        print(f"{a_y:.3f}\t\t{a_y_hat:.3f}")

train()
test()