import ds_manager
from sklearn.ensemble import RandomForestRegressor
import os


def run_plz():
    os.chdir("../../")
    dm = ds_manager.DSManager(btype="reflectance", ctype="rgbhsv")
    train_ds = dm.get_train_ds()
    test_ds = dm.get_test_ds()

    train_x = train_ds.get_x()
    train_y = train_ds.get_y()
    test_x = test_ds.get_x()
    test_y = test_ds.get_y()

    reg = RandomForestRegressor(max_depth=5, n_estimators=1000).fit(train_x, train_y)
    r2 = reg.score(test_x, test_y)

    print(r2)


if __name__ == "__main__":
    run_plz()