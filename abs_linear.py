import ds_manager
import linear


def run_plz():
    dm = ds_manager.DSManager()
    train_ds = dm.get_train_ds()
    test_ds = dm.get_test_ds()

    train_x = train_ds.get_x()
    train_y = train_ds.get_y()
    test_x = test_ds.get_x()
    test_y = test_ds.get_y()

    linear.get_r2(train_x, train_y, test_x, test_y)

    print("done")


if __name__ == "__main__":
    run_plz()