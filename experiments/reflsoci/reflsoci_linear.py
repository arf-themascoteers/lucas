import ds_manager
import evaluate
import os
import numpy as np


def run_plz():
    os.chdir("../../")
    dm = ds_manager.DSManager(si=["soci"], btype="reflectance")
    return evaluate.r2(dm, "linear")


if __name__ == "__main__":
    r2s = run_plz()
    print(r2s)
    print(np.mean(r2s))
