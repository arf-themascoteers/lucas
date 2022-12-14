import ds_manager
import evaluate
import os



def run_plz():
    os.chdir("../../")
    dm = ds_manager.DSManager(si=["soci"], ctype="rgb")
    return evaluate.r2_once(dm, "nn")


if __name__ == "__main__":
    r2s = run_plz()
    print(r2s)
    print(r2s)
