import numpy as np
import pandas as pd
import ds_manager
import evaluate


if __name__ == "__main__":
    columns = {"linear" : "Linear", "rf" : "RF", "nn" : "NN"}
    params = [
        {"btype" : "absorbance", "name" : "Absorbance"},
        {"btype" : "reflectance",  "name" : "Reflectance"},
        {"ctype" : "rgb",  "name" : "RGB"},
        {"ctype" : "hsv", "name" : "HSV"},
        {"ctype" : "rgbhsv", "name" : "RGB + HSV"},
        {"si" : ["soci"], "name" : "SOCI"},
        {"si" : ["ibs"], "name" : "IBS"},
        {"si" : ["soci", "ibs"], "name" : "SOCI + IBS"},
    ]
    data = np.zeros((len(params), len(columns)))
    column_values = [columns[key] for key in columns.keys()]
    names = [par["name"] for par in params]
    for index_par, par in enumerate(params):
        for index_col, col in enumerate(columns):
            print("Start",f"{col} - {par['name']}")
            ds = ds_manager.DSManager(**par)
            r2s = evaluate.r2(ds, col)
            r2_mean = np.mean(r2s)
            print(par["name"], col, r2_mean)
            r2_log = open("r2_log.txt", "a")
            r2_log.write(f"{col} - {par['name']}: {str(r2s)}\n")
            r2_log.close()
            data[index_par][index_col] = r2_mean
            df = pd.DataFrame(data=data, columns=column_values, index=names)
            df.to_csv("results2.csv")
