import numpy as np
import pandas as pd
import ds_manager
import evaluate
import os


if __name__ == "__main__":
    columns = {"linear" : "Linear", "rf" : "RF", "nn" : "NN"}
    params = [
        {"btype" : "absorbance", "name" : "Absorbance"},
        {"btype" : "reflectance",  "name" : "Reflectance"},
        {"si": ["soci"], "si_only":True, "name": "SOCI"},
        {"si": ["ibs"], "si_only":True, "name": "IBS"},
        {"si": ["soci","ibs"], "si_only":True, "name": "SOCI + IBS"},

        {"btype" : "absorbance", "si": ["soci"], "name": "Absorbance + SOCI"},
        {"btype" : "reflectance", "si": ["soci"], "name": "Reflectance + SOCI"},

        {"btype": "absorbance", "si": ["ibs"], "name": "Absorbance + IBS"},
        {"btype": "reflectance", "si": ["ibs"], "name": "Reflectance + IBS"},

        {"btype": "absorbance", "si": ["soci","ibs"], "name": "Absorbance + SOCI + IBS"},
        {"btype": "reflectance", "si": ["soci","ibs"], "name": "Reflectance + SOCI + IBS"},

        {"ctype" : "rgb",  "name" : "RGB"},
        {"ctype" : "hsv", "name" : "HSV"},
        {"ctype" : "rgbhsv", "name" : "RGB + HSV"},

        {"ctype": "rgb", "si": ["soci"], "name": "RGB + SOCI"},
        {"ctype": "hsv", "si": ["soci"], "name": "HSV + SOCI"},
        {"ctype": "rgbhsv", "si": ["soci"], "name": "RGB + HSV + SOCI"},

        {"ctype": "rgb", "si": ["ibs"], "name": "RGB + IBS"},
        {"ctype": "hsv", "si": ["ibs"], "name": "HSV + IBS"},
        {"ctype": "rgbhsv", "si": ["ibs"], "name": "RGB + HSV + IBS"},

        {"ctype": "rgbhsv", "si": ["soci", "ibs"], "name": "RGB + HSV + SOCI + IBS"}
    ]
    data = np.zeros((len(params), len(columns)))
    column_values = [columns[key] for key in columns.keys()]
    names = [par["name"] for par in params]

    path = "results2.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        df.drop(columns=df.columns[0], axis=1, inplace=True)
        part_data = df.to_numpy()
        data[0:part_data.shape[0], 0:part_data.shape[1]] = part_data

    for index_par, par in enumerate(params):
        for index_col, col in enumerate(columns):
            print("Start",f"{col} - {par['name']}")
            if data[index_par][index_col] != 0:
                print("Was done already: ", data[index_par][index_col])
            else:
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