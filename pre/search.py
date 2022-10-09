import pandas as pd
import os

spectra_src_dir = "data/spectra"
topsoil_file = "data/topsoil/topsoil.csv"
out_file = "data/out/out.csv"

x = os.path.join(spectra_src_dir,"spectra_CY.csv")
print(x)
a_df = pd.read_csv("data/spectra/spectra_ CY .csv")


x = (a_df.loc[a_df['PointID'] == 64641611140])
print(len(x))