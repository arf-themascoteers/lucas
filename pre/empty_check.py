import pandas as pd

spectra_src_dir = "data/spectra"
topsoil_file = "data/topsoil/topsoil.csv"
out_file = "data/out/out.csv"

topsoil_df = pd.read_csv(topsoil_file)
out = open(out_file, "w")
out.write("phc,phh,ec,oc,caco3,p,n,k,elevation,stones,lc1,lu1")
empties = []
for i in range(len(topsoil_df)):
    row = topsoil_df.iloc[i]
    x = row[6:].isna().sum()
    if x > 0:
        empties.append(row[0])

print(len(empties))
out.close()

