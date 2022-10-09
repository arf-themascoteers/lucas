import os

import pandas as pd

spectra_src_dir = "data/spectra"
topsoil_file = "data/topsoil/topsoil.csv"
out_file = "data/out/out.csv"

topsoil_df = pd.read_csv(topsoil_file)
out = open(out_file, "w")
out.write("id,oc,phc,phh,ec,caco3,p,n,k,elevation,stones,lc1,lu1")
spec = 400
while spec <= 2499.5:
    out.write(f",{spec}")
    spec = spec+0.5
out.write("\n")
done = []
for file in os.listdir(spectra_src_dir):
    path = os.path.join(spectra_src_dir, file)
    a_df = pd.read_csv(path)
    for i in range(len(a_df)):
        spectra_row = a_df.iloc[i]
        empties = spectra_row.isna().sum()
        if empties != 0:
            print(file, spectra_row["PointID"])
            continue

        point_id = spectra_row['PointID']
        if point_id in done:
            continue
        rows = (topsoil_df.loc[topsoil_df['Point_ID'] == point_id])
        if len(rows) == 0:
            print(point_id)
            continue
        topsoil_row = rows.iloc[0]
        lc1_desc = topsoil_row['LC1_Desc'].replace(",","")
        lu1_desc = topsoil_row['LU1_Desc'].replace(",","")
        out.write(f"{topsoil_row['Point_ID']},{topsoil_row['OC']},{topsoil_row['pH(H2O)']},{topsoil_row['EC']},"
                  f"{topsoil_row['pH(CaCl2)']},{topsoil_row['CaCO3']},{topsoil_row['P']},{topsoil_row['N']},{topsoil_row['K']},"
                  f"{topsoil_row['Elevation']},"
                  f"{topsoil_row['Soil_Stones']},{lc1_desc},{lu1_desc}")

        spec = 400
        while spec <= 2499.5:
            val = spec
            if int(val) == val:
                val = int(val)
            val = str(val)
            out.write(f",{spectra_row[val]}")
            spec = spec + 0.5
        out.write("\n")
        done.append(point_id)
        if len(done)%1000 == 0:
            print(f"Done {len(done)}")


out.close()
print("done")

