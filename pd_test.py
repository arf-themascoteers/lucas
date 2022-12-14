
data = np.zeros((4,3))
columns = ["Linear", "RF", "NN"]
index_values = ["Absorbance", "Reflectance", "Absorbance + SOCI", "Reflectance + SOCI"]
df = pd.DataFrame(data=data, columns=columns, index=index_values)
df.to_csv("results2.csv")