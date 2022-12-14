import colour
r = 0
g = 0
b = 0

for r in range (1,10):
    for g in range (1,10):
        for b in range(1,10):
            rgb = [r/10,g/10,b/10]

            XYZ = colour.sRGB_to_XYZ(rgb)
            xyY = colour.XYZ_to_xyY(XYZ)
            munsell = ""
            try:
                munsell = colour.xyY_to_munsell_colour(xyY)
            except:
                print(rgb, "is not possible")
                continue
            print(munsell)