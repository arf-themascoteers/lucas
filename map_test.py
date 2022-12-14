from ds_manager import DSManager


# a_map = {"btype" : "absorbance", "ctype" : "hsv"}
#
# dm = DSManager(**a_map)

# params = [
#     {"btype": "absorbance", "ctype": "rgb", "name": "Absorbance"},
#     {"btype": "reflectance", "ctype": "rgb", "name": "Reflectance"},
#     {"ctype": "hsv", "name": "HSV"},
#     {"ctype": "rgbhsv", "name": "RGB + HSV"},
#     {"si": ["soci"], "name": "SOCI"},
#     {"si": ["ibs"], "name": "IBS"},
#     {"si": ["soci", "ibs"], "name": "SOCI + IBS"},
# ]
#
# print(params[2]["ctype"])

def fun(param):
    print(param)
    print(param == False)

my_dict = {"param" : False}

fun(**my_dict)