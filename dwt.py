import pywt

def transform(data):
    tups = pywt.wavedec(data, 'db1', level=5)
    return tups[0]
