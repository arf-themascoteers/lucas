import pywt

def transform(data):
    tups = pywt.wavedec(data, 'db1', level=2)
    return tups[0]
