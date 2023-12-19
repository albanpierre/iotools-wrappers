
import zipfile
from io import BytesIO


def decode_zip(data):
    filelike = BytesIO(data)
    z = zipfile.ZipFile(filelike)
    data = {fn: None for fn in z.namelist() if not fn.endswith("/")}
    for subfilename in data:
        with z.open(subfilename, "r") as subf:
            data[subfilename] = subf.read()
    z.close()
    filelike.close()
    return data


def encode_zip(data):
    filelike = BytesIO()
    z = zipfile.ZipFile(filelike, "w")
    for subfilename, raw_data in data.items():
        with z.open(subfilename, "w") as subf:
            if isinstance(raw_data, str):
                raw_data = raw_data.encode('utf-8')
            subf.write(raw_data)
    z.close()
    filelike.seek(0)
    ret = filelike.read()
    filelike.close()
    return ret


def read_zip(openfile):
    z = zipfile.ZipFile(openfile)
    data = {fn: None for fn in z.namelist() if not fn.endswith("/")}
    for subfilename in data:
        with z.open(subfilename, "r") as subf:
            data[subfilename] = subf.read()
    z.close()
    return data


def write_zip(openfile, data):
    z = zipfile.ZipFile(openfile, "w")
    for subfilename, raw_data in data.items():
        with z.open(subfilename, "w") as subf:
            if isinstance(raw_data, str):
                raw_data = raw_data.encode('utf-8')
            subf.write(raw_data)
    z.close()


def load_zip(filename, fs=None):
    with open(filename, 'rb') as openfile:
        z = zipfile.ZipFile(openfile)
        data = {fn: None for fn in z.namelist() if not fn.endswith("/")}
        for subfilename in data:
            with z.open(subfilename, "r") as subf:
                data[subfilename] = subf.read()
        z.close()
    return data


def save_zip(filename, data):
    with open(filename, 'wb') as openfile:
        z = zipfile.ZipFile(openfile, "w")
        for subfilename, raw_data in data.items():
            with z.open(subfilename, "w") as subf:
                if isinstance(raw_data, str):
                    raw_data = raw_data.encode('utf-8')
                subf.write(raw_data)
        z.close()
