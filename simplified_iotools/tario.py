
import tarfile
from io import BytesIO


def decode_tar(data):
    b = BytesIO(data)
    t = tarfile.open(fileobj=b)
    ret = {}
    for subfilename in t.getnames():
        subdata = t.extractfile(subfilename)
        if subdata is not None:
            ret[subfilename] = subdata.read()
    t.close()
    b.close()
    return ret


def encode_tar(data):
    b = BytesIO()
    t = tarfile.open(fileobj=b, mode="w")
    for subfilename, raw_data in data.items():
        ti = tarfile.TarInfo(subfilename)
        ti.size = len(raw_data)
        if isinstance(raw_data, str):
            raw_data = raw_data.encode('utf-8')
        t.addfile(ti, BytesIO(raw_data))
    b.seek(0)
    ret = b.read()
    t.close()
    b.close()
    return ret


def read_tar(openfile):
    b = BytesIO(openfile.read())
    t = tarfile.open(fileobj=b)
    data = {}
    for subfilename in t.getnames():
        subdata = t.extractfile(subfilename)
        if subdata is not None:
            data[subfilename] = subdata.read()
    t.close()
    b.close()
    return data


def write_tar(openfile, data):
    b = BytesIO()
    t = tarfile.open(fileobj=b, mode="w")
    for subfilename, raw_data in data.items():
        ti = tarfile.TarInfo(subfilename)
        ti.size = len(raw_data)
        if isinstance(raw_data, str):
            raw_data = raw_data.encode('utf-8')
        t.addfile(ti, BytesIO(raw_data))
    b.seek(0)
    openfile.write(b.read())
    t.close()


def load_tar(filename):
    t = tarfile.open(filename)
    data = {}
    for subfilename in t.getnames():
        subdata = t.extractfile(subfilename)
        if subdata is not None:
            data[subfilename] = subdata.read()
    t.close()
    return data


def save_tar(filename, data):
    t = tarfile.open(filename, mode="w")
    for subfilename, raw_data in data.items():
        ti = tarfile.TarInfo(subfilename)
        ti.size = len(raw_data)
        if isinstance(raw_data, str):
            raw_data = raw_data.encode('utf-8')
        t.addfile(ti, BytesIO(raw_data))
    t.close()
