
import zstandard  # pip install zstandard
from io import BytesIO


def decode_zst(data):
    filelike = BytesIO(data)
    dctx = zstandard.ZstdDecompressor()
    reader = dctx.stream_reader(filelike)
    ret = reader.read()
    filelike.close()
    return ret


def encode_zst(data):
    filelike = BytesIO()
    ctx = zstandard.ZstdCompressor()
    writer = ctx.stream_writer(filelike)
    if isinstance(data, str):
        data = data.encode('utf-8')
    writer.write(data)
    writer.flush(zstandard.FLUSH_FRAME)
    filelike.flush()
    filelike.seek(0)
    ret = filelike.read()
    filelike.close()
    return ret


def read_zst(openfile):
    dctx = zstandard.ZstdDecompressor()
    reader = dctx.stream_reader(openfile)
    data = reader.read()
    return data


def write_zst(openfile, data):
    ctx = zstandard.ZstdCompressor()
    writer = ctx.stream_writer(openfile)
    if isinstance(data, str):
        data = data.encode('utf-8')
    writer.write(data)
    writer.flush(zstandard.FLUSH_FRAME)
    openfile.flush()


def load_zst(filename):
    with open(filename, 'rb') as openfile:
        dctx = zstandard.ZstdDecompressor()
        reader = dctx.stream_reader(openfile)
        data = reader.read()
    return data


def save_zst(filename, data):
    with open(filename, 'wb') as openfile:
        ctx = zstandard.ZstdCompressor()
        writer = ctx.stream_writer(openfile)
        if isinstance(data, str):
            data = data.encode('utf-8')
        writer.write(data)
        writer.flush(zstandard.FLUSH_FRAME)
        openfile.flush()
