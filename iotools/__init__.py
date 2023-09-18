
import importlib as _importlib


__all__ = [
    'decode_avi',
    'encode_avi',
    'read_avi',
    'write_avi',
    'load_avi',
    'save_avi',
    'help_avi',
    'decode_bytes',
    'encode_bytes',
    'read_bytes',
    'write_bytes',
    'load_bytes',
    'save_bytes',
    'help_bytes',
    'decode_csv',
    'encode_csv',
    'read_csv',
    'write_csv',
    'load_csv',
    'save_csv',
    'help_csv',
    'decode_gif',
    'encode_gif',
    'read_gif',
    'write_gif',
    'load_gif',
    'save_gif',
    'help_gif',
    'decode_gz',
    'encode_gz',
    'read_gz',
    'write_gz',
    'load_gz',
    'save_gz',
    'help_gz',
    'decode_image',
    'encode_image',
    'read_image',
    'write_image',
    'load_image',
    'save_image',
    'help_image',
    'decode_jpg',
    'encode_jpg',
    'read_jpg',
    'write_jpg',
    'load_jpg',
    'save_jpg',
    'help_jpg',
    'decode_json',
    'encode_json',
    'read_json',
    'write_json',
    'load_json',
    'save_json',
    'help_json',
    'decode_mp4',
    'encode_mp4',
    'read_mp4',
    'write_mp4',
    'load_mp4',
    'save_mp4',
    'help_mp4',
    'decode_npy',
    'encode_npy',
    'read_npy',
    'write_npy',
    'load_npy',
    'save_npy',
    'help_npy',
    'decode_pickle',
    'encode_pickle',
    'read_pickle',
    'write_pickle',
    'load_pickle',
    'save_pickle',
    'help_pickle',
    'decode_png',
    'encode_png',
    'read_png',
    'write_png',
    'load_png',
    'save_png',
    'help_png',
    'decode_tar',
    'encode_tar',
    'read_tar',
    'write_tar',
    'load_tar',
    'save_tar',
    'help_tar',
    'decode_txt',
    'encode_txt',
    'read_txt',
    'write_txt',
    'load_txt',
    'save_txt',
    'help_txt',
    'decode_video',
    'encode_video',
    'read_video',
    'write_video',
    'load_video',
    'save_video',
    'help_video',
    'decode_xml',
    'encode_xml',
    'read_xml',
    'write_xml',
    'load_xml',
    'save_xml',
    'help_xml',
    'decode_yaml',
    'encode_yaml',
    'read_yaml',
    'write_yaml',
    'load_yaml',
    'save_yaml',
    'help_yaml',
    'decode_zip',
    'encode_zip',
    'read_zip',
    'write_zip',
    'load_zip',
    'save_zip',
    'help_zip',
    'decode_zst',
    'encode_zst',
    'read_zst',
    'write_zst',
    'load_zst',
    'save_zst',
    'help_zst',
]


def __getattr__(attr):
    if attr in {
        'decode_avi',
        'encode_avi',
        'read_avi',
        'write_avi',
        'load_avi',
        'save_avi',
        'help_avi',
    }:
        aviio = _importlib.import_module('iotools.aviio')
        return getattr(aviio, attr)
    if attr in {
        'decode_bytes',
        'encode_bytes',
        'read_bytes',
        'write_bytes',
        'load_bytes',
        'save_bytes',
        'help_bytes',
    }:
        bytesio = _importlib.import_module('iotools.bytesio')
        return getattr(bytesio, attr)
    if attr in {
        'decode_csv',
        'encode_csv',
        'read_csv',
        'write_csv',
        'load_csv',
        'save_csv',
        'help_csv',
    }:
        csvio = _importlib.import_module('iotools.csvio')
        return getattr(csvio, attr)
    if attr in {
        'decode_gif',
        'encode_gif',
        'read_gif',
        'write_gif',
        'load_gif',
        'save_gif',
        'help_gif',
    }:
        gifio = _importlib.import_module('iotools.gifio')
        return getattr(gifio, attr)
    if attr in {
        'decode_gz',
        'encode_gz',
        'read_gz',
        'write_gz',
        'load_gz',
        'save_gz',
        'help_gz',
    }:
        gzio = _importlib.import_module('iotools.gzio')
        return getattr(gzio, attr)
    if attr in {
        'decode_image',
        'encode_image',
        'read_image',
        'write_image',
        'load_image',
        'save_image',
        'help_image',
    }:
        imageio = _importlib.import_module('iotools.imageio')
        return getattr(imageio, attr)
    if attr in {
        'decode_jpg',
        'encode_jpg',
        'read_jpg',
        'write_jpg',
        'load_jpg',
        'save_jpg',
        'help_jpg',
    }:
        jpgio = _importlib.import_module('iotools.jpgio')
        return getattr(jpgio, attr)
    if attr in {
        'decode_json',
        'encode_json',
        'read_json',
        'write_json',
        'load_json',
        'save_json',
        'help_json',
    }:
        jsonio = _importlib.import_module('iotools.jsonio')
        return getattr(jsonio, attr)
    if attr in {
        'decode_mp4',
        'encode_mp4',
        'read_mp4',
        'write_mp4',
        'load_mp4',
        'save_mp4',
        'help_mp4',
    }:
        mp4io = _importlib.import_module('iotools.mp4io')
        return getattr(mp4io, attr)
    if attr in {
        'decode_npy',
        'encode_npy',
        'read_npy',
        'write_npy',
        'load_npy',
        'save_npy',
        'help_npy',
    }:
        npyio = _importlib.import_module('iotools.npyio')
        return getattr(npyio, attr)
    if attr in {
        'decode_pickle',
        'encode_pickle',
        'read_pickle',
        'write_pickle',
        'load_pickle',
        'save_pickle',
        'help_pickle',
    }:
        pickleio = _importlib.import_module('iotools.pickleio')
        return getattr(pickleio, attr)
    if attr in {
        'decode_png',
        'encode_png',
        'read_png',
        'write_png',
        'load_png',
        'save_png',
        'help_png',
    }:
        pngio = _importlib.import_module('iotools.pngio')
        return getattr(pngio, attr)
    if attr in {
        'decode_tar',
        'encode_tar',
        'read_tar',
        'write_tar',
        'load_tar',
        'save_tar',
        'help_tar',
    }:
        tario = _importlib.import_module('iotools.tario')
        return getattr(tario, attr)
    if attr in {
        'decode_txt',
        'encode_txt',
        'read_txt',
        'write_txt',
        'load_txt',
        'save_txt',
        'help_txt',
    }:
        txtio = _importlib.import_module('iotools.txtio')
        return getattr(txtio, attr)
    if attr in {
        'decode_video',
        'encode_video',
        'read_video',
        'write_video',
        'load_video',
        'save_video',
        'help_video',
    }:
        videoio = _importlib.import_module('iotools.videoio')
        return getattr(videoio, attr)
    if attr in {
        'decode_xml',
        'encode_xml',
        'read_xml',
        'write_xml',
        'load_xml',
        'save_xml',
        'help_xml',
    }:
        xmlio = _importlib.import_module('iotools.xmlio')
        return getattr(xmlio, attr)
    if attr in {
        'decode_yaml',
        'encode_yaml',
        'read_yaml',
        'write_yaml',
        'load_yaml',
        'save_yaml',
        'help_yaml',
    }:
        yamlio = _importlib.import_module('iotools.yamlio')
        return getattr(yamlio, attr)
    if attr in {
        'decode_zip',
        'encode_zip',
        'read_zip',
        'write_zip',
        'load_zip',
        'save_zip',
        'help_zip',
    }:
        zipio = _importlib.import_module('iotools.zipio')
        return getattr(zipio, attr)
    if attr in {
        'decode_zst',
        'encode_zst',
        'read_zst',
        'write_zst',
        'load_zst',
        'save_zst',
        'help_zst',
    }:
        zstio = _importlib.import_module('iotools.zstio')
        return getattr(zstio, attr)
    raise AttributeError(attr)
