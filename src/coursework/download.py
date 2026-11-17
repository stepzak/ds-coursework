from pathlib import Path
import hashlib
import tarfile
import urllib.request
import zipfile
from .common import ROOT, dump

SOURCES = {
    'sentiment.zip': 'https://archive.ics.uci.edu/static/public/331/sentiment+labelled+sentences.zip',
    'annotations.tar.gz': 'https://thor.robots.ox.ac.uk/pets/annotations.tar.gz',
    'images.tar.gz': 'https://thor.robots.ox.ac.uk/pets/images.tar.gz',
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def prepare():
    raw = ROOT / 'data/raw'
    raw.mkdir(parents=True, exist_ok=True)
    manifest = []
    for name, url in SOURCES.items():
        path = raw / name
        if not path.exists():
            temporary = path.with_suffix(path.suffix + '.part')
            print(f'Downloading {url}', flush=True)
            urllib.request.urlretrieve(url, temporary)
            temporary.replace(path)
        print(f'Validating/extracting {name}', flush=True)
        if name.endswith('.zip'):
            with zipfile.ZipFile(path) as archive:
                if archive.testzip() is not None:
                    raise ValueError('Corrupt zip')
                for member in archive.infolist():
                    target = raw / 'sentiment' / member.filename
                    if not target.resolve().is_relative_to((raw / 'sentiment').resolve()):
                        raise ValueError('Unsafe zip path')
                    archive.extract(member, raw / 'sentiment')
        else:
            with tarfile.open(path) as archive:
                archive.extractall(raw, filter='data')
        manifest.append({'file': name, 'url': url, 'bytes': path.stat().st_size,
                         'sha256': sha256(path)})
    dump(manifest, ROOT / 'data/sources.json')


if __name__ == '__main__':
    prepare()
