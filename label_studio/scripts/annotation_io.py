from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LS = ROOT / 'label_studio'
CLASS_IDS = {'Cat': 1, 'Dog': 2}


def image_filename(task):
    url = task['data']['image']
    local = parse_qs(urlparse(url).query).get('d', [urlparse(url).path])[0]
    return Path(unquote(local)).name


def object_id(task, kind):
    explicit = task['data'].get('object_id')
    if kind == 'pets':
        inferred = Path(image_filename(task)).stem
        if explicit and explicit != inferred:
            raise ValueError(f'Image/object_id mismatch: {explicit} / {inferred}')
        return explicit or inferred
    if not explicit:
        raise ValueError('Text object_id is required')
    return explicit


def index_tasks(path, kind):
    tasks = json.loads(Path(path).read_text(encoding='utf-8'))
    indexed = {}
    for task in tasks:
        key = object_id(task, kind)
        if key in indexed:
            raise ValueError(f'Duplicate object_id {key} in {path}')
        indexed[key] = task
    return indexed


def load_pair(kind, stage='fixed'):
    if kind not in {'pets', 'sentiment'} or stage not in {'initial', 'fixed'}:
        raise ValueError((kind, stage))
    suffix = '-fixed' if stage == 'fixed' else ''
    pair = [index_tasks(LS / f'annotations/{stage}/{kind}-{n}{suffix}.json', kind) for n in [1, 2]]
    tasks = index_tasks(LS / f'{kind}_tasks.json', kind)
    if set(pair[0]) != set(pair[1]) or set(pair[0]) != set(tasks):
        raise ValueError('The two exports and task list must contain the same object IDs')
    for key in tasks:
        for exported in pair:
            field = 'text' if kind == 'sentiment' else 'image'
            value = exported[key]['data'][field]
            expected = tasks[key]['data'][field]
            if value != expected:
                raise ValueError(f'{key}: exported data differs from the original task')
    return pair


def annotation(task):
    answers = task.get('annotations', [])
    if len(answers) != 1:
        raise ValueError('Expected exactly one answer per task in each annotator export')
    return answers[0]


def skipped(task):
    return bool(annotation(task).get('was_cancelled'))


def sentiment_label(task):
    if skipped(task):
        return None
    results = [r['value'].get('choices', []) for r in annotation(task)['result']
               if r['type'] == 'choices' and r['from_name'] == 'sentiment']
    if len(results) != 1 or len(results[0]) != 1 or results[0][0] not in {'Negative', 'Positive'}:
        raise ValueError('Accepted text requires one valid sentiment choice; an empty answer is not a skip')
    return {'Negative': 0, 'Positive': 1}[results[0][0]]


def decode_rle(rle, width, height):
    """Decode LS's packed bit stream to RGBA; check bounds before allocation."""
    if not rle or any(not isinstance(v, int) or not 0 <= v <= 255 for v in rle):
        raise ValueError('RLE must be a nonempty byte list')
    bits = ''.join(format(value, '08b') for value in rle)
    cursor = 0

    def take(count):
        nonlocal cursor
        if cursor + count > len(bits):
            raise ValueError('Truncated RLE stream')
        value = int(bits[cursor:cursor + count], 2)
        cursor += count
        return value

    length = take(32)
    word_bits = take(5) + 1
    run_bits = [take(4) + 1 for _ in range(4)]
    if length != width * height * 4 or word_bits != 8:
        raise ValueError(f'Unexpected RLE dimensions/word size: {length}, {word_bits}')
    decoded = np.empty(length, dtype=np.uint8)
    position = 0
    while position < length:
        repeated = take(1)
        count = take(run_bits[take(2)]) + 1
        end = position + count
        if end > length:
            raise ValueError('RLE run exceeds declared image dimensions')
        if repeated:
            decoded[position:end] = take(word_bits)
        else:
            decoded[position:end] = [take(word_bits) for _ in range(count)]
        position = end
    return decoded.reshape(height, width, 4)


def pet_mask(task, size):
    if skipped(task):
        return None
    width, height = size
    target = np.zeros((height, width), dtype=np.uint8)
    regions = [r for r in annotation(task)['result'] if r['type'] == 'brushlabels']
    if not regions:
        raise ValueError('Accepted image has no brush regions')
    for region in regions:
        if (region['original_width'], region['original_height']) != size or region.get('image_rotation', 0) != 0:
            raise ValueError('Image dimensions/orientation differ from the exported mask')
        labels = region['value'].get('brushlabels', [])
        if len(labels) != 1 or labels[0] not in CLASS_IDS or region['value'].get('format') != 'rle':
            raise ValueError('Expected one Cat/Dog brush label in RLE format')
        selected = decode_rle(region['value']['rle'], width, height)[:, :, 3] > 0
        class_id = CLASS_IDS[labels[0]]
        if np.any(selected & (target != 0) & (target != class_id)):
            raise ValueError('Cat/Dog brush regions overlap')
        target[selected] = class_id
    if not np.any(target):
        raise ValueError('Accepted image has an empty animal mask')
    return target


def write_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
