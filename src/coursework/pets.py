"""Full image/mask audit, three-class conversion, hashes and protected test split."""
import hashlib
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image
from scipy.fft import dctn
from sklearn.model_selection import StratifiedGroupKFold
from .common import ROOT, PROCESSED, SEED, table, savefig, dump, describe

RAW = ROOT / 'data/raw'
COLORS = ['#263747', '#f59e42', '#3eb4aa']
CMAP = ListedColormap(COLORS)


def read_metadata(name):
    rows = []
    for line in (RAW / 'annotations' / name).read_text().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        image_id, class_id, species, breed_id = line.split()
        rows.append({'id': image_id, 'class_id': int(class_id), 'species': int(species),
                     'breed_id': int(breed_id), 'breed': image_id.rsplit('_', 1)[0]})
    return pd.DataFrame(rows)


def convert_mask(mask, species):
    if species not in [1, 2] or not set(np.unique(mask)).issubset({1, 2, 3}):
        raise ValueError('Unexpected species or trimap values')
    # README: 1=foreground, 2=background, 3=not classified.
    # Keep uncertainty: target 255 is ignored by loss and metrics.
    output = np.full(mask.shape, 255, dtype=np.uint8)
    output[mask == 2] = 0
    output[mask == 1] = species
    return output


def phash(rgb):
    gray = Image.fromarray(rgb).convert('L').resize((32, 32), Image.Resampling.LANCZOS)
    coeff = dctn(np.asarray(gray, dtype=float), norm='ortho')[:8, :8].ravel()
    bits = coeff > np.median(coeff[1:])
    bits[0] = False
    return int.from_bytes(np.packbits(bits).tobytes(), 'big')


def inspect_image(path):
    result = {'id': path.stem, 'file': path.name, 'bytes': path.stat().st_size}
    try:
        result['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        with Image.open(path) as image:
            result['format'] = image.format
            result['mode'] = image.mode
            image.load()  # force complete decode, no truncated-image override
            rgb = np.asarray(image.convert('RGB'))
            height, width = rgb.shape[:2]
        result.update(width=width, height=height, aspect=width / height, megapixels=width * height / 1e6,
                      pixel_sha256=hashlib.sha256(str(rgb.shape).encode() + rgb.tobytes()).hexdigest(),
                      phash=f'{phash(rgb):016x}', readable=True)
        gray = .299 * rgb[:, :, 0] + .587 * rgb[:, :, 1] + .114 * rgb[:, :, 2]
        result.update(luminance_mean=float(gray.mean()), luminance_std=float(gray.std()))
        mask_path = RAW / 'annotations/trimaps' / f'{path.stem}.png'
        if mask_path.exists():
            with Image.open(mask_path) as image:
                raw_mask = np.asarray(image)
            if raw_mask.shape == gray.shape:
                for label, title in [(1, 'foreground'), (2, 'background')]:
                    selected = gray[raw_mask == label]
                    result[f'{title}_luminance_mean'] = float(selected.mean()) if selected.size else np.nan
                    result[f'{title}_luminance_std'] = float(selected.std()) if selected.size else np.nan
    except Exception as exc:
        result.update(readable=False, error=f'{type(exc).__name__}: {exc}')
    return result


def inspect_mask(path):
    result = {'id': path.stem, 'mask_file': path.name}
    try:
        with Image.open(path) as image:
            image.load()
            mask = np.asarray(image)
            result.update(mask_format=image.format, mask_mode=image.mode)
        if mask.ndim != 2:
            raise ValueError(f'Not a single-channel mask: {mask.shape}')
        unique, counts = np.unique(mask, return_counts=True)
        result.update(mask_readable=True, mask_width=mask.shape[1], mask_height=mask.shape[0],
                      mask_values=','.join(map(str, unique)), invalid_values=int(np.isin(mask, [1, 2, 3], invert=True).sum()))
        for value, count in zip(unique, counts):
            result[f'pixels_{value}'] = int(count)
        for value in [1, 2, 3]:
            result.setdefault(f'pixels_{value}', 0)
        yy, xx = np.where(mask == 1)
        result['foreground_fraction'] = float((mask == 1).mean())
        result['uncertain_fraction'] = float((mask == 3).mean())
        result['empty_foreground'] = len(xx) == 0
        if len(xx):
            result.update(centroid_x=float((xx.mean() + .5) / mask.shape[1]),
                          centroid_y=float((yy.mean() + .5) / mask.shape[0]),
                          bbox_width=float((xx.max() - xx.min() + 1) / mask.shape[1]),
                          bbox_height=float((yy.max() - yy.min() + 1) / mask.shape[0]))
    except Exception as exc:
        result.update(mask_readable=False, mask_error=f'{type(exc).__name__}: {exc}')
    return result


class UnionFind:
    def __init__(self, ids):
        self.parent = {x: x for x in ids}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        x, y = self.find(x), self.find(y)
        self.parent[max(x, y)] = min(x, y)


def hash_pairs(images, threshold=6):
    rows = images[images.readable].sort_values('id')
    ids = rows.id.to_numpy()
    h = np.array([int(x, 16) for x in rows.phash], dtype=np.uint64)
    popcount = np.array([i.bit_count() for i in range(256)], dtype=np.uint8)
    pairs = []
    for i in range(len(h) - 1):
        xor = np.bitwise_xor(h[i], h[i+1:])
        distances = popcount[xor.view(np.uint8).reshape(-1, 8)].sum(axis=1)
        for offset in np.flatnonzero(distances <= threshold):
            j = i + 1 + offset
            pairs.append({'id_a': ids[i], 'id_b': ids[j], 'phash_distance': int(distances[offset])})
    return pd.DataFrame(pairs, columns=['id_a', 'id_b', 'phash_distance'])


def make_splits(df, pairs):
    valid = df[df.usable].copy()
    uf = UnionFind(valid.id)
    for column in ['sha256', 'pixel_sha256']:
        for _, group in valid.groupby(column):
            ids = list(group.id)
            for other in ids[1:]:
                uf.union(ids[0], other)
    for row in pairs.itertuples():
        if row.id_a in uf.parent and row.id_b in uf.parent:
            uf.union(row.id_a, row.id_b)
    valid['duplicate_group'] = valid.id.map(uf.find)
    test_groups = set(valid.loc[valid.official_split == 'test', 'duplicate_group'])
    valid['split'] = 'train'
    valid.loc[valid.official_split == 'test', 'split'] = 'test'
    overlap = (valid.official_split == 'trainval') & valid.duplicate_group.isin(test_groups)
    valid.loc[overlap, 'split'] = 'excluded_test_overlap'
    pool = valid[(valid.official_split == 'trainval') & ~overlap]
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    _, val_idx = next(splitter.split(pool, pool.class_id, groups=pool.duplicate_group))
    valid.loc[pool.iloc[val_idx].index, 'split'] = 'validation'
    df = df.merge(valid[['id', 'split', 'duplicate_group']], on='id', how='left', validate='one_to_one')
    df['split'] = df.split.fillna('excluded_invalid')
    df['image_path'] = 'data/raw/images/' + df.id + '.jpg'
    df['mask_path'] = 'data/processed/pets_masks/' + df.id + '.png'
    df.to_csv(PROCESSED / 'pets_splits.csv', index=False)
    table(pd.crosstab(df.split, df.species_name).reset_index(), 'pets_split_balance')
    # No metadata, byte, decoded pixel, or candidate duplicate group leakage.
    usable = df[df.split.isin(['train', 'validation', 'test'])]
    for key in ['id', 'sha256', 'pixel_sha256', 'duplicate_group']:
        sets = [set(usable.loc[usable.split == s, key]) for s in ['train', 'validation', 'test']]
        assert not any(sets[i] & sets[j] for i in range(3) for j in range(i)), key
    assert set(df[df.split == 'test'].id) == set(df[(df.official_split == 'test') & df.usable].id)
    return df


def gallery(df, name, ids=None):
    if ids is None:
        ids = list(df[df.species == 1].sort_values('id').iloc[[0, 100, 200]].id) + list(df[df.species == 2].sort_values('id').iloc[[0, 100, 200]].id)
    fig, axes = plt.subplots(len(ids), 3, figsize=(11, 3 * len(ids)), squeeze=False)
    for axrow, image_id in zip(axes, ids):
        row = df[df.id == image_id].iloc[0]
        with Image.open(RAW / 'images' / f'{image_id}.jpg') as image:
            rgb = np.asarray(image.convert('RGB'))
        with Image.open(RAW / 'annotations/trimaps' / f'{image_id}.png') as image:
            mask = np.asarray(image)
        target = convert_mask(mask, int(row.species))
        display = np.ma.masked_equal(target, 255)
        cmap = CMAP.copy(); cmap.set_bad('#eeeeee')
        axrow[0].imshow(rgb)
        axrow[0].set_title(f'{image_id} ({row.species_name})', fontsize=9)
        axrow[1].imshow(display, cmap=cmap, vmin=0, vmax=2, interpolation='nearest')
        axrow[1].set_title('0 background; 1 cat; 2 dog; grey ignore')
        axrow[2].imshow(rgb)
        axrow[2].imshow(np.ma.masked_where((target == 0) | (target == 255), target), cmap=CMAP, vmin=0, vmax=2, alpha=.5, interpolation='nearest')
        axrow[2].set_title(f'Foreground {row.foreground_fraction:.1%}')
        for ax in axrow:
            ax.axis('off')
    savefig(name)


def main():
    meta = read_metadata('list.txt')
    trainval, test = read_metadata('trainval.txt'), read_metadata('test.txt')
    assert meta.id.is_unique and trainval.id.is_unique and test.id.is_unique
    assert not set(trainval.id) & set(test.id)
    assert set(meta.id) == set(trainval.id) | set(test.id)
    assert set(meta.species) == {1, 2} and meta.class_id.nunique() == 37
    assert meta.groupby('class_id').species.nunique().max() == 1
    assert meta.id.str[0].str.isupper().eq(meta.species.eq(1)).all()
    official = pd.concat([trainval.assign(official_split='trainval'), test.assign(official_split='test')])
    assert meta.sort_values('id').reset_index(drop=True).equals(official.drop(columns='official_split').sort_values('id').reset_index(drop=True))
    meta = meta.merge(official[['id', 'official_split']], on='id', validate='one_to_one')
    meta['species_name'] = meta.species.map({1: 'cat', 2: 'dog'})
    paths = sorted((RAW / 'images').glob('*.jpg'))
    all_mask_paths = sorted((RAW / 'annotations/trimaps').glob('*.png'))
    # The archive contains macOS AppleDouble metadata alongside real PNGs.
    # These are auxiliary files, not corrupt segmentation masks.
    sidecars = [p for p in all_mask_paths if p.name.startswith('._') and p.read_bytes()[:4] == bytes.fromhex('00051607')]
    sidecar_set = set(sidecars)
    mask_paths = [p for p in all_mask_paths if p not in sidecar_set]
    table(pd.DataFrame([{'file': str(p.relative_to(RAW)), 'bytes': p.stat().st_size,
                         'kind': 'AppleDouble metadata', 'magic': '00051607'} for p in sidecars]),
          'pets_auxiliary_files')
    print(f'Auditing {len(paths)} images and {len(mask_paths)} masks', flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        images = pd.DataFrame(pool.map(inspect_image, paths))
        masks = pd.DataFrame(pool.map(inspect_mask, mask_paths))
    table(images, 'pets_image_audit')
    table(masks, 'pets_mask_audit')
    inventory = []
    for image_id in sorted(set(images.id) | set(masks.id) | set(meta.id)):
        inventory.append({'id': image_id, 'has_image': image_id in set(images.id),
                          'has_mask': image_id in set(masks.id), 'has_metadata': image_id in set(meta.id)})
    table(pd.DataFrame(inventory), 'pets_file_inventory')
    df = meta.merge(images, on='id', how='left', validate='one_to_one').merge(masks, on='id', how='left', validate='one_to_one')
    df['size_match'] = df.width.eq(df.mask_width) & df.height.eq(df.mask_height)
    df['usable'] = (df.readable.fillna(False) & df.mask_readable.fillna(False) & df.size_match &
                    df.invalid_values.eq(0))
    table(df[~df.usable], 'pets_invalid_annotations')
    table(df[df.empty_foreground.fillna(False)], 'pets_empty_foreground_masks')
    for key in ['sha256', 'pixel_sha256']:
        table(images[images.duplicated(key, keep=False)].sort_values(key), f'pets_exact_duplicates_{key}')
    pairs = hash_pairs(images)
    for key in ['official_split', 'species_name']:
        lookup = meta.set_index('id')[key]
        pairs[key + '_a'] = pairs.id_a.map(lookup)
        pairs[key + '_b'] = pairs.id_b.map(lookup)
    table(pairs, 'pets_perceptual_duplicate_candidates')
    df = make_splits(df, pairs)
    destination = PROCESSED / 'pets_masks'
    destination.mkdir(exist_ok=True)
    for row in df[df.usable].itertuples():
        with Image.open(RAW / 'annotations/trimaps' / f'{row.id}.png') as image:
            target = convert_mask(np.asarray(image), row.species)
        Image.fromarray(target).save(destination / f'{row.id}.png')
    table(pd.crosstab(df.species_name, df.official_split).reset_index(), 'pets_species_counts')
    table(df.groupby(['class_id', 'breed', 'species_name']).size().reset_index(name='images'), 'pets_breed_counts')
    good = df[df.usable]
    table(describe(good, ['width', 'height', 'aspect', 'megapixels', 'foreground_fraction', 'uncertain_fraction',
                          'centroid_x', 'centroid_y', 'luminance_mean', 'luminance_std',
                          'foreground_luminance_mean', 'background_luminance_mean', 'background_luminance_std']),
          'pets_descriptive_statistics', index=True)
    table(good.groupby('species_name')[['foreground_fraction', 'width', 'height', 'luminance_mean',
                                       'background_luminance_mean', 'background_luminance_std', 'centroid_x', 'centroid_y']].describe().stack(0, future_stack=True).reset_index(), 'pets_statistics_by_species')
    pixels = {'background': int(good.pixels_2.sum()),
              'cat': int(good.loc[good.species == 1, 'pixels_1'].sum()),
              'dog': int(good.loc[good.species == 2, 'pixels_1'].sum()),
              'ignore': int(good.pixels_3.sum())}
    table(pd.DataFrame([{'class': k, 'pixels': v, 'fraction_all': v / sum(pixels.values()),
                         'fraction_supervised': v / (sum(pixels.values()) - pixels['ignore']) if k != 'ignore' else np.nan}
                        for k, v in pixels.items()]), 'pets_pixel_counts')
    extremes = pd.concat([good.nsmallest(12, 'foreground_fraction').assign(reason='smallest_foreground'),
                          good.nlargest(12, 'foreground_fraction').assign(reason='largest_foreground'),
                          good[good.foreground_fraction.lt(.01) | good.foreground_fraction.gt(.9)].assign(reason='threshold_extreme')]).drop_duplicates('id')
    table(extremes, 'pets_extreme_objects')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    good.species_name.value_counts().reindex(['cat', 'dog']).plot.bar(ax=axes[0], color=COLORS[1:], rot=0)
    axes[0].set(title='Images by species', ylabel='Images')
    axes[1].bar(list(pixels), np.array(list(pixels.values())) / sum(pixels.values()), color=COLORS + ['#bbbbbb'])
    axes[1].set(title='Native-resolution pixel fractions', ylabel='Fraction, including uncertainty')
    savefig('pets_balance')
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes[0, 0].scatter(good.width, good.height, s=5, alpha=.3)
    axes[0, 0].set(xlabel='Width', ylabel='Height', title='Image dimensions')
    axes[0, 1].hist(good.aspect, bins=60, color='#476a9f'); axes[0, 1].set(xlabel='Width / height', title='Aspect ratios')
    for species, color in zip(['cat', 'dog'], COLORS[1:]):
        subset = good[good.species_name == species]
        axes[1, 0].hist(subset.foreground_fraction, bins=40, alpha=.55, label=species, color=color)
        axes[1, 1].hist(subset.megapixels, bins=40, alpha=.55, label=species, color=color)
    axes[1, 0].legend(); axes[1, 0].set(xlabel='Foreground / all pixels', title='Relative object area')
    axes[1, 1].legend(); axes[1, 1].set(xlabel='Megapixels', title='Image resolutions')
    savefig('pets_geometry')
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for species, color in zip(['cat', 'dog'], COLORS[1:]):
        part = good[good.species_name == species]
        axes[0].scatter(part.centroid_x, part.centroid_y, s=5, alpha=.3, label=species, color=color)
        axes[1].hist(part.luminance_mean, bins=30, alpha=.5, label=species, color=color)
        axes[2].hist(part.luminance_std, bins=30, alpha=.5, label=species, color=color)
    axes[0].set(xlabel='Relative x', ylabel='Relative y', title='Foreground centroid', xlim=(0, 1), ylim=(1, 0))
    axes[1].set(xlabel='Mean luminance (0..255)', title='Lighting proxy')
    axes[2].set(xlabel='Luminance standard deviation', title='Contrast proxy')
    for ax in axes: ax.legend()
    savefig('pets_position_lighting')
    gallery(good, 'pets_examples')
    nonempty = good[good.foreground_fraction.gt(0)]
    gallery(good, 'pets_area_extremes', list(nonempty.nsmallest(2, 'foreground_fraction').id) + list(good.nlargest(2, 'foreground_fraction').id))
    empty = good[good.empty_foreground]
    if len(empty):
        gallery(good, 'pets_empty_mask_examples', list(empty.head(3).id) + ['Egyptian_Mau_20'])
    if len(pairs):
        first = pairs[pairs.id_a.isin(good.id) & pairs.id_b.isin(good.id)].sort_values('phash_distance').head(4)
        ids = list(dict.fromkeys(first[['id_a', 'id_b']].to_numpy().ravel()))
        if ids: gallery(good, 'pets_duplicate_examples', ids)
    exact_rows = int(images.duplicated('pixel_sha256', keep=False).sum())
    cross = pairs[(pairs.official_split_a.notna()) & (pairs.official_split_b.notna()) & (pairs.official_split_a != pairs.official_split_b)]
    summary = {'metadata': len(meta), 'images_on_disk': len(images), 'masks_on_disk': len(masks),
               'appledouble_sidecars': len(sidecars),
               'unlisted_images': int((~images.id.isin(meta.id)).sum()), 'unlisted_masks': int((~masks.id.isin(meta.id)).sum()),
               'unreadable_images_all_files': int((~images.readable).sum()), 'unreadable_masks_all_files': int((~masks.mask_readable).sum()),
               'missing_images': int((~meta.id.isin(images.id)).sum()), 'missing_masks': int((~meta.id.isin(masks.id)).sum()),
               'invalid_metadata_rows': int((~df.usable).sum()), 'size_mismatches': int((~df.size_match).sum()),
               'invalid_mask_pixels': int(masks.invalid_values.fillna(0).sum()),
               'empty_masks': int(masks.empty_foreground.fillna(False).sum()),
               'empty_masks_official': int(df.empty_foreground.fillna(False).sum()),
               'species_counts': meta.species_name.value_counts().to_dict(), 'breeds': meta.breed.nunique(),
               'pixels': pixels, 'official_splits': meta.official_split.value_counts().to_dict(),
               'splits': df.split.value_counts().to_dict(), 'decoded_duplicate_rows': exact_rows,
               'phash_candidate_pairs': len(pairs), 'phash_cross_official_pairs': len(cross), 'phash_threshold': 6,
               'split_intersections': 0, 'seed': SEED,
               'object_area_below_1pct': int(good.foreground_fraction.lt(.01).sum()),
               'nonempty_object_area_below_1pct': int(nonempty.foreground_fraction.lt(.01).sum()),
               'object_area_above_90pct': int(good.foreground_fraction.gt(.9).sum()),
               'statistics': describe(good, ['width', 'height', 'aspect', 'megapixels', 'foreground_fraction', 'uncertain_fraction']).to_dict(orient='index')}
    dump(summary, ROOT / 'reports/pets_summary.json')
    print('Pets complete:', summary['metadata'], 'metadata rows;', summary['splits'], flush=True)
    return summary


if __name__ == '__main__':
    main()
