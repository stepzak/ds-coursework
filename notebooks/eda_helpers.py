import hashlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image
from scipy.fft import dctn

def inspect_image(path, mask_dir):
    result = {'id': path.stem, 'file': path.name, 'bytes': path.stat().st_size}
    try:
        result['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        with Image.open(path) as image:
            result['format'] = image.format
            result['mode'] = image.mode
            image.load()
            rgb = np.asarray(image.convert('RGB'))
            height, width = rgb.shape[:2]
        result.update(width=width, height=height, aspect=width / height, megapixels=width * height / 1e6,
                      pixel_sha256=hashlib.sha256(str(rgb.shape).encode() + rgb.tobytes()).hexdigest(),
                      readable=True)
        thumbnail = Image.fromarray(rgb).convert('L').resize((32, 32), Image.Resampling.LANCZOS)
        coefficients = dctn(np.asarray(thumbnail, dtype=float), norm='ortho')[:8, :8].ravel()
        bits = coefficients > np.median(coefficients[1:])
        bits[0] = False
        result['phash'] = int.from_bytes(np.packbits(bits).tobytes(), 'big')
        gray = .299 * rgb[:, :, 0] + .587 * rgb[:, :, 1] + .114 * rgb[:, :, 2]
        result.update(luminance_mean=float(gray.mean()), luminance_std=float(gray.std()))
        mask_path = mask_dir / f'{path.stem}.png'
        try:
            if mask_path.exists():
                with Image.open(mask_path) as image:
                    raw_mask = np.asarray(image)
                if raw_mask.shape == gray.shape:
                    for label, title in [(1, 'foreground'), (2, 'background')]:
                        selected = gray[raw_mask == label]
                        result[f'{title}_luminance_mean'] = float(selected.mean()) if selected.size else np.nan
                        result[f'{title}_luminance_std'] = float(selected.std()) if selected.size else np.nan
        except Exception as exc:
            result['background_statistics_error'] = str(exc)
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


def show_masks(ids, frame, image_dir, target_dir):
    """Показать исходник, уже преобразованную маску и её наложение."""
    colors = ListedColormap(['#263747', '#f59e42', '#3eb4aa'])
    colors.set_bad('#eeeeee')
    fig, axes = plt.subplots(len(ids), 3, figsize=(11, 2.8 * len(ids)), squeeze=False)
    for image_id, row_axes in zip(ids, axes):
        species = frame.loc[frame.id.eq(image_id), 'species_name'].iloc[0]
        with Image.open(image_dir / f'{image_id}.jpg') as image:
            rgb = np.asarray(image.convert('RGB'))
        with Image.open(target_dir / f'{image_id}.png') as image:
            target = np.asarray(image)
        row_axes[0].imshow(rgb)
        row_axes[0].set_title(f'{image_id} ({species})', fontsize=9)
        row_axes[1].imshow(np.ma.masked_equal(target, 255), cmap=colors, vmin=0, vmax=2, interpolation='nearest')
        row_axes[1].set_title('0 фон / 1 кошка / 2 собака / серый ignore', fontsize=9)
        row_axes[2].imshow(rgb)
        overlay = np.ma.masked_where((target == 0) | (target == 255), target)
        row_axes[2].imshow(overlay, cmap=colors, vmin=0, vmax=2, alpha=.5, interpolation='nearest')
        row_axes[2].set_title(f'Foreground: {(target == 1).sum() / target.size + (target == 2).sum() / target.size:.1%}')
        for ax in row_axes:
            ax.axis('off')
    plt.tight_layout()
    plt.show()
