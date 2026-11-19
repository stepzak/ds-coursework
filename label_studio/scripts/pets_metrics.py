import argparse
import numpy as np
import pandas as pd
from PIL import Image
from annotation_io import ROOT, LS, load_pair, pet_mask, image_filename, skipped, write_json


def class_counts(first, second, class_id):
    a, b = first == class_id, second == class_id
    intersection = int(np.count_nonzero(a & b))
    total = int(a.sum() + b.sum())
    union = total - intersection
    return intersection, union, total


def calculate(stage='fixed', export_masks=False):
    first, second = load_pair('pets', stage)
    rows, objects = [], []
    destination = ROOT / 'data/processed/handmade/masks'
    if export_masks:
        for name in ['annotator_1', 'annotator_2', 'agreement']:
            (destination / name).mkdir(parents=True, exist_ok=True)
    for key in sorted(first):
        if skipped(first[key]) or skipped(second[key]):
            objects.append({'object_id': key, 'status': 'skip'})
            continue
        with Image.open(LS / 'images' / image_filename(first[key])) as image:
            image.load()
            size = image.size
        a, b = pet_mask(first[key], size), pet_mask(second[key], size)
        disagreements = a != b
        objects.append({'object_id': key, 'status': 'accepted', 'width': size[0], 'height': size[1],
                        'pixel_agreement': float((~disagreements).mean()),
                        'uncertain_pixels': int(disagreements.sum()), 'pixels': a.size})
        for class_id, name in [(0, 'background'), (1, 'cat'), (2, 'dog')]:
            intersection, union, total = class_counts(a, b, class_id)
            rows.append({'object_id': key, 'class_id': class_id, 'class': name,
                         'intersection': intersection, 'union': union, 'sum_areas': total,
                         'iou': intersection / union if union else np.nan,
                         'dice': 2 * intersection / total if total else np.nan})
        if export_masks:
            if stage != 'fixed':
                raise ValueError('Export training masks from fixed only')
            agreement = np.where(disagreements, 255, a).astype(np.uint8)
            for name, mask in [('annotator_1', a), ('annotator_2', b), ('agreement', agreement)]:
                Image.fromarray(mask).save(destination / name / f'{key}.png')
    per_class = pd.DataFrame(rows)
    audit = pd.DataFrame(objects)
    if per_class.empty:
        raise ValueError('No paired accepted images')
    foreground = per_class[per_class.class_id.isin([1, 2])]
    mean = per_class.groupby('class')[['iou', 'dice']].mean()
    summary = {'stage': stage, 'objects': len(audit), 'paired_accepted': int((audit.status == 'accepted').sum()),
               'excluded_any_skip': int((audit.status == 'skip').sum()),
               'foreground_macro_iou': float(foreground.iou.mean()),
               'foreground_macro_dice': float(foreground.dice.mean()),
               'per_class_mean': mean.to_dict(orient='index'),
               'per_class_defined_objects': per_class.groupby('class').iou.count().to_dict(),
               'pixel_agreement_mean': float(audit.pixel_agreement.mean()),
               'uncertain_fraction_pooled': float(audit.uncertain_pixels.sum() / audit.pixels.sum()),
               'aggregation': 'mean over image/class pairs with nonempty union; both absent excluded'}
    return summary, per_class, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['initial', 'fixed'], default='fixed')
    parser.add_argument('--export-masks', action='store_true')
    args = parser.parse_args()
    if args.export_masks and args.stage != 'fixed':
        parser.error('--export-masks is available for fixed only')
    summary, per_class, audit = calculate(args.stage, args.export_masks)
    output = LS / 'annotations/metrics' / args.stage
    write_json(summary, output / 'pets_summary.json')
    per_class.to_csv(output / 'pets_per_image_class.csv', index=False)
    audit.to_csv(output / 'pets_objects.csv', index=False)
    print(f"{args.stage}: paired={summary['paired_accepted']}, foreground "
          f"IoU={summary['foreground_macro_iou']:.4f}, Dice={summary['foreground_macro_dice']:.4f}")


if __name__ == '__main__':
    main()
