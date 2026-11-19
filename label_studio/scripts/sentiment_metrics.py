import argparse
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from annotation_io import LS, load_pair, sentiment_label, write_json


def calculate(stage='fixed'):
    first, second = load_pair('sentiment', stage)
    rows = []
    for key in sorted(first):
        a, b = sentiment_label(first[key]), sentiment_label(second[key])
        status = 'skip' if a is None or b is None else 'agreed' if a == b else 'disagreement'
        rows.append({'object_id': key, 'label_1': a, 'label_2': b, 'status': status})
    audit = pd.DataFrame(rows)
    accepted = audit[audit.status != 'skip']
    if len(accepted) < 2:
        raise ValueError('At least two paired accepted labels are required')
    a, b = accepted.label_1.astype(int), accepted.label_2.astype(int)
    matrix = confusion_matrix(a, b, labels=[0, 1])
    expected = float((matrix.sum(axis=0) * matrix.sum(axis=1)).sum() / matrix.sum() ** 2)
    kappa = float(cohen_kappa_score(a, b, labels=[0, 1])) if expected < 1 else None
    summary = {'stage': stage, 'objects': len(audit), 'paired_accepted': len(accepted),
               'skip_1': int(audit.label_1.isna().sum()), 'skip_2': int(audit.label_2.isna().sum()),
               'excluded_any_skip': int((audit.status == 'skip').sum()),
               'agreements': int((a == b).sum()), 'disagreements': int((a != b).sum()),
               'agreement': float((a == b).mean()), 'cohen_kappa': kappa,
               'confusion_matrix_labels': [0, 1], 'confusion_matrix': matrix.tolist()}
    return summary, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['initial', 'fixed'], default='fixed')
    args = parser.parse_args()
    summary, audit = calculate(args.stage)
    output = LS / 'annotations/metrics' / args.stage
    write_json(summary, output / 'sentiment_summary.json')
    audit.to_csv(output / 'sentiment_objects.csv', index=False)
    print(f"{args.stage}: paired={summary['paired_accepted']}, skips={summary['excluded_any_skip']}, "
          f"agreement={summary['agreement']:.4f}, kappa={summary['cohen_kappa']}")


if __name__ == '__main__':
    main()
