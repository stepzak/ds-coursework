from pathlib import Path
import json
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/frameworks-coursework-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SEED = 42
FIG = ROOT / 'reports/figures'
TAB = ROOT / 'reports/tables'
PROCESSED = ROOT / 'data/processed'
for folder in [FIG, TAB, PROCESSED]:
    folder.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'figure.dpi': 120, 'savefig.dpi': 150, 'axes.spines.top': False,
                     'axes.spines.right': False, 'font.size': 10})


def table(df, name, index=False):
    df.to_csv(TAB / f'{name}.csv', index=index)


def savefig(name):
    plt.tight_layout()
    plt.savefig(FIG / f'{name}.png', bbox_inches='tight')
    plt.close()


def dump(data, path):
    def encode(x):
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, np.ndarray):
            return x.tolist()
        raise TypeError(type(x).__name__)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, default=encode), encoding='utf-8')


def describe(df, columns):
    return df[columns].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).T
