import re
import unicodedata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from .common import ROOT, SEED, FIG, PROCESSED, table, savefig, dump, describe

FILES = {'amazon': 'amazon_cells_labelled.txt', 'imdb': 'imdb_labelled.txt', 'yelp': 'yelp_labelled.txt'}


def normalize(text):
    text = unicodedata.normalize('NFC', text).casefold()
    return ' '.join(''.join(' ' if unicodedata.category(c).startswith('P') else c for c in text).split())


def parse_file(path, source):
    content = path.read_bytes().decode('utf-8', errors='strict')
    records, anomalies, pending = [], [], []
    start = 1
    for line_no, line in enumerate(content.split('\n'), 1):
        if not pending and not line.strip():
            continue
        if not pending:
            start = line_no
        pending.append(line.rstrip('\r'))
        if re.search(r'\t[01]\s*$', line):
            joined = '\n'.join(pending)
            text, label = joined.rsplit('\t', 1)
            records.append({'id': f'{source}_{len(records)+1:04}', 'source': source,
                            'line_start': start, 'line_end': line_no, 'text': text, 'label': int(label)})
            if len(pending) > 1:
                anomalies.append({'source': source, 'line': start, 'kind': 'multiline_reconstructed', 'text': joined})
            if any(c in text for c in ['\x85', '\u2028', '\u2029']):
                anomalies.append({'source': source, 'line': start, 'kind': 'unicode_line_separator', 'text': joined})
            pending = []
    if pending and any(x.strip() for x in pending):
        raise ValueError(f'Unparsed trailing record: {path}: {pending}')
    return records, anomalies, {'source': source, 'file': path.name, 'encoding': 'utf-8 (strict)',
                               'physical_lf_lines': len(content.rstrip('\n').split('\n')), 'records': len(records),
                               'header': False, 'delimiter': 'TAB',
                               'unicode_separator_records': sum(x['kind'] == 'unicode_line_separator' for x in anomalies)}


def build_splits(df):
    conflicts = set(df.groupby('normalized').label.nunique().loc[lambda s: s > 1].index)
    eligible = df[~df.normalized.isin(conflicts)].copy()
    clean = eligible.sort_values(['source', 'id']).drop_duplicates('normalized').copy()
    train, rest = train_test_split(clean, test_size=.3, random_state=SEED,
                                  stratify=clean.source + '_' + clean.label.astype(str))
    val, test = train_test_split(rest, test_size=.5, random_state=SEED,
                                stratify=rest.source + '_' + rest.label.astype(str))
    clean['split'] = ''
    for name, part in [('train', train), ('validation', val), ('test', test)]:
        clean.loc[part.index, 'split'] = name
    mapping = clean.set_index('normalized')['split']
    df['split'] = df.normalized.map(mapping).fillna('excluded_conflict')
    clean.to_csv(PROCESSED / 'sentiment_splits.csv', index=False)
    df.to_csv(PROCESSED / 'sentiment_all_rows.csv', index=False)
    sets = [set(clean.loc[clean.split == x, 'normalized']) for x in ['train', 'validation', 'test']]
    assert not any(sets[i] & sets[j] for i in range(3) for j in range(i))
    table(pd.crosstab([clean.split, clean.source], clean.label).reset_index(), 'sentiment_split_balance')
    return clean, conflicts


def main():
    records, anomalies, structure = [], [], []
    for source, name in FILES.items():
        rows, issues, info = parse_file(ROOT / 'data/raw/sentiment/sentiment labelled sentences' / name, source)
        records.extend(rows); anomalies.extend(issues); structure.append(info)
    df = pd.DataFrame(records)
    df['normalized'] = df.text.map(normalize)
    df['chars'] = df.text.str.len()
    df['words'] = df.text.map(lambda s: len(re.findall(r"\b\w+(?:'\w+)?\b", s)))
    table(pd.DataFrame(structure), 'sentiment_file_structure')
    table(pd.DataFrame(anomalies), 'sentiment_parse_anomalies')
    assert len(df) == 3000 and set(df.label) == {0, 1}
    assert not df.text.isna().any() and not df.normalized.eq('').any()
    balance = pd.crosstab(df.source, df.label)
    table(balance.reset_index(), 'sentiment_class_counts')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    df.label.value_counts().sort_index().plot.bar(ax=axes[0], color=['#d95f5f', '#38a690'], rot=0)
    axes[0].set(title='All reviews', xlabel='Label', ylabel='Reviews')
    balance.plot.bar(ax=axes[1], rot=0, color=['#d95f5f', '#38a690'])
    axes[1].set(title='Reviews by source', ylabel='Reviews')
    savefig('sentiment_balance')
    duplicate_stats = {}
    for key in ['text', 'normalized']:
        groups = df.groupby(key).size()
        duplicate_stats[key] = {'groups': int((groups > 1).sum()), 'rows_in_groups': int(groups[groups > 1].sum()),
                                'redundant_rows': int((groups - 1).clip(lower=0).sum())}
        table(df[df.duplicated(key, keep=False)].sort_values(key), f'sentiment_duplicates_{key}')
    clean, conflicts = build_splits(df)
    table(df[df.normalized.isin(conflicts)], 'sentiment_conflicting_labels')
    table(describe(df, ['chars', 'words']), 'sentiment_lengths_overall', index=True)
    for by in ['label', 'source']:
        table(df.groupby(by)[['chars', 'words']].describe(percentiles=[.05, .5, .95]).stack(0, future_stack=True).reset_index(), f'sentiment_lengths_by_{by}')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for label, color in [(0, '#d95f5f'), (1, '#38a690')]:
        axes[0].hist(df[df.label == label].words, bins=np.arange(0, 91, 3), alpha=.55, label=str(label), color=color)
    axes[0].legend(title='Label'); axes[0].set(xlabel='Words', ylabel='Reviews', title='Review length by sentiment')
    axes[1].boxplot([df[df.source == s].words for s in FILES], tick_labels=list(FILES), showfliers=True)
    axes[1].set(ylabel='Words', title='Review length by source')
    savefig('sentiment_lengths')
    vectorizer = CountVectorizer(lowercase=True, stop_words='english', min_df=2, token_pattern=r'(?u)\b[a-zA-Z][a-zA-Z\x27]+\b')
    X = vectorizer.fit_transform(df.text)
    vocab = vectorizer.get_feature_names_out()
    terms = []
    for grouping in ['label', 'source']:
        for value in sorted(df[grouping].unique()):
            selected = (df[grouping] == value).to_numpy()
            counts = np.asarray(X[selected].sum(axis=0)).ravel()
            others = np.asarray(X[~selected].sum(axis=0)).ravel()
            docfreq = np.asarray((X[selected] > 0).sum(axis=0)).ravel()
            logratio = np.log((counts + 1) / (counts.sum() + len(vocab))) - np.log((others + 1) / (others.sum() + len(vocab)))
            frame = pd.DataFrame({'grouping': grouping, 'group': value, 'word': vocab,
                                  'count': counts, 'document_frequency': docfreq, 'log_frequency_ratio': logratio})
            terms.append(frame)
    terms = pd.concat(terms, ignore_index=True)
    table(terms, 'sentiment_word_statistics')
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for label in [0, 1]:
        part = terms[(terms.grouping == 'label') & (terms['group'] == label)]
        for col, metric in [(0, 'count'), (1, 'log_frequency_ratio')]:
            top = part[part.document_frequency >= 5].nlargest(12, metric).sort_values(metric)
            axes[label, col].barh(top.word, top[metric], color=['#d95f5f', '#38a690'][label])
            axes[label, col].set_title(f'Label {label}: {metric}')
    savefig('sentiment_vocabulary')
    tfidf = TfidfVectorizer(min_df=2, max_df=.98, sublinear_tf=True)
    T = tfidf.fit_transform(df.text)
    svd = TruncatedSVD(n_components=50, random_state=SEED)
    reduced = svd.fit_transform(T)
    coords = TSNE(n_components=2, perplexity=35, init='pca', learning_rate='auto', random_state=SEED,
                  max_iter=1000, n_jobs=2).fit_transform(reduced)
    table(pd.DataFrame({'id': df.id, 'source': df.source, 'label': df.label, 'x': coords[:, 0], 'y': coords[:, 1]}), 'sentiment_tsne')
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for value in [0, 1]:
        mask = df.label == value
        axes[0].scatter(coords[mask, 0], coords[mask, 1], s=7, alpha=.5, label=str(value))
    for value in FILES:
        mask = df.source == value
        axes[1].scatter(coords[mask, 0], coords[mask, 1], s=7, alpha=.5, label=value)
    for ax, title in zip(axes, ['Sentiment labels', 'Sources']):
        ax.legend(); ax.set(title=title, xticks=[], yticks=[])
    savefig('sentiment_tsne')
    pos = re.compile(r'\b(good|great|excellent|love|best|nice|happy)\b', re.I)
    neg = re.compile(r'\b(bad|poor|worst|hate|terrible|boring|awful|not|never)\b', re.I)
    df['mixed_cues'] = df.text.map(lambda t: bool(pos.search(t) and neg.search(t)))
    samples = pd.concat([df.groupby(['source', 'label'], group_keys=False).sample(n=3, random_state=SEED).assign(reason='stratified_sample'),
                         df[df.mixed_cues].sort_values('chars').head(20).assign(reason='mixed_lexical_cues'),
                         df.nsmallest(8, 'words').assign(reason='very_short')]).drop_duplicates('id')
    table(samples, 'sentiment_review_candidates')
    summary = {'rows': len(df), 'class_counts': df.label.value_counts().sort_index().to_dict(),
               'by_source': balance.to_dict(), 'missing': int(df.text.isna().sum()),
               'empty': int(df.normalized.eq('').sum()), 'duplicates': duplicate_stats,
               'conflicting_groups': len(conflicts), 'conflicting_rows': int(df.normalized.isin(conflicts).sum()),
               'clean_rows': len(clean), 'splits': clean.split.value_counts().to_dict(),
               'multiline_records': sum(x['kind'] == 'multiline_reconstructed' for x in anomalies),
               'unicode_separator_records': sum(x['kind'] == 'unicode_line_separator' for x in anomalies),
               'svd_explained_variance': svd.explained_variance_ratio_.sum(),
               'lengths': describe(df, ['chars', 'words']).to_dict(orient='index'),
               'lengths_by_source': {s: {'mean_words': float(g.words.mean()), 'median_words': float(g.words.median()),
                                       'mean_chars': float(g.chars.mean())} for s, g in df.groupby('source')},
               'seed': SEED, 'split_intersections': 0}
    dump(summary, ROOT / 'reports/sentiment_summary.json')
    print('Sentiment complete:', len(df), 'raw;', len(clean), 'clean', flush=True)
    return summary


if __name__ == '__main__':
    main()
