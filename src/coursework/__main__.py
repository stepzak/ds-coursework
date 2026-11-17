import argparse
from . import download, sentiment, pets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['all', 'sentiment', 'pets'], default='all')
    parser.add_argument('--download', action='store_true', help='Download and extract official archives first')
    args = parser.parse_args()
    if args.download:
        download.prepare()
    if args.dataset in ['all', 'sentiment']:
        sentiment.main()
    if args.dataset in ['all', 'pets']:
        pets.main()


if __name__ == '__main__':
    main()
