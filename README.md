# frameworks-coursework

Полный воспроизводимый EDA двух датасетов: Sentiment Labelled Sentences и Oxford-IIIT Pet.

Основной результат:

- [01_sentiment_eda.ipynb](notebooks/01_sentiment_eda.ipynb)
- [02_pets_eda.ipynb](notebooks/02_pets_eda.ipynb)

Графики: `reports/figures/`; таблицы аудита и статистики: `reports/tables/`; итоговые разбиения: `data/processed/*_splits.csv`. Полные исходные архивы и извлечённые файлы находятся в `data/raw/`, целевые PNG-маски - `data/processed/pets_masks/`.

## Запуск

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -r requirements-notebook.txt
```

## Методические решения

- Seed = 42. Тексты: 70/15/15 по source×label после нормализованной дедупликации; все исходные строки сохраняются отдельно. Нормализованный ключ не заменяет исходный текст для обучения.
- Pet: официальный test сохраняется; validation ~20% trainval с группировкой близких изображений и стратификацией пород. Trainval-группы, связанные с test, исключаются и остаются в манифесте с причиной.
- Собственная разметка: закрашиваются только кошки и собаки; незакрашенные пиксели автоматически становятся фоном (0). Фон отдельно закрашивать не нужно.
- Маски Pet: 0/1/2 - три целевых класса; 255 - игнорируемая неопределённая область. Использовать `ignore_index=255`. При resize маски допускается только nearest neighbour.
- Perceptual hash даёт кандидатов, а не доказательство идентичности животных; группировка кандидатов консервативна. Подробные ограничения приведены в отчёте.

## Ручное обогащение и разметка

- [Третий ноутбук: fixed, метрики и обогащение](notebooks/03_handmade_enrichment.ipynb).
- [Инструкция, скрипты Cohen/IoU/Dice и результаты initial/fixed](label_studio/README.md).
- [Codebook отзывов](docs/codebook_sentiment.md), [codebook сегментации](docs/codebook_pets.md).

Исходные JSON двух аннотаторов сохранены. Skip остаются исключёнными. Маски agreement сохраняют совпавшие пиксели; разногласия получают 255 и не объявляются экспертной разметкой. Метрики измеряют согласие людей, не качество модели.

Готовые объединённые манифесты находятся в data/processed/enriched/. Использовать только train/validation/test, исключая строки excluded_*. Производные PNG доступны локально в data/processed/handmade/masks/ и воспроизводятся третьим ноутбуком или командой pets_metrics.py --stage fixed --export-masks.
