# frameworks-coursework

Полный воспроизводимый EDA двух датасетов: Sentiment Labelled Sentences и Oxford-IIIT Pet.

Основной результат:

- [01_sentiment_eda.ipynb](notebooks/01_sentiment_eda.ipynb)
- [02_pets_eda.ipynb](notebooks/02_pets_eda.ipynb)

Графики: `reports/figures/`; таблицы аудита и статистики: `reports/tables/`; итоговые разбиения: `data/processed/*_splits.csv`. Полные исходные архивы и извлечённые файлы находятся в `data/raw/`, целевые PNG-маски - `data/processed/pets_masks/`.

## Запуск

Из корня проекта (Python 3.11+, проверено на Python 3.13):

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -r requirements-notebook.txt
python -m pip install -e .
python -m src.coursework --download
```

`--download` загружает отсутствующие официальные архивы и извлекает их с проверкой безопасных путей. Существующие архивы повторно не скачиваются; их контрольные суммы сохраняются в `data/sources.json`. Проверка целостности не является сверкой с контрольной суммой издателя, поскольку издатель её не предоставил. Если данные уже распакованы, достаточно `PYTHONPATH=src python -m coursework`. Повреждённый архив удалить и загрузить заново.

Отдельные запуски и обновление выполненных ноутбуков:

```bash
python -m src.coursework --dataset sentiment
python -m src.coursework --dataset pets
python -m src.coursework.build_notebooks sentiment --execute
python -m src.coursework.build_notebooks pets --execute
```

Каждая команда ноутбука заново выполняет соответствующий EDA и сохраняет результаты ячеек. При необходимости запустить интерфейс: `jupyter lab notebooks/` (дополнительная установка `jupyterlab`). В PyCharm открыть корень проекта и выбрать `.venv/bin/python` как интерпретатор; ноутбуки можно просматривать с уже сохранёнными результатами. В архивном проекте не фиксируется пользовательская конфигурация `.idea`.

## Методические решения

- Seed = 42. Тексты: 70/15/15 по source×label после нормализованной дедупликации; все исходные строки сохраняются отдельно. Нормализованный ключ не заменяет исходный текст для обучения.
- Pet: официальный test сохраняется; validation ≈20% trainval с группировкой близких изображений и стратификацией пород. Trainval-группы, связанные с test, исключаются и остаются в манифесте с причиной.
- Собственная разметка: закрашиваются только кошки и собаки; незакрашенные пиксели автоматически становятся фоном (0). Фон отдельно закрашивать не нужно.
- Маски Pet: 0/1/2 - три целевых класса; 255 - игнорируемая неопределённая область. Использовать `ignore_index=255`. При resize маски допускается только nearest neighbour.
- Perceptual hash даёт кандидатов, а не доказательство идентичности животных; группировка кандидатов консервативна. Подробные ограничения приведены в отчёте.