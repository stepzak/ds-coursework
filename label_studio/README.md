# Задания для Label Studio

### 156 английских отзывов и 51 фотография. Codebook 1.0.

| Файл | Назначение                                   |
|---|----------------------------------------------|
| `sentiment_tasks.json` | Предложения без готовых ответов              |
| `pets_tasks.json` | Ссылки на локальные фотографии без масок     |
| `sentiment.xml`, `pets.xml` | Интерфейсы разметки                          |
| `manifest.json` | Источники отзывов, авторство и лицензии фото |
| [Codebook отзывов](../docs/codebook_sentiment.md) | Правила Positive / Negative / Skip           |
| [Codebook животных](../docs/codebook_pets.md) | Правила кистевой сегментации Cat / Dog       |

## Подготовка проектов

Запустить Docker Desktop, затем выполнить из папки `label_studio/`:

```sh
docker compose up -d
```

1. Открыть `http://127.0.0.1:8080`, создать учётную запись и проекты Sentiment и Pets.
2. В каждом проекте открыть Settings → Labeling Interface и вставить соответствующий XML.
3. Для Pets открыть Settings → Cloud Storage → Add Source Storage → Local Files. Указать `/label-studio/files/images`. Сохранить подключение **без Sync**, чтобы не создать дополнительные задачи из фотографий.
4. Импортировать в проекты соответствующие `*_tasks.json`.

> Работа и экспорт описаны в codebook