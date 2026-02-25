# LSM Tree & Inverted Index Project

Реализация LSM-дерева с leveling стратегией и инвертированного индекса с Roaring bitmaps.

## Структура проекта

```
.
├── lsm_tree/              # LSM-дерево с leveling compaction
│   ├── __init__.py
│   └── lsm_tree.py        # Основная реализация LSM
│
├── inverted_index/        # Инвертированный индекс с Roaring bitmaps
│   ├── __init__.py
│   ├── core.py            # Standalone реализация
│   └── lsm_based.py       # Интеграция с LSM-деревом
│
├── tests/                 # Тесты
│   ├── __init__.py
│   ├── test_lsm_tree.py           # Тесты LSM-дерева
│   ├── test_inverted_index.py     # Тесты standalone индекса
│   └── test_lsm_inverted_index.py # Тесты LSM-интегрированного индекса
│
└── requirements.txt       # Зависимости
```

## Быстрый старт

### Установка

```bash
# Создать виртуальное окружение
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate     # Windows

# Установить зависимости
pip install -r requirements.txt
```

### Использование LSM-дерева

```python
from lsm_tree import LSMTree

# Создание LSM-дерева
lsm = LSMTree(storage_dir="my_lsm_storage")

# Добавление данных
lsm.add("key1", "value1")
lsm.add("key2", "value2")

# Поиск
value = lsm.get("key1")
print(value)  # "value1"

# Range query
results = lsm.range_get("key1", "key5")
```

### Использование инвертированного индекса

#### Вариант 1: Standalone (простой)

```python
from inverted_index import InvertedIndex

# Создание индекса
index = InvertedIndex(use_stemming=True, remove_stopwords=True)

# Добавление документов
index.add_document("Python is great for machine learning")
index.add_document("Java is used in enterprise")

# Поиск с булевыми запросами
results = index.search_boolean("python AND machine")
print(results)  # [0]
```

#### Вариант 2: LSM-интегрированный (рекомендуется)

```python
from inverted_index import LSMInvertedIndex

# Создание индекса с LSM-деревом
index = LSMInvertedIndex(
    storage_dir="my_index",
    use_stemming=True,
    remove_stopwords=True
)

# Добавление документов
index.add_document("Python is great for machine learning")
index.add_document("Java is used in enterprise")

# Поиск
results = index.search_boolean("python AND machine")
print(results)  # [0]

# Получение документа
doc = index.get_document(0)
print(doc)  # "Python is great for machine learning"
```

## Запуск тестов

```bash
# Тесты LSM-дерева
python3 tests/test_lsm_tree.py

# Тесты инвертированного индекса
python3 tests/test_inverted_index.py

# Тесты LSM-интегрированного индекса
python3 tests/test_lsm_inverted_index.py
```

## Основные возможности

### LSM-дерево
- Memory Buffer для быстрой записи
- Leveling compaction (R=2)
- Bloom filters для быстрого поиска
- Range queries
- Запись на диск (disk persistance)

### Инвертированный индекс
- Roaring Bitmaps для эффективного хранения ID документов
- Булевы запросы: AND, OR, NOT с поддержкой скобок
- Препроцессинг текста:
  - Stemming (Porter Stemmer)
  - Stop-words removal
  - Tokenization
- Использует LSM-дерево для хранения
