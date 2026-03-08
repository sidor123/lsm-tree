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
│   ├── lsm_based.py       # Интеграция с LSM-деревом
│   └── kgram_utils.py     # K-gram утилиты для wildcard поиска
│
├── tests/                 # Тесты
│   ├── __init__.py
│   ├── test_lsm_tree.py           # Тесты LSM-дерева
│   ├── test_inverted_index.py     # Тесты standalone индекса
│   ├── test_lsm_inverted_index.py # Тесты LSM-интегрированного индекса
│   └── test_kgram_utils.py        # Тесты k-gram утилит
│
├── demo_prefix_wildcard.py # Демонстрация prefix и wildcard поиска
└── requirements.txt        # Зависимости
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

# Создание индекса с поддержкой k-gram для wildcard поиска
index = InvertedIndex(use_stemming=True, remove_stopwords=True, enable_kgram=True)

# Добавление документов
index.add_document("Python is great for machine learning")
index.add_document("Java is used in enterprise")

# Поиск с булевыми запросами
results = index.search_boolean("python AND machine")
print(results)  # [0]

# Поиск по префиксу
results = index.search_prefix("mach")
print(results)  # [0] - находит "machine"

# Wildcard поиск
results = index.search_wildcard("mach*")
print(results)  # [0] - находит "machine"

results = index.search_wildcard("*thon")
print(results)  # [0] - находит "python"
```

#### Вариант 2: LSM-интегрированный (рекомендуется)

```python
from inverted_index import LSMInvertedIndex

# Создание индекса с LSM-деревом и k-gram поддержкой
index = LSMInvertedIndex(
    storage_dir="my_index",
    use_stemming=True,
    remove_stopwords=True,
    enable_kgram=True
)

# Добавление документов
index.add_document("Python is great for machine learning")
index.add_document("Java is used in enterprise")

# Булевый поиск
results = index.search_boolean("python AND machine")
print(results)  # [0]

# Поиск по префиксу (использует LSM range query)
results = index.search_prefix("mach")
print(results)  # [0] - находит "machine"

# Wildcard поиск (использует k-gram индекс)
results = index.search_wildcard("*thon")
print(results)  # [0] - находит "python"

# Получение документа
doc = index.get_document(0)
print(doc)  # "Python is great for machine learning"
```

## Запуск тестов

```bash
# Тесты LSM-дерева
PYTHONPATH=. python3 tests/test_lsm_tree.py

# Тесты инвертированного индекса
PYTHONPATH=. python3 tests/test_inverted_index.py

# Тесты LSM-интегрированного индекса
PYTHONPATH=. python3 tests/test_lsm_inverted_index.py

# Тесты k-gram утилит
PYTHONPATH=. python3 tests/test_kgram_utils.py
```

## Демонстрация

```bash
# Запустить демо prefix и wildcard поиска
python3 demo_prefix_wildcard.py
```

## Основные возможности

### LSM-дерево
- Memory Buffer для быстрой записи
- Leveling compaction (R=2)
- Bloom filters для быстрого поиска
- Range queries
- Запись на диск (disk persistance)

### Инвертированный индекс
- **Roaring Bitmaps** для эффективного хранения ID документов
- **Булевы запросы**: AND, OR, NOT с поддержкой скобок
- **Поиск по префиксу**: быстрый поиск терминов, начинающихся с заданного префикса
- **Wildcard поиск с k-gram индексом**: поддержка паттернов с одним символом `*`
  - Примеры: `"prog*"`, `"*thon"`, `"pro*am"`
  - Использует биграммы (k=2) для эффективного поиска
- **Препроцессинг текста**:
  - Stemming (Porter Stemmer)
  - Stop-words removal
  - Tokenization
- **Персистентность**: использует LSM-дерево для хранения

### K-Gram индекс
- Автоматическая генерация биграмм для всех терминов
- Граничные маркеры ($) для точного сопоставления
- Эффективная фильтрация кандидатов через пересечение множеств
- Финальная проверка через regex для точности
- Опциональное отключение для экономии памяти (`enable_kgram=False`)

## Примеры использования

### Поиск по префиксу

```python
from inverted_index import InvertedIndex

index = InvertedIndex(enable_kgram=True)
index.add_document("Python programming is powerful")
index.add_document("Java programming language")

# Найти все документы с терминами, начинающимися на "prog"
results = index.search_prefix("prog")
print(results)  # [0, 1] - оба документа содержат "programming"
```

### Wildcard поиск

```python
from inverted_index import InvertedIndex

index = InvertedIndex(enable_kgram=True)
index.add_document("Python programming is powerful")
index.add_document("JavaScript for web development")

# Wildcard в конце
results = index.search_wildcard("prog*")
print(results)  # [0] - находит "programming"

# Wildcard в начале
results = index.search_wildcard("*thon")
print(results)  # [0] - находит "python"

# Wildcard в середине
results = index.search_wildcard("java*")
print(results)  # [1] - находит "javascript"
```

### Комбинированный поиск

```python
from inverted_index import LSMInvertedIndex

index = LSMInvertedIndex(storage_dir="my_index", enable_kgram=True)

# Добавляем документы
docs = [
    "Python programming for data science",
    "Machine learning with Python",
    "Java enterprise applications"
]
for doc in docs:
    index.add_document(doc)

# Булевый поиск
bool_results = index.search_boolean("python AND (data OR machine)")
print(f"Boolean: {bool_results}")  # [0, 1]

# Поиск по префиксу
prefix_results = index.search_prefix("mach")
print(f"Prefix: {prefix_results}")  # [1] - находит "machine"

# Wildcard поиск
wildcard_results = index.search_wildcard("*thon")
print(f"Wildcard: {wildcard_results}")  # [0, 1] - находит "python"
```

## Технические детали

### Алгоритм k-gram поиска

1. **Генерация k-грамм**: Для термина "program" генерируются биграммы:
   ```
   "$program$" → ["$p", "pr", "ro", "og", "gr", "ra", "am", "m$"]
   ```

2. **Извлечение k-грамм из паттерна**: Для `"pro*ing"`:
   ```
   Части: ["pro", "ing"]
   K-граммы: ["$p", "pr", "ro", "in", "ng", "g$"]
   ```

3. **Поиск кандидатов**: Пересечение множеств термов, содержащих все k-граммы

4. **Фильтрация**: Проверка кандидатов через regex `^pro.*ing$`

5. **Результат**: Объединение документов для всех подходящих терминов

### Производительность

- **Поиск по префиксу**:
  - InvertedIndex: O(V) где V - размер словаря
  - LSMInvertedIndex: O(log V) благодаря range query

- **Wildcard поиск**: O(G × T) где:
  - G - количество k-грамм в паттерне
  - T - среднее количество терминов на k-грамму

- **Overhead памяти**: ~3-5x от основного индекса для k-gram индекса

## Дополнительная документация

- [`ARCHITECTURE_PLAN.md`](ARCHITECTURE_PLAN.md) - Детальная архитектура и алгоритмы
- [`KGRAM_DIAGRAM.md`](KGRAM_DIAGRAM.md) - Визуальные примеры работы k-gram индекса
- [`IMPLEMENTATION_SUMMARY.md`](IMPLEMENTATION_SUMMARY.md) - Краткое руководство по реализации
