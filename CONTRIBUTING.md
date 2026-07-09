

## `.github/workflows/ci.yml`

```yaml
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]

env:
  PYTHON_VERSION: "3.11"
  DOCKER_BUILDKIT: 1

jobs:
  # ===========================================================================
  # LINT & FORMAT CHECK
  # ===========================================================================
  lint:
    name: Lint & Format
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Install linters
        run: |
          pip install black==24.4.2 isort==5.13.2 flake8==7.0.0

      - name: Check black formatting
        run: black --check --diff src/ tests/

      - name: Check import sorting
        run: isort --check-only --diff src/ tests/

      - name: Run flake8
        run: flake8 src/ tests/ --max-line-length=100 --extend-ignore=E203,W503

  # ===========================================================================
  # UNIT TESTS
  # ===========================================================================
  test:
    name: Unit Tests
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Install dependencies
        run: |
          pip install pytest==8.2.0 pytest-asyncio==0.23.7 httpx==0.27.0
          pip install -r src/gateway/requirements.txt

      - name: Run tests
        run: pytest tests/ -v --tb=short

  # ===========================================================================
  # DOCKER BUILD
  # ===========================================================================
  docker-build:
    name: Docker Build
    runs-on: ubuntu-latest
    strategy:
      matrix:
        service: [gateway]
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Build ${{ matrix.service }}
        uses: docker/build-push-action@v5
        with:
          context: .
          file: deployment/docker/${{ matrix.service }}.Dockerfile
          push: false
          tags: tensor-pipeline/${{ matrix.service }}:latest
          cache-from: type=gha
          cache-to: type=gha,mode=max

  # ===========================================================================
  # DOCKER COMPOSE VALIDATION
  # ===========================================================================
  compose-check:
    name: Compose Validation
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Validate docker-compose.yml
        run: docker compose config

  # ===========================================================================
  # SECURITY SCAN (Trivy)
  # ===========================================================================
  security-scan:
    name: Security Scan
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Build gateway image
        run: docker build -f deployment/docker/gateway.Dockerfile -t tensor-pipeline/gateway:scan .

      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: tensor-pipeline/gateway:scan
          format: sarif
          output: trivy-results.sarif

      - name: Upload scan results
        uses: github/codeql-action/upload-sarif@v3
        if: always()
        with:
          sarif_file: trivy-results.sarif
```

---

## `CONTRIBUTING.md`

```markdown
# Contributing to Tensor Pipeline

Спасибо за интерес к проекту! Ниже — правила, как вносить изменения.

---

## 🚀 Быстрый старт

```bash
# 1. Fork репозитория на GitHub

# 2. Клонируй свой fork
git clone https://github.com/karamik/tensor-pipeline.git
cd tensor-pipeline

# 3. Создай ветку
git checkout -b feature/краткое-описание

# 4. Внеси изменения, закоммить
git add .
git commit -m "feat: добавил X"

# 5. Push и открой Pull Request
git push origin feature/краткое-описание
```

---

## 📋 Требования к Pull Request

### Перед отправкой

- [ ] Код проходит линтеры: `make lint`
- [ ] Тесты проходят: `make test`
- [ ] Docker образы собираются: `docker compose build`
- [ ] README обновлён, если менял API или архитектуру
- [ ] Commit messages соответствуют [Conventional Commits](https://www.conventionalcommits.org/)

### Формат коммитов

| Префикс | Когда использовать |
|---------|-------------------|
| `feat:` | Новая функциональность |
| `fix:` | Исправление бага |
| `docs:` | Изменения в документации |
| `refactor:` | Рефакторинг без новых фич |
| `test:` | Добавление/изменение тестов |
| `chore:` | Обновление зависимостей, CI и т.д. |

Примеры:
```bash
git commit -m "feat: add TensorRT INT8 calibration support"
git commit -m "fix: handle empty batch in Gateway"
git commit -m "docs: update benchmark table with A100 results"
```

---

## 🏗️ Структура веток

```
main        → production-ready, стабильная версия
develop     → интеграционная ветка для фич
feature/*   → новая функциональность
fix/*       → исправление багов
hotfix/*    → срочные фиксы для main
```

---

## 🧪 Локальное тестирование

```bash
# Линтеры
make lint

# Тесты (требует запущенных сервисов)
make serve
# в другом терминале:
make test

# Нагрузочное тестирование
make benchmark
```

---

## 🐛 Сообщение о баге

Открывай [Issue](https://github.com/karamik/tensor-pipeline/issues/new) с шаблоном:

```markdown
**Описание**
Что сломалось?

**Как воспроизвести**
1. Шаг 1
2. Шаг 2
3. ...

**Ожидаемое поведение**
Что должно было произойти?

**Окружение**
- OS:
- Docker version:
- GPU:
- Логи (если есть):
```

---

## 💡 Предложение фичи

Открывай [Discussion](https://github.com/karamik/tensor-pipeline/discussions) или Issue с префиксом `[RFC]`:

```markdown
[RFC] Добавить поддержку ONNX Runtime DirectML

**Проблема**
Сейчас нет поддержки Windows GPU без CUDA.

**Предложение**
Добавить бэкенд ONNX Runtime с DirectML execution provider.

**Альтернативы**
- OpenVINO (медленнее на Windows)
- TensorRT (требует CUDA)
```

---

## 📞 Контакты

- Email: totalprotocol@proton.me
- Issues: https://github.com/karamik/tensor-pipeline/issues
- Discussions: https://github.com/karamik/tensor-pipeline/discussions

---

## 📄 Лицензия

Внося изменения, ты соглашаешься с [Apache-2.0 License](LICENSE).
```

---

