# Step 1 : Build the application
FROM python:3.11-slim
RUN pip install poetry
WORKDIR /app
COPY . /app/
RUN poetry config virtualenvs.in-project true \
 && poetry install --no-interaction --no-ansi \
 && poetry build \
 && pip install --no-cache-dir dist/*.whl
CMD python3 -m ts341_example.app
