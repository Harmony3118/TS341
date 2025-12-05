# Step 1 : Build the application
FROM python:3.11-slim
RUN pip install poetry
WORKDIR /app
COPY . /app/
RUN apt update && apt install -y ffmpeg libsm6 libxext6 wget
RUN poetry config virtualenvs.in-project true \
 && poetry install --no-interaction --no-ansi \
 && poetry build \
 && pip install --no-cache-dir dist/*.whl
RUN echo "cd ts341_project && python3 -m ts341_project.yolo_plus_imm" > ~/.bashrc
CMD /bin/bash

#FROM python:3.10-slim
#RUN apt update && apt install -y ffmpeg libsm6 libxext6 wget && wget https://degen-robots.serveminecraft.net/ts341-0.1.0-py3-none-any.whl
#RUN pip install *.whl
