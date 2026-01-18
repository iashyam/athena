FROM python:3.12-slim

WORKDIR /app

COPY ./requirements_production.txt .

RUN pip install --no-cache-dir -r requirements_production.txt

COPY ./app ./app
# COPY ./model.onnx .
# COPY ./model.onnx.data .

EXPOSE 5911

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5911"]