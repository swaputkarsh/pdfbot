FROM python:3.11.5

WORKDIR /app

COPY requirements.txt .
RUN pip --no-cache-dir install -r requirements.txt

COPY . .

EXPOSE 5052

CMD ["gunicorn", "-b", "0.0.0.0:5052", "app:app"]
