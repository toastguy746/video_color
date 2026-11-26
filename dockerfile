FROM python:3.12-slim

# ffmpeg / ffprobe 설치 (영상 처리에 필수)
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

COPY app.py index.html ./

# 영상/결과물은 /data (볼륨)에 저장
ENV WORK_DIR=/data \
    NO_BROWSER=1 \
    PYTHONUNBUFFERED=1

EXPOSE 5000

# 이 앱은 메모리에 상태(FILES, JOBS)를 두므로 워커는 반드시 1개, 대신 스레드를 여러 개 사용
# 0.0.0.0 으로 열어야 컨테이너 밖에서 접속 가능. --timeout 0 : 큰 영상 업로드/인코딩이 끊기지 않게
CMD ["gunicorn", "-w", "1", "--threads", "16", "--timeout", "0", "-b", "0.0.0.0:5000", "app:app"]
