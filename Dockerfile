FROM python:3.11-slim

# Prevents Streamlit from asking for input
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# Install system dependencies for numpy, scipy, opencv
RUN apt-get update && apt-get install -y \
    build-essential \
    libatlas-base-dev \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app/Home.py", "--server.address=0.0.0.0"]
