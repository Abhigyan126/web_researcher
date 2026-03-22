# Use lightweight Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your code
COPY . .

# Expose port
EXPOSE 8000

# Run FastAPI
CMD ["python", "main.py"]
