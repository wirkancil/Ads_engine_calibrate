# Gunakan image dasar Python
FROM python:3.11-slim

# Set direktori kerja
WORKDIR /apps

# Copy file requirements dan install dependency
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy semua file ke dalam container
COPY . .

# Expose port Flask
EXPOSE 5500

# Jalankan aplikasi Flask
CMD ["python", "simulate_calibrate.py"]
