# 1. Gunakan base image Node.js
FROM node:18

# 2. Install dependencies tambahan untuk TensorFlow.js
RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6

# 3. Tentukan direktori kerja
WORKDIR /app

# 4. Salin file service-account.json ke dalam container
COPY service-account.json /app/

# 5. Set environment variable untuk kredensial
ENV GOOGLE_APPLICATION_CREDENTIALS=/app/service-account.json

# 6. Salin file package.json dan package-lock.json
COPY package*.json ./

# 7. Install dependensi aplikasi
RUN npm install

# 8. Salin semua file proyek ke dalam container
COPY . .

# 9. Expose port (Cloud Run otomatis menggunakan $PORT)
EXPOSE 8080

# 10. Jalankan aplikasi
CMD ["node", "index.js"]
