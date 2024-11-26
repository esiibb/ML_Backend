# 1. Gunakan base image Node.js
FROM node:18

# 2. Install dependencies tambahan untuk TensorFlow.js
RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6

# 3. Tentukan direktori kerja
WORKDIR /app
COPY firestore-key.json /app/
# 4. Salin file package.json dan package-lock.json
COPY package*.json ./

# 5. Install dependensi aplikasi
RUN npm install

# 6. Salin semua file proyek ke container
COPY . .

# 7. Expose port (Cloud Run otomatis menggunakan $PORT)
EXPOSE 8080

# 8. Jalankan aplikasi
CMD ["node", "index.js"]
