# 1. Gunakan base image Node.js
FROM node:18

# 2. Tentukan direktori kerja di dalam container
WORKDIR /app

# 3. Salin file package.json dan package-lock.json
COPY package*.json ./

# 4. Install dependensi aplikasi
RUN npm install

# 5. Salin semua file dari proyek Anda ke dalam container
COPY . .

# 6. Expose port 3000 (sesuai dengan aplikasi Anda)
EXPOSE 8080

# 7. Tentukan perintah untuk menjalankan aplikasi
CMD ["node", "index.js"]
