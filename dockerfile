# Dockerfile tanpa service-account.json
FROM node:18
RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
EXPOSE 8080
CMD ["node", "index.js"]
