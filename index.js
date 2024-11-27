const Hapi = require('@hapi/hapi');
const tf = require('@tensorflow/tfjs-node'); // TensorFlow.js untuk Node.js
const { v4: uuidv4 } = require('uuid'); // Untuk ID unik
const path = require('path');
const fs = require('fs');
const Joi = require('joi');
const Boom = require('@hapi/boom');
const stream = require('stream');
const { Firestore } = require("@google-cloud/firestore");  // Baris 8

const server = Hapi.server({
    port: process.env.PORT || 8080,
    host: process.env.HOST || '0.0.0.0',
    routes: {
        cors: {
            origin: ['*'],
        },
    },
});

const firestore = new Firestore({
    projectId: 'spry-water-442811-c5'
});

// Memuat model machine learning
const loadModel = async () => {
    try {
        const modelUrl = 'https://storage.googleapis.com/submissionmlgc-esibutarbutar-bucket/models/model.json';
        let model = await tf.loadGraphModel(modelUrl);
        console.log('Model loaded successfully');
    } catch (error) {
        console.error('Error loading model:', error.message);
        process.exit(1);
    }
};

// Rute untuk prediksi
server.route({
    method: 'POST',
    path: '/predict',
    options: {
        payload: {
            maxBytes: 1000000, // Max 1MB
            parse: true,
            multipart: true,
            output: 'stream',
        },
        validate: {
            payload: Joi.object({
                image: Joi.object().required(),
            }),
        },
    },
    handler: async (request, h) => {
        try {
            const file = request.payload.image;
            if (!file) {
                return Boom.badRequest('No image file provided');
            }

            const fileSize = file.bytes;
            if (fileSize > 1000000) {  // Jika ukuran file lebih dari 1MB
                console.log('File size exceeds 1MB');
                return Boom.payloadTooLarge('Payload content length greater than maximum allowed: 1000000');
            }

            // Verifikasi format file gambar (misalnya .jpg, .png)
            const fileExtension = path.extname(file.hapi.filename).toLowerCase();
            if (!['.jpg', '.jpeg', '.png'].includes(fileExtension)) {
                console.log('Invalid image format');
                return Boom.badRequest('Invalid image format. Only JPG, JPEG, and PNG are allowed.');
            }
            // Membaca file gambar yang diupload
            const buffer = await new Promise((resolve, reject) => {
                const chunks = [];
                file.pipe(new stream.Writable({
                    write(chunk, encoding, callback) {
                        chunks.push(chunk);
                        callback();
                    },
                    final() {
                        resolve(Buffer.concat(chunks));
                    },
                }));
            });

            // Mengubah buffer gambar menjadi tensor
            const imageTensor = tf.node.decodeImage(buffer, 3);
            console.log('Ukuran gambar sebelum resize:', imageTensor.shape);

            // Melakukan resize gambar agar sesuai dengan input model
            const resizedImage = tf.image.resizeBilinear(imageTensor, [224, 224]);

            // Normalisasi gambar
            const normalizedImage = resizedImage.sub(0.5).div(0.5);

            // Melakukan prediksi
            const prediction = model.predict(normalizedImage.expandDims(0));
            const predictionData = await prediction.data();

            // Probabilitas untuk kelas kanker dan non-kanker
            const cancerProb = predictionData[0];
            const nonCancerProb = 1 - cancerProb;

            console.log(`Probabilitas Cancer: ${cancerProb}`);
            console.log(`Probabilitas Non-cancer: ${nonCancerProb}`);

            // Menentukan hasil prediksi
            let result = 'Non-cancer';
            let suggestion = 'Penyakit kanker tidak terdeteksi.';

            if (cancerProb > nonCancerProb) {
                result = 'Cancer';
                suggestion = 'Segera periksa ke dokter!';
            } else {
                result = 'Non-cancer';
                suggestion = 'Penyakit kanker tidak terdeteksi.';
            }
            // Membuat data hasil prediksi
            const data = {
                id: uuidv4(),
                result: result,
                suggestion: suggestion,
                createdAt: new Date().toISOString(),
            };

            // Mengembalikan response
            const response = {
                status: 'success',
                message: 'Model is predicted successfully',
                data: data,
            };

            console.log('Response to frontend:', response);
            return h.response(response).code(200);
        } catch (error) {
            console.error('Error during prediction:', error);
            return h.response({
                status: 'fail',
                message: 'Terjadi kesalahan dalam melakukan prediksi',
            }).code(400);
        }
    }
});
// Endpoint riwayat prediksi
server.route({
    method: 'GET',
    path: '/predict/histories',
    handler: async (request, h) => {
        try {
            const collectionRef = firestore.collection('predictions');
            const snapshot = await collectionRef.get();

            if (snapshot.empty) {
                return h.response({
                    status: 'success',
                    data: [],
                }).code(200);
            }

            const histories = [];
            snapshot.forEach(doc => {
                histories.push({
                    id: doc.id,
                    history: doc.data(),
                });
            });

            return h.response({
                status: 'success',
                data: histories,
            }).code(200);
        } catch (error) {
            console.error('Error fetching prediction histories:', error);
            return h.response({
                status: 'fail',
                message: 'Terjadi kesalahan dalam mengambil riwayat prediksi',
            }).code(500);
        }
    },
});

const start = async () => {
    try {
        await loadModel();  // Memuat model ML sebelum memulai server
        await server.start();
        console.log('Server berjalan di ' + server.info.uri);
    } catch (err) {
        console.log(err);
    }
};

start();
