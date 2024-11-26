const Hapi = require('@hapi/hapi');
const tf = require('@tensorflow/tfjs-node'); // TensorFlow.js untuk Node.js
const { v4: uuidv4 } = require('uuid'); // Untuk ID unik
const path = require('path');
const fs = require('fs');
const Joi = require('joi');
const Boom = require('@hapi/boom');
const stream = require('stream');

// Inisialisasi server Hapi
const server = Hapi.server({
    port: process.env.PORT || 8080,
    host: '0.0.0.0',
    routes: {
        payload: {
            maxBytes: 1000000, // Maksimal ukuran gambar 1MB
            parse: true,
            multipart: true,
        }
    }
});

// Memuat model machine learning
let model;

async function loadModel() {
    const modelPath = path.join(__dirname, 'models', 'model.json'); // Pastikan ada model.json
    if (fs.existsSync(modelPath)) {
        model = await tf.loadGraphModel(`file://${modelPath}`);
        console.log('Model berhasil dimuat');
    } else {
        throw new Error('Model file not found');
    }
}

// Menyiapkan server dan endpoint
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
            // Pastikan ada file gambar di payload
            const file = request.payload.image;
            if (!file) {
                return Boom.badRequest('No image file provided');
            }

        
            // Mengolah gambar menjadi tensor
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

            const imageTensor = tf.node.decodeImage(buffer, 3);
            console.log('Ukuran gambar sebelum resize:', imageTensor.shape);

            const resizedImage = tf.image.resizeBilinear(imageTensor, [224, 224]);

            const normalizedImage = resizedImage.sub(0.5).div(0.5);

            const prediction = model.predict(normalizedImage.expandDims(0));
            const predictionData = await prediction.data();

            const cancerProb = predictionData[0]; 
            const nonCancerProb = 1 - cancerProb; 

            console.log(`Probabilitas Cancer: ${cancerProb}`);
            console.log(`Probabilitas Non-cancer: ${nonCancerProb}`);

            let result = 'Non-cancer';
            let suggestion = 'Penyakit kanker tidak terdeteksi.';

            if (cancerProb > nonCancerProb) {
                result = 'Cancer';
                suggestion = 'Segera periksa ke dokter!';
            } else {
                result = 'Non-cancer';
                suggestion = 'Penyakit kanker tidak terdeteksi.';
            }



            console.log(`Hasil Prediksi: ${result}`);


            const response = {
                status: 'success',
                message: 'Model is predicted successfully',
                data: {
                    id: uuidv4(),
                    result: result,
                    suggestion: suggestion,
                    createdAt: new Date().toISOString(),
                },
            };

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

// Menjalankan server
const start = async () => {
    try {
        await loadModel(); // Memuat model
        await server.start();
        console.log('Server berjalan di ' + server.info.uri);
    } catch (err) {
        console.log(err);
    }
};

start();
