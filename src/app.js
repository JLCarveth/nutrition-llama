import express from "express";
import {
  LlamaChatSession,
  LlamaContext,
  LlamaJsonSchemaGrammar,
  LlamaModel,
} from "node-llama-cpp";
import sharp from "sharp";
import path from "path";
import { fileURLToPath } from "url";
import multer from "multer";
import Tesseract from "tesseract.js";
import cluster from "node:cluster";
import os from "node:os";
import fs from "node:fs";
import https from "node:https";

const app = express();

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const model = new LlamaModel({
  modelPath: path.join(__dirname, "..", "models", "rocket-3b.Q2_K.gguf"),
  contextSize: 1024, // Reduced context size
  batchSize: 2048, // Increased batch size
});

// Set up multer for file uploads
const upload = multer({ storage: multer.memoryStorage() });

const nutritionSchema = {
  type: "object",
  properties: {
    servingSize: {
      type: "object",
      properties: {
        value: { type: "number" },
        unit: { type: "string", enum: ["g", "ml"] },
      },
      required: ["value", "unit"],
    },
    calories: {
      type: "object",
      properties: {
        value: { type: "number" },
        unit: { type: "string", enum: ["kcal"] },
      },
      required: ["value", "unit"],
    },
    totalFat: {
      type: "object",
      properties: {
        value: { type: "number" },
        unit: { type: "string", enum: ["g"] },
      },
      required: ["value", "unit"],
    },
    carbohydrates: {
      type: "object",
      properties: {
        value: { type: "number" },
        unit: { type: "string", enum: ["g"] },
      },
      required: ["value", "unit"],
    },
    fiber: {
      type: "object",
      properties: {
        value: { type: "number" },
        unit: { type: "string", enum: ["g"] },
      },
      required: ["value", "unit"],
    },
    sugars: {
      type: "object",
      properties: {
        value: { type: "number" },
        unit: { type: "string", enum: ["g"] },
      },
      required: ["value", "unit"],
    },
    protein: {
      type: "object",
      properties: {
        value: { type: "number" },
        unit: { type: "string", enum: ["g"] },
      },
      required: ["value", "unit"],
    },
    cholesterol: {
      type: "object",
      properties: {
        value: { type: "number" },
        unit: { type: "string", enum: ["mg"] },
      },
      required: ["value", "unit"],
    },
    sodium: {
      type: "object",
      properties: {
        value: { type: "number" },
        unit: { type: "string", enum: ["mg"] },
      },
      required: ["value", "unit"],
    },
  },
  required: ["servingSize", "calories", "totalFat", "protein", "carbohydrates"],
};

const grammar = new LlamaJsonSchemaGrammar(nutritionSchema);
const context = new LlamaContext({ model });

// Create Tesseract worker
let worker;
(async () => {
  worker = await Tesseract.createWorker("eng", 1, {
    langPath: "http://tessdata.projectnaptha.com/4.0.0_fast/eng.traineddata.gz",
  });
})();

app.get("/version", (req, res) => {
  return res.json({ version: process.env["npm_package_version"] });
});

app.post("/analyze-nutrition", upload.single("image"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: "No image file uploaded" });
    }

    const logTimings = process.env.LOG_TIMINGS === "true";

    if (logTimings) console.time("Total Request Time");
    if (logTimings) console.time("Image Processing Time");

    // Process the uploaded image
    const image = await sharp(req.file.buffer)
      .resize(800) // Resize for consistency
      .toBuffer();

    if (!image) {
      console.error("No image...");
      return res.status(500).json({ error: "Image processing failed" });
    }

    if (logTimings) console.timeEnd("Image Processing Time");
    if (logTimings) console.time("OCR Time");

    // Use worker.recognize instead of Tesseract.recognize
    const { data: { text } } = await worker.recognize(image, {
      tessedit_pageseg_mode: "6",
      tessedit_ocr_engine_mode: "1", // This is equivalent to OEM_LSTM_ONLY
    });
    console.log(text);

    if (logTimings) console.timeEnd("OCR Time");
    if (logTimings) console.time("Text Truncation Time");

    // Truncate or split the extracted text if it's too long
    const maxTextLength = 1000; // Adjust this value based on your model's capacity
    const truncatedText = text.slice(0, maxTextLength);

    if (logTimings) console.timeEnd("Text Truncation Time");
    if (logTimings) console.time("Model Inference Time");

    // Prepare prompt for the model
    const prompt =
      `Analyze the nutritional facts table in this text and provide the information in a structured format:\n${truncatedText}`;

    const session = new LlamaChatSession({ context });

    // Run inference with structured output
    const result = await session.prompt(prompt, {
      grammar,
      maxTokens: context.getContextSize(),
    });

    console.log(JSON.stringify(result));
    const parsedResult = grammar.parse(result);

    if (logTimings) console.timeEnd("Model Inference Time");
    if (logTimings) console.timeEnd("Total Request Time");

    // Send the structured output as JSON response
    res.json(parsedResult);
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: "An error occurred during analysis" });
  }
});

const PORT = process.env.PORT || 3000;
if (cluster.isPrimary) {
  const numCPUs = os.cpus().length;
  console.log(
    `Master ${process.pid} is running, ${numCPUs} threads available.`,
  );

  // Fork workers.
  for (let i = 0; i < numCPUs; i++) {
    cluster.fork();
  }

  cluster.on("exit", (worker, code, signal) => {
    console.log(
      `Worker ${worker.process.pid} died with code: ${code}, and signal: ${signal}`,
    );
    console.log("Starting a new worker");
    cluster.fork();
  });
} else {
  if (process.env.NODE_ENV === "development") {
    const options = {
      key: fs.readFileSync(process.env.SSL_KEY_PATH),
      cert: fs.readFileSync(process.env.SSL_CERT_PATH),
    };

    https.createServer(options, app).listen(PORT, () => {
      console.log(`Server running on port ${PORT}`);
    });
  } else {
    app.listen(PORT, () => {
      console.log(`Server running on port ${PORT}`);
    });
  }
}

// Properly terminate the Tesseract worker when shutting down
process.on("SIGINT", async () => {
  if (worker) {
    await worker.terminate();
  }
  process.exit(0);
});
