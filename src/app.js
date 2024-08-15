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
  modelPath: path.join(__dirname, "..", "models", "llama-2-7b-chat.Q4_0.gguf"),
  contextSize: 2048, // Reduced context size
  batchSize: 1024, // Increased batch size
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

app.post("/analyze-nutrition", upload.single("image"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: "No image file uploaded" });
    }

    // Process the uploaded image
    const image = await sharp(req.file.buffer)
      .resize(800) // Resize for consistency
      .toBuffer();

    if (!image) console.error(`No image...`);

    // Use Tesseract.js to extract text from the image
    const { data: { text } } = await Tesseract.recognize(image, "eng");
    console.log(text);

    // Truncate or split the extracted text if it's too long
    const maxTextLength = 1000; // Adjust this value based on your model's capacity
    const truncatedText = text.slice(0, maxTextLength);

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
      key: fs.readFileSync("localhost-key.pem"),
      cert: fs.readFileSync("localhost.pem"),
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
