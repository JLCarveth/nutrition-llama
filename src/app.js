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
    calories: {
      type: "number",
    },
    totalFat: { type: "number" },
    carbohydrates: { type: "number" },
    fiber: { type: "number" },
    sugars: { type: "number" },
    protein: { type: "number" },
    cholesterol: { type: "number" },
    sodium: { type: "number" },
  },
  required: ["calories", "totalFat", "protein", "carbohydrates"],
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
    const prompt = `Analyze the nutritional facts table in this text and provide the information in a structured format:\n${truncatedText}`;

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
app.listen(PORT, () => console.log(`Now listening on ${PORT}`));
