# { INSERT PROJECT NAME }

This project is a backend service that analyzes nutritional information tables from images using OCR (Optical Character Recognition) and a language model. The service extracts nutritional data and returns it in a structured JSON format.

## Features

- **OCR Integration**: Uses Tesseract.js to extract text from images.
- **Language Model**: Utilizes `node-llama-cpp` to analyze and structure the extracted text.
- **Multi-threading**: Leverages Node.js clustering to run the backend on multiple processes for improved performance.
- **JSON Schema**: Defines a structured output format for nutritional information.

## Prerequisites

- Node.js (v14.x or higher)
- npm (v6.x or higher)

## Installation

1. **Clone the repository:**

   ```sh
   git clone https://github.com/yourusername/nutrition-analyzer-backend.git
   cd nutrition-analyzer-backend
