# Nutrition Facts Analyzer

## Overview

The **Nutrition Facts Analyzer** is a Node.js application designed to extract and analyze nutritional information from images of nutrition labels. Using `Tesseract.js` for Optical Character Recognition (OCR) and the `node-llama-cpp` package for structured data inference, this tool can identify key nutritional data and present it in a JSON format. The application leverages a custom schema for nutrition facts and can run on multiple CPU cores for improved performance.

## Features

- **Image Processing:** Automatically resizes and processes images to optimize OCR accuracy.
- **OCR Integration:** Utilizes `Tesseract.js` for reliable text extraction from nutrition labels.
- **Structured Data Inference:** Employs `node-llama-cpp` to extract structured nutritional data in accordance with a predefined schema.
- **Multi-Threading:** Supports clustering with multiple CPU cores to handle concurrent requests efficiently.
- **HTTPS Support:** Secure server setup for production environments.

## Installation

### Prerequisites

- **Node.js** (version 16 or later)
- **npm** (Node Package Manager)

### Setting Up the Project

1. **Clone the Repository:**

   ```bash
   git clone <your-repository-url>
   cd nutrition-facts-analyzer

2. **Install Dependencies:**
    ```bash
    npm install
    ```

3. **Download a Model:**
Models can be downloaded from Huggingface using the `ipull` command:
    ```bash
    npx ipull <model-download-link>
    ```
    The code by default uses the [rocket-3b GGUF](https://huggingface.co/TheBloke/rocket-3B-GGUF/) model.

4. **Set Up Environment Variables:**
Create a `.env` file in the project root and configure the necessary variables:
    ```bash
    PORT=3000
    NODE_ENV=development
    SSL_KEY_PATH=/path/to/key.pem
    SSL_CERT_PATH=/path/to/cert.pem
    ```

## Usage
1. Start the Server:
    ```
    npm run dev
    ```

2. Upload and Analyze an Image:
    Use an HTTP client like cURL or [Bruno](https://github.com/usebruno/bruno) to POST an `image` file to the `/analyze-nutrition` route.

    Example `curl` command:
    ```bash
    curl -X POST -F "image=@/path/to/image.jpg" http://localhost:3000/analyze-nutrition
    ```
    The server will return a JSON response with the extracted and structured nutritional data. 

## Acknowledgments
Tesseract.js for OCR processing.  

node-llama-cpp (and llama-cpp, of course!) for integrating LLaMA-based models.
