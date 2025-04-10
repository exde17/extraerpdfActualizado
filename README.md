# PDF to Text Converter API

This is a FastAPI-based service that converts PDF files to text files, preserving the content including tables.

## Features

- Upload PDF files via a REST API
- Extract all text content from PDFs
- Detect and convert tables to text format with proper tabulation
- Download the resulting text file

## Requirements

- Python 3.7+
- Dependencies listed in `requirements.txt`

## Installation

1. Clone this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. For table extraction, you'll need Java installed (required by tabula-py)

## Usage

1. Start the server:

```bash
python main.py
```

Or using uvicorn directly:

```bash
uvicorn main:app --reload
```

2. The server will start at http://localhost:8000

3. Use the API:
   - Access the interactive API documentation at http://localhost:8000/docs
   - Upload a PDF file using the `/convert` endpoint
   - The API will return a text file for download

## API Endpoints

- `GET /`: Welcome message
- `POST /convert`: Upload a PDF file and receive a text file in response

## How It Works

1. The uploaded PDF is saved temporarily
2. Text is extracted using PyPDF2
3. Tables are detected and extracted using tabula-py
4. Tables are formatted with proper spacing using tabs
5. All content is combined into a single text file
6. The text file is returned for download
7. Temporary files are cleaned up automatically
