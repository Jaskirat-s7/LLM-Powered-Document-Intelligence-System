# Multimodal LLM-Powered Document Intelligence System

## Project Overview
This project implements an end-to-end RAG (Retrieval-Augmented Generation) pipeline using LangChain, OpenAI, and Streamlit. It is designed to process both text and image-based documents, providing a comprehensive document intelligence solution. The system architecture leverages local vector storage (FAISS) and multimodal LLMs (GPT-4o) for high-accuracy retrieval and question answering.

## Features

- **Multimodal Support**: Upload and query PDFs containing both text and images.
- **Image Intelligence**: Uses OpenAI's GPT-4o Vision API to understand and summarize charts, graphs, and images within documents.
- **Semantic Search**: FAISS vector store for fast and accurate retrieval.
- **Interactive UI**: Streamlit-based chat interface.

## Prerequisites

- Python 3.8+
- [Poppler](https://github.com/OSGeo/OSGeo4W/blob/master/src/poppler/text/poppler.txt) (Required for `pdf2image`)
    - **Mac**: `brew install poppler`
    - **Windows**: Download the latest binary and add `bin/` to your PATH.
    - **Linux**: `sudo apt-get install poppler-utils`

## Setup

1. **Clone the repository**
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Environment Setup**
   Copy `.env.example` to `.env` and add your OpenAI API Key.
   ```bash
   cp .env.example .env
   ```
   Edit `.env`:
   ```
   OPENAI_API_KEY=sk-...
   ```

## Usage

Run the Streamlit application:
```bash
streamlit run app.py
```
