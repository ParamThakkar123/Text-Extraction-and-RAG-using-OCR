# Text-Extraction-and-RAG-using-OCR

This project focuses on extracting text from images using Optical Character Recognition (OCR) and leveraging a Retrieval-Augmented Generation (RAG) model for searching relevant sentences related to a keyword. The model utilized in this application is **Qwen2VL-2B-Instruct**.

## Dependencies

The following libraries are required for this project:

- `transformers`
- `qwen_vl_utils`
- `pillow`
- `streamlit`
- `flash-attn`

## Setup Instructions

To set up the environment and install the necessary dependencies, run the following commands:

```bash
pip install -q git+https://github.com/huggingface/transformers.git qwen-vl-utils flash-attn
pip install streamlit -q

## Run the application
in the command line enter : 

```bash
streamlit run app.py
