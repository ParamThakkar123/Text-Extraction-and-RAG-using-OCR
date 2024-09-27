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
```

## Run the application
in the command line enter : 

```bash
streamlit run app.py
```
## Screenshots

![image](https://github.com/user-attachments/assets/8f31d872-8b47-4d85-9c62-0ba902b9d149)

![image](https://github.com/user-attachments/assets/6485ea99-aee7-4241-a8f6-a814ab4f8d9b)

![image](https://github.com/user-attachments/assets/d6b1a64d-10ed-4d72-9d37-01729c2035f9)

Hindi : 

![image](https://github.com/user-attachments/assets/1ae38c4e-50ee-489f-901a-91ca23a43b38)

JSON output : 
{
  "text": "चलने वाले पैरों में कितना फर्क होता है एक आगे तो एक पीछे लेकिन न कभी आगे वाले को अभिमान होता है, और न ही पीछे वाले का अपमान क्योंकि उन्हें पता होता है कि कुछ ही समय में स्थिति बदलने वाली है इसी को जीवन कहते हैं,",
  "author": "RPSharma"
}
