import streamlit as st
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

def load_model_and_processor():
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct", torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    return model, processor

st.title('Image OCR and RAG')

with st.sidebar:
    st.header("Upload your image")
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        st.success("Image uploaded successfully!")

model, processor = load_model_and_processor()

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {"type": "text", "text": "Extract all the text present in the image and give the output in JSON format"},
                ],
            }
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Generate output using the model
        generated_ids = model.generate(**inputs, max_new_tokens=260)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        # Display the extracted text in JSON format
        st.subheader("Extracted Text in JSON Format:")
        st.markdown(output_text[0])
        
        keyword = st.text_input("Enter the keyword to search for : ")
        if keyword:
            keyword_messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image,
                        },
                        {"type": "text", "text": f"Extract text from the image that closely matches to {keyword}"},
                    ],
                }
            ]
            keyword_search = processor.apply_chat_template(keyword_messages, tokenize=False, add_generation_prompt=True)
            keyword_inputs = processor(
                text=[keyword_search],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            keyword_inputs = keyword_inputs.to("cuda")
        
            generated_ids = model.generate(**keyword_inputs, max_new_tokens=260)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            st.text(output_text[0])

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

else:
    st.write("Please upload an image from the sidebar")
