import os
import json
from openai import OpenAI
import base64
from PIL import Image
import os
import io
import secrets



# Initialize

client = OpenAI()  # reads OPENAI_API_KEY from the environment

def get_base64_encoded_image(image_file):
    if isinstance(image_file, str):
        img = Image.open(image_file)  # Open image from a path
    else:
        image_file.seek(0)  # Reset file pointer to the beginning if it's a file-like object
        img = Image.open(image_file)  # Open image from a file-like object

    with img:
        # Convert the image to PNG format and store it in a BytesIO buffer
        with io.BytesIO() as buffer:
            img.save(buffer, format="PNG")
            # Get the buffer's content as bytes
            png_data = buffer.getvalue()

    # Encode the PNG data to base64
    base_64_encoded_data = base64.b64encode(png_data)
    base64_string = base_64_encoded_data.decode('utf-8')
    return base64_string

def orc_processor(img_file):

    message_list = [
        {
            "role": 'user',
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                             "data": get_base64_encoded_image(img_file)}},
                {"type": "text", "text": "Transcribe this text. Only output the text and nothing else."}
            ]
        }
    ]

    completion = client.chat.completions.create(
        model='gpt-4o',
        messages=message_list
    )
    return completion.choices[0].message.content.strip().lower()


def compress_image(uploaded_file, base_dir):
    """Compress an in-memory image file if larger than 200 KB and save it to disk.

    Args:
        uploaded_file: The file-like object (from Streamlit file_uploader).
        base_dir (str): Base directory to save the compressed file.
        quality (int): The image quality, on a scale from 1 (worst) to 95 (best). Defaults to 85.
    """
    # Make sure the base directory exists
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # Determine the file size by seeking to the end of the file object
    uploaded_file.seek(0, os.SEEK_END)
    file_size = uploaded_file.tell()
    uploaded_file.seek(0)  # Reset pointer to the start of the file
    if file_size > 200 * 1024:
        quality = 95  # Start with high quality
        while file_size > 200 * 1024 and quality >= 10:
            with Image.open(uploaded_file) as img:
                if img.mode in ("RGBA", "P"):
                    img = img.convert("RGB")

                random_filename = f"{secrets.token_hex(8)}.jpg"
                output_path = os.path.join(base_dir, random_filename)
                img.save(output_path, quality=quality, optimize=True)

                file_size = os.path.getsize(output_path)
                print(f"Trying quality={quality}, Size={file_size} bytes")
                quality -= 10  # Decrease quality to try to meet size requirement

                if file_size > 200 * 1024:
                    # If still not enough, delete the file and try again
                    os.remove(output_path)
                else:
                    print(f"Image compressed and saved to {output_path}, size: {file_size} bytes")
                    return output_path
    else:
        with Image.open(uploaded_file) as img:
            # Convert image to RGB if it is a .png to avoid issues with alpha channels
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")

            # Save image without compression to a random file name
            random_filename = f"{secrets.token_hex(8)}.jpg"
            output_path = os.path.join(base_dir, random_filename)
            img.save(output_path)

            # File was not compressed, report original size
            new_file_size = os.path.getsize(output_path)
            print(f"Image not compressed, saved to {output_path}, size: {new_file_size} bytes")
            return output_path