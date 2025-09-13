# backend/steganography/lsb.py

from PIL import Image
from cryptography.fernet import InvalidToken
# Import our new utility functions using a relative import
from .utils import bytes_to_bin, bin_to_bytes, bin_to_text, encrypt_data, decrypt_data

def encode_text_in_image(text: str, input_image_path: str, output_image_path: str, password: str = None):
    """
    Encodes text into an image using LSB steganography, with an optional encryption layer.
    """
    print("LSB_TEXT: Starting encoding process.")
    
    payload_to_embed_bytes = text.encode('utf-8')
    salt_len_bytes = 0
    
    if password:
        print("LSB_TEXT: Password provided. Encrypting data.")
        encrypted_data_bytes, salt = encrypt_data(payload_to_embed_bytes, password)
        payload_to_embed_bytes = salt + encrypted_data_bytes
        salt_len_bytes = len(salt)
    
    binary_payload = bytes_to_bin(payload_to_embed_bytes)
    
    header_bits = ('1' if password else '0') + format(salt_len_bytes, '08b')
    full_binary_payload = header_bits + binary_payload
    
    delimiter = '1111111111111110'
    data_to_hide = full_binary_payload + delimiter

    print(f"LSB_TEXT: Total bits to hide (including header and delimiter): {len(data_to_hide)}")

    try:
        with Image.open(input_image_path) as img:
            if img.mode not in ['RGB', 'RGBA']:
                img = img.convert('RGBA')

            max_bits = len(img.getdata()) * 3
            if len(data_to_hide) > max_bits:
                raise ValueError(f"Payload is too large for this image. Required: {len(data_to_hide)} bits, Available: {max_bits} bits.")

            data_index = 0
            new_pixels = []

            for pixel in img.getdata():
                r, g, b = pixel[0], pixel[1], pixel[2]
                alpha = pixel[3] if len(pixel) == 4 else 255

                if data_index < len(data_to_hide):
                    r = (r & ~1) | int(data_to_hide[data_index]); data_index += 1
                if data_index < len(data_to_hide):
                    g = (g & ~1) | int(data_to_hide[data_index]); data_index += 1
                if data_index < len(data_to_hide):
                    b = (b & ~1) | int(data_to_hide[data_index]); data_index += 1
                
                new_pixels.append((r, g, b, alpha))

            img.putdata(new_pixels)
            img.save(output_image_path, 'PNG')
            
            # --- THIS IS THE FIX ---
            # Explicitly close the image to release the file lock
            img.close()
            # ---------------------

            print("LSB_TEXT: Encoding successful. Image saved.")
    except Exception as e:
        print(f"LSB_TEXT: An error occurred: {e}")
        raise e


def decode_image_to_text(input_image_path: str, password: str = None) -> str:
    """
    Decodes text from an LSB-encoded image, with an optional decryption layer.
    """
    print("LSB_TEXT_DECODE: Starting decoding process.")
    try:
        with Image.open(input_image_path) as img:
            if img.mode not in ['RGB', 'RGBA']:
                img = img.convert('RGBA')

            binary_bits = []
            delimiter = '1111111111111110'

            for pixel in img.getdata():
                r, g, b = pixel[0], pixel[1], pixel[2]
                binary_bits.extend([str(r & 1), str(g & 1), str(b & 1)])

                if "".join(binary_bits[-len(delimiter):]) == delimiter:
                    print("LSB_TEXT_DECODE: Delimiter found.")
                    break
            else:
                raise ValueError("Delimiter not found. The image may not contain a message or is corrupted.")

            full_binary_data = "".join(binary_bits)
            payload_with_header = full_binary_data[:full_binary_data.rfind(delimiter)]
            
            header_len = 9
            if len(payload_with_header) < header_len:
                raise ValueError("Invalid data: payload is too short to contain a header.")

            header = payload_with_header[:header_len]
            is_encrypted = header[0] == '1'
            salt_len_bytes = int(header[1:], 2)
            
            payload_binary = payload_with_header[header_len:]
            
            if is_encrypted:
                if not password:
                    raise ValueError("Message is encrypted, but no password was provided.")
                
                payload_bytes = bin_to_bytes(payload_binary)
                if len(payload_bytes) < salt_len_bytes:
                    raise ValueError("Invalid encrypted data: payload is smaller than salt length.")
                
                salt = payload_bytes[:salt_len_bytes]
                encrypted_data = payload_bytes[salt_len_bytes:]
                
                try:
                    decrypted_bytes = decrypt_data(encrypted_data, password, salt)
                    return decrypted_bytes.decode('utf-8')
                except InvalidToken:
                    raise ValueError("Decryption failed. Incorrect password or corrupted data.")
            else:
                return bin_to_text(payload_binary)

    except Exception as e:
        print(f"LSB_TEXT_DECODE: An error occurred: {e}")
        raise e