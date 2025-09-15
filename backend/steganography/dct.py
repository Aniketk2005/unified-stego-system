from PIL import Image
import numpy as np
import io
from scipy.fftpack import dct, idct
from cryptography.fernet import InvalidToken

from .utils import (
    bytes_to_bin, bin_to_bytes, 
    text_to_bin, bin_to_text,
    encrypt_data, decrypt_data
)

# --- HELPER FUNCTIONS (REWRITTEN FOR ROBUSTNESS) ---

def _image_to_binary(image_file_bytes: bytes) -> str:
    """Converts a secret image's bytes into a binary string with a robust metadata header."""
    with Image.open(io.BytesIO(image_file_bytes)) as img:
        img = img.convert("RGB")
        width, height = img.size
        
        metadata = f"{width},{height}"
        binary_metadata = text_to_bin(metadata)
        
        metadata_len_header = format(len(binary_metadata), '016b')
        
        pixel_data = np.array(img).flatten().tobytes()
        binary_pixel_data = bytes_to_bin(pixel_data)
        
        full_payload = metadata_len_header + binary_metadata + binary_pixel_data
        print(f"DCT: Secret image converted to {len(full_payload)} bits.")
        return full_payload

def _binary_to_image(binary_data: str, output_path: str):
    """Reconstructs an image from a binary string using the robust metadata header."""
    try:
        if len(binary_data) < 16:
            raise ValueError("Data is too short to contain a metadata header.")
        metadata_len = int(binary_data[:16], 2)
        
        metadata_end_index = 16 + metadata_len
        if len(binary_data) < metadata_end_index:
            raise ValueError("Data is too short to contain the full metadata.")
            
        metadata_bin = binary_data[16:metadata_end_index]
        pixel_data_bin = binary_data[metadata_end_index:]
        
        metadata = bin_to_text(metadata_bin)
        width, height = map(int, metadata.split(','))
        
        pixel_bytes = bin_to_bytes(pixel_data_bin)
        pixels = np.frombuffer(pixel_bytes, dtype=np.uint8)
        
        expected_pixels = width * height * 3
        if len(pixels) < expected_pixels:
             raise ValueError(f"Not enough pixel data to reconstruct image.")
        
        pixels = pixels[:expected_pixels].reshape((height, width, 3))
        reconstructed_img = Image.fromarray(pixels, 'RGB')
        reconstructed_img.save(output_path, 'PNG')
    except Exception as e:
        raise ValueError(f"Failed to reconstruct image: {e}")

# --- MAIN ENCODING/DECODING FUNCTIONS ---

def encode_image_in_image(secret_image_bytes: bytes, cover_image_path: str, output_path: str, password: str = None):
    print("DCT_IMAGE: Starting DCT encoding.")
    
    with Image.open(cover_image_path) as cover_img:
        cover_img = cover_img.convert('RGB')
        cover_width, cover_height = cover_img.size
        
        secret_binary_payload = _image_to_binary(secret_image_bytes)
        payload_to_embed_bytes = bin_to_bytes(secret_binary_payload)
        
        salt_len_bytes = 0
        if password:
            encrypted_data_bytes, salt = encrypt_data(payload_to_embed_bytes, password)
            payload_to_embed_bytes = salt + encrypted_data_bytes
            salt_len_bytes = len(salt)
        
        binary_to_embed = bytes_to_bin(payload_to_embed_bytes)
        header = ('1' if password else '0') + format(salt_len_bytes, '08b') + format(len(binary_to_embed), '032b')
        data_to_hide = header + binary_to_embed

        capacity = (cover_width // 8) * (cover_height // 8) * 3 * 15 
        if len(data_to_hide) > capacity:
            raise ValueError(f"Secret image is too large for this cover image.")

        img_array = np.array(cover_img, dtype=np.float32) - 128.0
        data_index = 0

        # --- START OF CRITICAL FIX ---
        for i in range(0, cover_height - (cover_height % 8), 8):
            for j in range(0, cover_width - (cover_width % 8), 8):
                for channel in range(3):
                    if data_index >= len(data_to_hide): break
                    block = img_array[i:i+8, j:j+8, channel]
                    dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
                    
                    # Flatten and modify the coefficients
                    flat_coeffs = dct_block.flatten()
                    for k in range(10, 25):
                        if data_index < len(data_to_hide):
                            coeff = round(flat_coeffs[k])
                            bit_to_hide = int(data_to_hide[data_index])
                            flat_coeffs[k] = (coeff & ~1) | bit_to_hide
                            data_index += 1
                        else:
                            break
                    
                    # Reshape the modified coefficients back into an 8x8 block
                    modified_dct_block = flat_coeffs.reshape((8, 8))
                    
                    # Apply the inverse DCT to the modified block
                    img_array[i:i+8, j:j+8, channel] = idct(idct(modified_dct_block.T, norm='ortho').T, norm='ortho')
                if data_index >= len(data_to_hide): break
            if data_index >= len(data_to_hide): break
        # --- END OF CRITICAL FIX ---
        
        stego_img_array = np.clip(img_array + 128.0, 0, 255).astype(np.uint8)
        stego_img = Image.fromarray(stego_img_array, 'RGB')
        stego_img.save(output_path, 'PNG')
        stego_img.close()
        print("DCT_IMAGE: Encoding successful.")


def decode_image_from_image(stego_image_path: str, output_path: str, password: str = None):
    print("DCT_IMAGE_DECODE: Starting DCT decoding.")
    
    with Image.open(stego_image_path) as stego_img:
        stego_img = stego_img.convert('RGB')
        width, height = stego_img.size
        img_array = np.array(stego_img, dtype=np.float32) - 128.0
        
        extracted_bits = []
        for i in range(0, height - (height % 8), 8):
            for j in range(0, width - (width % 8), 8):
                for channel in range(3):
                    block = img_array[i:i+8, j:j+8, channel]
                    dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
                    flat_coeffs = dct_block.flatten()
                    for k in range(10, 25):
                        extracted_bits.append(str(round(abs(flat_coeffs[k])) % 2))

        extracted_binary = "".join(extracted_bits)
        
        header_len = 1 + 8 + 32
        if len(extracted_binary) < header_len:
            raise ValueError("Not a valid stego image: could not extract a full header.")
        
        is_encrypted = extracted_binary[0] == '1'
        salt_len_bytes = int(extracted_binary[1:9], 2)
        payload_len_bits = int(extracted_binary[9:header_len], 2)
        
        total_bits_needed = header_len + payload_len_bits
        if len(extracted_binary) < total_bits_needed:
            raise ValueError("Data is corrupted or incomplete. Could not extract full payload.")
            
        payload_binary = extracted_binary[header_len:total_bits_needed]
        payload_bytes = bin_to_bytes(payload_binary)
        
        if is_encrypted:
            if not password:
                raise ValueError("Image is encrypted, but no password was provided.")
            if len(payload_bytes) < salt_len_bytes:
                 raise ValueError("Invalid encrypted data.")
            try:
                salt = payload_bytes[:salt_len_bytes]
                encrypted_data = payload_bytes[salt_len_bytes:]
                decrypted_payload_bytes = decrypt_data(encrypted_data, password, salt)
                final_binary_payload = bytes_to_bin(decrypted_payload_bytes)
            except InvalidToken:
                raise ValueError("Decryption failed. Incorrect password.")
        else:
            final_binary_payload = payload_binary

        _binary_to_image(final_binary_payload, output_path)

