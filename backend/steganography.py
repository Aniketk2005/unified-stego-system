from PIL import Image
import os
import io
import numpy as np
from scipy.fftpack import dct, idct # For Discrete Cosine Transform

# For Encryption/Decryption
from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import base64

# --- Encryption/Decryption Utilities ---

def _derive_key(password: str, salt: bytes) -> bytes:
    """Derives a Fernet key from a password and salt."""
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000, # Recommended iteration count
        backend=default_backend()
    )
    return base64.urlsafe_b64encode(kdf.derive(password.encode()))

def encrypt_data(data: bytes, password: str) -> tuple[bytes, bytes]:
    """Encrypts data using a password. Returns (encrypted_data, salt)."""
    print("CRYPTO: Encrypting data...")
    salt = os.urandom(16) # Generate a new salt for each encryption (16 bytes)
    key = _derive_key(password, salt)
    f = Fernet(key)
    encrypted_data = f.encrypt(data)
    print(f"CRYPTO: Data encrypted. Encrypted size: {len(encrypted_data)} bytes.")
    return encrypted_data, salt

def decrypt_data(encrypted_data: bytes, password: str, salt: bytes) -> bytes:
    """Decrypts data using a password and salt. Raises InvalidToken if password is wrong."""
    print("CRYPTO: Decrypting data...")
    key = _derive_key(password, salt)
    f = Fernet(key)
    decrypted_data = f.decrypt(encrypted_data) # This will raise an InvalidToken error if password is wrong
    print(f"CRYPTO: Data decrypted.")
    return decrypted_data

# --- Binary/Bytes Conversion Utilities ---

def text_to_bin(text: str) -> str:
    """Convert string to binary string."""
    return ''.join(format(ord(c), '08b') for c in text)

def bin_to_text(binary_string: str) -> str:
    """Convert binary string to text."""
    text = ""
    for i in range(0, len(binary_string), 8):
        byte = binary_string[i:i+8]
        if len(byte) == 8:
            text += chr(int(byte, 2))
        else:
            print(f"WARNING: Incomplete byte found at end of binary string in bin_to_text: {byte}")
    return text

def bytes_to_bin(data_bytes: bytes) -> str:
    """Convert bytes to binary string."""
    return ''.join(format(byte, '08b') for byte in data_bytes)

def bin_to_bytes(binary_string: str) -> bytes:
    """Convert binary string to bytes."""
    # Pad binary string to be a multiple of 8 bits
    padded_binary_string = binary_string + '0' * ((8 - len(binary_string) % 8) % 8)
    return int(padded_binary_string, 2).to_bytes(len(padded_binary_string) // 8, byteorder='big')

# --- Image Utilities ---

def image_to_bin(image_file):
    """
    Converts a PIL Image object (secret image) into a binary string.
    Includes image dimensions and mode in the binary data for reconstruction.
    
    Args:
        image_file: A file-like object or path to the image to be converted.
    Returns:
        tuple: (binary_string of image data, original image mode, original image size)
    """
    print("LSB/DCT: Converting secret image to binary.")
    try:
        # Use with statement for safe file handling
        if isinstance(image_file, str):
            with Image.open(image_file) as secret_img:
                secret_img.load() # Load pixels into memory
                original_mode = secret_img.mode
                original_size = secret_img.size
                
                if secret_img.mode == 'RGBA':
                    secret_img = secret_img.convert('RGB')
                elif secret_img.mode == 'P' or secret_img.mode == 'L':
                    secret_img = secret_img.convert('RGB')
                
                binary_bits = []
                for pixel in secret_img.getdata():
                    for value in pixel[:3]: # Only consider R, G, B for embedding
                        binary_bits.append(format(value, '08b'))
                binary_data = "".join(binary_bits)
        else: # Assume it's a file-like object (e.g., from request.files)
            with Image.open(io.BytesIO(image_file.read())) as secret_img:
                secret_img.load() # Load pixels into memory
                original_mode = secret_img.mode
                original_size = secret_img.size
                
                if secret_img.mode == 'RGBA':
                    secret_img = secret_img.convert('RGB')
                elif secret_img.mode == 'P' or secret_img.mode == 'L':
                    secret_img = secret_img.convert('RGB')
                
                binary_bits = []
                for pixel in secret_img.getdata():
                    for value in pixel[:3]: # Only consider R, G, B for embedding
                        binary_bits.append(format(value, '08b'))
                binary_data = "".join(binary_bits)
        
        print(f"LSB/DCT: Secret image original mode: {original_mode}, size: {original_size}")
        print(f"LSB/DCT: Secret image converted to {len(binary_data)} bits.")
        return binary_data, original_mode, original_size
    except Exception as e:
        print(f"LSB/DCT ERROR: Failed to convert secret image to binary: {e}")
        raise Exception(f"Failed to process secret image: {e}")

def bin_to_image(binary_data, output_path, original_size, original_mode):
    """
    Converts a binary string back into an image based on original dimensions and mode.
    
    Args:
        binary_data (str): The binary string representing image pixel data.
        output_path (str): Path to save the reconstructed image.
        original_size (tuple): (width, height) of the original image.
        original_mode (str): Mode ('RGB', 'RGBA', etc.) of the original image.
    """
    print(f"LSB/DCT: Reconstructing image from {len(binary_data)} bits.")
    
    pixels = []
    channels_per_pixel = 3 # Always assume 3 channels (RGB) for the embedded data

    for i in range(0, len(binary_data), channels_per_pixel * 8):
        pixel_values = []
        for j in range(channels_per_pixel):
            byte_start = i + (j * 8)
            byte_end = byte_start + 8
            if byte_end <= len(binary_data):
                byte_str = binary_data[byte_start:byte_end]
                pixel_values.append(int(byte_str, 2))
            else:
                print("LSB/DCT WARNING: Incomplete pixel data during reconstruction. Truncating.")
                break
        if len(pixel_values) == channels_per_pixel:
            pixels.append(tuple(pixel_values))
        else:
            break

    try:
        reconstructed_img = Image.new('RGB', original_size)
        reconstructed_img.putdata(pixels)
        
        if original_mode == 'RGBA':
            final_img = Image.new('RGBA', original_size)
            final_img.paste(reconstructed_img, (0,0))
            reconstructed_img = final_img
            
        reconstructed_img.save(output_path)
        print(f"LSB/DCT: Reconstructed image saved to {output_path}")
    except Exception as e:
        print(f"LSB/DCT ERROR: Failed to save reconstructed image: {e}")
        raise Exception(f"Failed to save reconstructed image: {e}")


# --- LSB Encoding/Decoding Algorithms ---

def encode_text_in_image(text: str, input_image_path: str, output_image_path: str, password: str = None):
    """
    Encodes text into image using LSB steganography.
    """
    print(f"LSB_TEXT: Starting encoding process for {input_image_path}")
    
    # --- Encryption Layer ---
    original_data_bytes = text.encode('utf-8')
    payload_to_embed_bytes = original_data_bytes
    
    salt_len_bytes = 0 # No salt if not encrypted
    if password:
        encrypted_data_bytes, salt = encrypt_data(original_data_bytes, password)
        payload_to_embed_bytes = salt + encrypted_data_bytes
        salt_len_bytes = len(salt)
        print(f"LSB_TEXT: Encrypted data bytes length: {len(encrypted_data_bytes)} bytes.")
    
    binary_payload = bytes_to_bin(payload_to_embed_bytes)
    # Prepend a header indicating if encrypted and salt length (for LSB text, simpler header)
    # Header: 1 bit for encrypted (1=yes, 0=no) + 8 bits for salt length in bytes (if encrypted)
    header_bits = '1' if password else '0' # Is encrypted?
    header_bits += format(salt_len_bytes, '08b') # Salt length in bytes (8 bits)
    
    full_binary_payload = header_bits + binary_payload
    print(f"LSB_TEXT: Full binary payload (with header): {len(full_binary_payload)} bits.")
    # --- End Encryption Layer ---

    delimiter = '1111111111111110'
    full_binary_payload_with_delimiter = full_binary_payload + delimiter
    print(f"LSB_TEXT: Binary payload length (with delimiter): {len(full_binary_payload_with_delimiter)} bits")

    try:
        with Image.open(input_image_path) as img: # Use with statement
            print(f"LSB_TEXT: Cover image opened successfully. Mode: {img.mode}, Size: {img.size}")

            if img.mode not in ['RGB', 'RGBA']:
                print(f"LSB_TEXT: Converting cover image from {img.mode} to RGBA")
                img = img.convert('RGBA')

            max_bits = len(img.getdata()) * 3
            print(f"LSB_TEXT: Image capacity: {max_bits} bits")

            if len(full_binary_payload_with_delimiter) > max_bits:
                raise ValueError(f"Payload too long to encode in this image. Required: {len(full_binary_payload_with_delimiter)} bits, Available: {max_bits} bits.")

            data_index = 0
            img_data = list(img.getdata())
            new_pixels = []

            for pixel in img_data:
                r, g, b = pixel[0], pixel[1], pixel[2]
                alpha = pixel[3] if len(pixel) == 4 else None

                if data_index < len(full_binary_payload_with_delimiter):
                    r = (r & ~1) | int(full_binary_payload_with_delimiter[data_index])
                    data_index += 1
                if data_index < len(full_binary_payload_with_delimiter):
                    g = (g & ~1) | int(full_binary_payload_with_delimiter[data_index])
                    data_index += 1
                if data_index < len(full_binary_payload_with_delimiter):
                    b = (b & ~1) | int(full_binary_payload_with_delimiter[data_index])
                    data_index += 1
                
                if alpha is not None:
                    new_pixels.append((r, g, b, alpha))
                else:
                    new_pixels.append((r, g, b))

            img.putdata(new_pixels)

            print(f"LSB_TEXT: Saving encoded image to {output_image_path}")
            img.save(output_image_path)
            print("LSB_TEXT: Image saved successfully.")
    except FileNotFoundError:
        raise FileNotFoundError(f"Cover image not found: {input_image_path}")
    except Exception as e:
        raise Exception(f"Failed to open/process cover image: {e}")


def decode_image_to_text(input_image_path: str, password: str = None):
    """
    Decodes text from an LSB-encoded image.
    """
    print(f"LSB_TEXT_DECODE: Starting decoding process for {input_image_path}")
    try:
        with Image.open(input_image_path) as img: # Use with statement
            print(f"LSB_TEXT_DECODE: Image opened successfully for decoding. Mode: {img.mode}, Size: {img.size}")

            if img.mode not in ['RGB', 'RGBA']:
                print(f"LSB_TEXT_DECODE: Converting image for decoding from {img.mode} to RGBA")
                img = img.convert('RGBA')

            binary_bits_list = []
            delimiter = '1111111111111110'
            img_data = list(img.getdata())

            for pixel in img_data:
                r, g, b = pixel[0], pixel[1], pixel[2]
                
                binary_bits_list.append(str(r & 1))
                binary_bits_list.append(str(g & 1))
                binary_bits_list.append(str(b & 1))

                current_binary_data_segment = "".join(binary_bits_list[-len(delimiter):])
                if delimiter in current_binary_data_segment:
                    print("LSB_TEXT_DECODE: Delimiter found during decoding.")
                    full_binary_data = "".join(binary_bits_list)
                    full_binary_data = full_binary_data[:full_binary_data.index(delimiter)]
                    break
            else:
                print("LSB_TEXT_DECODE WARNING: Delimiter not found. Message might be incomplete or image not encoded.")
                full_binary_data = "".join(binary_bits_list)
            
            # --- Decryption Layer ---
            # Header: 1 bit for encrypted (1=yes, 0=no) + 8 bits for salt length in bytes (if encrypted)
            header_len_bits = 1 + 8
            if len(full_binary_data) < header_len_bits:
                raise ValueError("Payload too short to contain header for decryption.")
            
            header_bits = full_binary_data[:header_len_bits]
            is_encrypted = header_bits[0] == '1'
            salt_len_bytes = int(header_bits[1:header_len_bits], 2)
            
            payload_binary_after_header = full_binary_data[header_len_bits:]
            
            if is_encrypted:
                if not password:
                    raise ValueError("Payload is encrypted but no password was provided.")
                
                if len(payload_binary_after_header) < salt_len_bytes * 8:
                    raise ValueError("Encrypted payload too short to contain salt.")
                
                try:
                    payload_bytes = bin_to_bytes(payload_binary_after_header)
                    
                    salt_bytes = payload_bytes[:salt_len_bytes]
                    encrypted_data_bytes = payload_bytes[salt_len_bytes:]
                    
                    decrypted_data_bytes = decrypt_data(encrypted_data_bytes, password, salt_bytes)
                    decoded_text = decrypted_data_bytes.decode('utf-8')
                    print(f"LSB_TEXT_DECODE: Decrypted text: '{decoded_text}'")
                    return decoded_text
                except Exception as e:
                    print(f"CRYPTO ERROR: Decryption failed: {e}")
                    raise ValueError(f"Decryption failed. Incorrect password or corrupted data: {e}")
            else:
                decoded_text = bin_to_text(payload_binary_after_header)
                print(f"LSB_TEXT_DECODE: Decoded text (unencrypted): '{decoded_text}'")
                return decoded_text
            # --- End Decryption Layer ---
    except FileNotFoundError:
        raise FileNotFoundError(f"Encoded image not found: {input_image_path}")
    except Exception as e:
        raise Exception(f"Failed to open/process image for decoding: {e}")


def encode_image_dct(secret_image_file, cover_image_path, output_image_path, password=None):
    """
    Encodes a secret image into a cover image using DCT-based steganography.
    """
    print(f"DCT_IMAGE: Starting image-in-image DCT encoding.")
    try:
        secret_binary_data, secret_original_mode, secret_original_size = image_to_bin(secret_image_file)
        
        # Metadata encoding: mode|width|height|data_length_in_bits
        metadata = f"{secret_original_mode}|{secret_original_size[0]}|{secret_original_size[1]}|{len(secret_binary_data)}"
        binary_metadata = text_to_bin(metadata)
        
        # Combine metadata and image data
        full_payload_binary = binary_metadata + secret_binary_data
        
        # --- Encryption Layer ---
        payload_to_embed_bytes = bin_to_bytes(full_payload_binary)
        salt_len_bytes = 0
        if password:
            encrypted_data_bytes, salt = encrypt_data(payload_to_embed_bytes, password)
            payload_to_embed_bytes = salt + encrypted_data_bytes
            salt_len_bytes = len(salt)
            print(f"DCT_IMAGE: Encrypted data bytes length: {len(encrypted_data_bytes)} bytes.")
        
        # Prepend header: 1 bit for encrypted (1=yes, 0=no) + 8 bits for salt length in bytes
        header_bits = '1' if password else '0'
        header_bits += format(salt_len_bytes, '08b') # Salt length in bytes (8 bits)
        
        full_binary_to_embed = header_bits + bytes_to_bin(payload_to_embed_bytes)
        print(f"DCT_IMAGE: Full binary data to embed (with header): {len(full_binary_to_embed)} bits.")
        # --- End Encryption Layer ---

        # Prepend total payload length as a fixed 32-bit binary string (for DCT specific extraction)
        payload_len_bin = format(len(full_binary_to_embed), '032b') # Use 32 bits for length for larger payloads
        final_binary_stream_to_embed = payload_len_bin + full_binary_to_embed
        
        print(f"DCT_IMAGE: Final binary data to embed (with length header): {len(final_binary_stream_to_embed)} bits")

        with Image.open(cover_image_path) as cover_img: # Use with statement
            print(f"DCT_IMAGE: Cover image opened. Mode: {cover_img.mode}, Size: {cover_img.size}")

            if cover_img.mode != 'RGB':
                print(f"DCT_IMAGE: Converting cover image from {cover_img.mode} to RGB.")
                cover_img = cover_img.convert('RGB')

            width, height = cover_img.size
            if width % 8 != 0 or height % 8 != 0:
                print("DCT_IMAGE WARNING: Cover image dimensions not multiple of 8. Cropping/resizing for encoding.")
                new_width = (width // 8) * 8
                new_height = (height // 8) * 8
                cover_img = cover_img.crop((0, 0, new_width, new_height))
                width, height = cover_img.size
                print(f"DCT_IMAGE: Resized cover image to {width}x{height}.")

            img_array = np.array(cover_img, dtype=np.float32)
            img_array -= 128 

            coeffs_per_block_for_embedding = 20 # A tunable parameter for embedding bits per 8x8 block
            total_blocks = (width // 8) * (height // 8) * 3 # R, G, B channels
            max_capacity_bits = total_blocks * coeffs_per_block_for_embedding
            print(f"DCT_IMAGE: Estimated cover image capacity (DCT): {max_capacity_bits} bits.")

            if len(final_binary_stream_to_embed) > max_capacity_bits:
                raise ValueError(f"Secret data too large for the chosen cover image (DCT capacity exceeded). Required: {len(final_binary_stream_to_embed)} bits, Available: {max_capacity_bits} bits.")

            bit_index = 0
            
            for i in range(0, height, 8):
                for j in range(0, width, 8):
                    for c in range(3):
                        if bit_index >= len(final_binary_stream_to_embed):
                            break

                        block = img_array[i:i+8, j:j+8, c]
                        
                        dct_block = _dct2d(block)
                        
                        quant_block = np.round(dct_block / QUANTIZATION_MATRIX).astype(int)

                        zigzag_coeffs = _zigzag_scan(quant_block)
                        
                        for coeff_idx in range(5, min(25, len(zigzag_coeffs))): # Embed in a range of coeffs
                            if bit_index < len(final_binary_stream_to_embed):
                                original_coeff = zigzag_coeffs[coeff_idx]
                                bit_to_embed = int(final_binary_stream_to_embed[bit_index])
                                zigzag_coeffs[coeff_idx] = (original_coeff & ~1) | bit_to_embed
                                bit_index += 1
                            else:
                                break
                        
                        if bit_index < len(final_binary_stream_to_embed) and coeff_idx == min(25, len(zigzag_coeffs)) - 1:
                            pass

                        modified_quant_block = _inverse_zigzag_scan(zigzag_coeffs)

                        idct_input = (modified_quant_block * QUANTIZATION_MATRIX).astype(np.float32)
                        
                        modified_block = _idct2d(idct_input)
                        
                        img_array[i:i+8, j:j+8, c] = np.clip(modified_block + 128, 0, 255).astype(np.uint8)

            stego_img = Image.fromarray(img_array.astype(np.uint8), 'RGB')
            print(f"DCT_IMAGE: Saving encoded image to {output_image_path}")
            stego_img.save(output_image_path, quality=95)
            print("DCT_IMAGE: Encoded image saved successfully.")

    except FileNotFoundError:
        raise FileNotFoundError(f"Cover image not found: {cover_image_path}")
    except Exception as e:
        print(f"DCT_IMAGE ERROR: DCT encoding failed: {e}")
        raise Exception(f"DCT encoding failed: {e}")


def decode_image_dct(encoded_image_path: str, output_secret_image_path: str, password: str = None):
    """
    Decodes a hidden image from a DCT-encoded cover image.
    """
    print(f"DCT_IMAGE_DECODE: Starting image-from-image DCT decoding for {encoded_image_path}")
    try:
        with Image.open(encoded_image_path) as encoded_img: # Use with statement
            print(f"DCT_IMAGE_DECODE: Encoded image opened. Mode: {encoded_img.mode}, Size: {encoded_img.size}")

            if encoded_img.mode != 'RGB':
                encoded_img = encoded_img.convert('RGB')

            width, height = encoded_img.size
            if width % 8 != 0 or height % 8 != 0:
                print("DCT_IMAGE_DECODE WARNING: Encoded image dimensions not multiple of 8. Cropping/resizing for decoding.")
                new_width = (width // 8) * 8
                new_height = (height // 8) * 8
                encoded_img = encoded_img.crop((0, 0, new_width, new_height))
                width, height = encoded_img.size
                print(f"DCT_IMAGE_DECODE: Resized encoded image to {width}x{height}.")

            img_array = np.array(encoded_img, dtype=np.float32)
            img_array -= 128

            extracted_binary_bits_list = []
            coeffs_per_block_for_embedding = 20 

            for i in range(0, height, 8):
                for j in range(0, width, 8):
                    for c in range(3):
                        block = img_array[i:i+8, j:j+8, c]
                        
                        dct_block = _dct2d(block)
                        quant_block = np.round(dct_block / QUANTIZATION_MATRIX).astype(int)
                        zigzag_coeffs = _zigzag_scan(quant_block)
                        
                        for coeff_idx in range(5, min(25, len(zigzag_coeffs))):
                            extracted_bit = str(zigzag_coeffs[coeff_idx] & 1)
                            extracted_binary_bits_list.append(extracted_bit)
            
            full_extracted_binary_data_with_len = "".join(extracted_binary_bits_list)
            print(f"DCT_IMAGE_DECODE: Extracted {len(full_extracted_binary_data_with_len)} bits from cover image.")

            # --- Extract payload length (first 32 bits) ---
            payload_len_header_bits = 32
            if len(full_extracted_binary_data_with_len) < payload_len_header_bits:
                raise ValueError("Not enough data to extract payload length header (less than 32 bits).")
            
            payload_len_bin = full_extracted_binary_data_with_len[:payload_len_header_bits]
            payload_length = int(payload_len_bin, 2)
            print(f"DCT_IMAGE_DECODE: Extracted payload length (binary): {payload_len_bin}, (decimal): {payload_length} bits.")

            full_extracted_binary_data = full_extracted_binary_data_with_len[payload_len_header_bits : payload_len_header_bits + payload_length]
            
            if len(full_extracted_binary_data) < payload_length:
                raise ValueError(f"Extracted data is shorter than expected payload length. Expected: {payload_length}, Got: {len(full_extracted_binary_data)}")
            elif len(full_extracted_binary_data) > payload_length:
                print(f"DCT_IMAGE_DECODE WARNING: Extracted data is longer than expected. Truncating to expected length: {payload_length}.")
                full_extracted_binary_data = full_extracted_binary_data[:payload_length] # Trim any excess bits


            # --- Decryption Layer ---
            # Header: 1 bit for encrypted (1=yes, 0=no) + 8 bits for salt length in bytes
            header_len_bits = 1 + 8
            if len(full_extracted_binary_data) < header_len_bits:
                raise ValueError("Payload too short to contain header for decryption.")
            
            header_bits = full_extracted_binary_data[:header_len_bits]
            is_encrypted = header_bits[0] == '1'
            salt_len_bytes = int(header_bits[1:header_len_bits], 2)
            
            payload_binary_after_header = full_extracted_binary_data[header_len_bits:]

            if is_encrypted:
                if not password:
                    raise ValueError("Payload is encrypted but no password was provided.")
                
                if len(payload_binary_after_header) < salt_len_bytes * 8:
                    raise ValueError("Encrypted payload too short to contain salt.")
                
                try:
                    payload_bytes = bin_to_bytes(payload_binary_after_header)
                    
                    salt_bytes = payload_bytes[:salt_len_bytes]
                    encrypted_data_bytes = payload_bytes[salt_len_bytes:]
                    
                    decrypted_payload_bytes = decrypt_data(encrypted_data_bytes, password, salt_bytes)
                    decrypted_payload_binary = bytes_to_bin(decrypted_payload_bytes)
                    print(f"DCT_IMAGE_DECODE: Decrypted payload length: {len(decrypted_payload_binary)} bits.")
                except InvalidToken as e:
                    print(f"CRYPTO ERROR: Decryption failed: {e}")
                    raise ValueError(f"Decryption failed. Incorrect password or corrupted data: {e}")
            else:
                decrypted_payload_binary = payload_binary_after_header
                print(f"DCT_IMAGE_DECODE: Unencrypted payload length: {len(decrypted_payload_binary)} bits.")
            # --- End Decryption Layer ---


            # Now, parse the metadata from the decrypted binary payload
            # Let's define a fixed length for metadata length header, e.g., 16 bits.
            METADATA_LENGTH_HEADER_BITS = 16
            
            if len(decrypted_payload_binary) < METADATA_LENGTH_HEADER_BITS:
                raise ValueError("Decrypted payload too short to contain metadata length header.")
            
            metadata_len_bin_header = decrypted_payload_binary[:METADATA_LENGTH_HEADER_BITS]
            metadata_length_bits = int(metadata_len_bin_header, 2)
            
            if len(decrypted_payload_binary) < METADATA_LENGTH_HEADER_BITS + metadata_length_bits:
                raise ValueError("Decrypted payload too short for stated metadata length.")
            
            metadata_bin = decrypted_payload_binary[METADATA_LENGTH_HEADER_BITS : METADATA_LENGTH_HEADER_BITS + metadata_length_bits]
            image_binary_content = decrypted_payload_binary[METADATA_LENGTH_HEADER_BITS + metadata_length_bits :]
            
            metadata_str = bin_to_text(metadata_bin)
            print(f"DCT_IMAGE_DECODE: Extracted metadata string: '{metadata_str}'")

            try:
                mode_str, width_str, height_str, data_length_str = metadata_str.split('|')
                original_mode = mode_str
                original_size = (int(width_str), int(height_str))
                expected_data_length = int(data_length_str) # The length of the image's binary data
                print(f"DCT_IMAGE_DECODE: Parsed original mode: {original_mode}, size: {original_size}, expected data length: {expected_data_length} bits")
            except ValueError as e:
                raise ValueError(f"Failed to parse hidden image metadata: {e}. Extracted string: '{metadata_str}'")
            except Exception as e:
                raise Exception(f"Error extracting hidden image metadata: {e}")

            if len(image_binary_content) < expected_data_length:
                print(f"DCT_IMAGE_DECODE ERROR: Extracted image data is shorter than expected. Expected: {expected_data_length}, Got: {len(image_binary_content)}. Attempting partial reconstruction.")
            elif len(image_binary_content) > expected_data_length:
                 print(f"DCT_IMAGE_DECODE WARNING: Extracted image data is longer than expected. Truncating to expected length: {expected_data_length}.")
                 image_binary_content = image_binary_content[:expected_data_length]

            bin_to_image(image_binary_content, output_secret_image_path, original_size, original_mode)
            print(f"DCT_IMAGE_DECODE: Secret image reconstructed to {output_secret_image_path}")
            return output_secret_image_path

    except FileNotFoundError:
        raise FileNotFoundError(f"Encoded image not found: {encoded_image_path}")
    except Exception as e:
        print(f"DCT_IMAGE_DECODE ERROR: DCT decoding failed: {e}")
        raise Exception(f"DCT decoding failed: {e}")
