from PIL import Image
import numpy as np
import io
from scipy.fftpack import dct, idct
from cryptography.fernet import InvalidToken
import re
import os
import uuid

from .utils import (
    bytes_to_bin, bin_to_bytes,
    text_to_bin, bin_to_text,
    encrypt_data, decrypt_data
)
# --- LEGACY JPEG-LIKE QUANTIZATION AND ZIGZAG HELPERS (for backward compatibility) ---
QUANTIZATION_MATRIX = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
], dtype=np.float32)

def _zigzag_scan(block: np.ndarray) -> list:
    """Reconstructs a list of coefficients from an 8x8 block using zigzag scan."""
    h, w = block.shape
    result = []
    for s in range(h + w - 1):
        if s % 2 == 0:
            for i in range(min(s, h - 1), max(-1, s - w), -1):
                j = s - i
                result.append(block[i, j])
        else:
            for i in range(max(0, s - w + 1), min(s + 1, h)):
                j = s - i
                result.append(block[i, j])
    return result

def _inverse_zigzag_scan(coeffs: list) -> np.ndarray:
    """Reconstructs an 8x8 block from zigzag-ordered coefficients."""
    block = np.zeros((8, 8), dtype=int)
    idx = 0
    for s in range(8 + 8 - 1):
        if s % 2 == 0:
            for i in range(min(s, 8 - 1), max(-1, s - 8), -1):
                j = s - i
                block[i, j] = coeffs[idx]; idx += 1
        else:
            for i in range(max(0, s - 8 + 1), min(s + 1, 8)):
                j = s - i
                block[i, j] = coeffs[idx]; idx += 1
    return block

# --- HELPER FUNCTIONS (REWRITTEN FOR ROBUSTNESS) ---

def _image_to_binary(image_file_bytes: bytes) -> tuple[str, int, int]:
    """Converts a secret image's bytes into a binary string with a robust metadata header.
    Returns (full_payload_bits, width, height).
    """
    with Image.open(io.BytesIO(image_file_bytes)) as img:
        img = img.convert("RGB")
        width, height = img.size

        # Include a magic prefix to improve header alignment detection on decode
        metadata = f"DCT1|{width},{height}"
        binary_metadata = text_to_bin(metadata)

        metadata_len_header = format(len(binary_metadata), '016b')

        pixel_data = np.array(img).flatten().tobytes()
        binary_pixel_data = bytes_to_bin(pixel_data)

        full_payload = metadata_len_header + binary_metadata + binary_pixel_data
        print(f"DCT: Secret image converted to {len(full_payload)} bits.")
        return full_payload, width, height

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
        # Support both legacy ("w,h") and new magic ("DCT1|w,h") formats
        if '|' in metadata:
            try:
                _magic, dims = metadata.split('|', 1)
                width, height = map(int, dims.split(','))
            except Exception as e:
                raise ValueError(f"Malformed metadata with magic header: {e}")
        else:
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
    """Encodes a secret image into a cover image using DCT-based steganography."""
    print("DCT_IMAGE: Starting DCT encoding.")

    with Image.open(cover_image_path) as cover_img:
        cover_img = cover_img.convert('RGB')
        cover_width, cover_height = cover_img.size

        secret_binary_payload, secret_w, secret_h = _image_to_binary(secret_image_bytes)
        payload_to_embed_bytes = bin_to_bytes(secret_binary_payload)

        salt_len_bytes = 0
        if password:
            encrypted_data_bytes, salt = encrypt_data(payload_to_embed_bytes, password)
            payload_to_embed_bytes = salt + encrypted_data_bytes
            salt_len_bytes = len(salt)

        binary_to_embed = bytes_to_bin(payload_to_embed_bytes)
        # Out-of-band sync header in first blocks: 16-bit magic + 16-bit w + 16-bit h + 32-bit payload length
        MAGIC = 0x53A7  # fixed magic to resync
        oob_sync_header = format(MAGIC, '016b') + format(secret_w, '016b') + format(secret_h, '016b') + format(len(binary_to_embed), '032b')
        # In-band header remains for backward compatibility
        header = ('1' if password else '0') + format(salt_len_bytes, '08b') + format(len(binary_to_embed), '032b')
        data_to_hide = oob_sync_header + header + binary_to_embed

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

                    # Quantize and zigzag to stabilize coefficient ordering (matches decoder path)
                    quant_block = np.round(dct_block / QUANTIZATION_MATRIX).astype(int)
                    zigzag_coeffs = _zigzag_scan(quant_block)

                    for k in range(10, 25):
                        if data_index < len(data_to_hide):
                            coeff = int(zigzag_coeffs[k])
                            bit_to_hide = int(data_to_hide[data_index])
                            zigzag_coeffs[k] = (coeff & ~1) | bit_to_hide
                            data_index += 1
                        else:
                            break

                    # Inverse zigzag and de-quantize, then inverse DCT
                    modified_quant_block = _inverse_zigzag_scan(zigzag_coeffs)
                    modified_dct_block = (modified_quant_block * QUANTIZATION_MATRIX).astype(np.float32)
                    img_array[i:i+8, j:j+8, channel] = idct(idct(modified_dct_block.T, norm='ortho').T, norm='ortho')
                if data_index >= len(data_to_hide): break
            if data_index >= len(data_to_hide): break
        # --- END OF CRITICAL FIX ---
        print(f"DCT_IMAGE: Bits intended to embed: {len(data_to_hide)}; Bits actually embedded: {data_index}")
        if data_index < len(data_to_hide):
            print("DCT_IMAGE WARNING: Payload truncated due to capacity limits or early exit.")

        stego_img_array = np.clip(img_array + 128.0, 0, 255).astype(np.uint8)
        stego_img = Image.fromarray(stego_img_array, 'RGB')
        stego_img.save(output_path, 'PNG')
        stego_img.close()
        print("DCT_IMAGE: Encoding successful.")


def decode_image_from_image(stego_image_path: str, output_path: str, password: str = None):
    """Decodes a hidden image from a DCT-encoded cover image."""
    print("DCT_IMAGE_DECODE: Starting DCT decoding.")

    with Image.open(stego_image_path) as stego_img:
        stego_img = stego_img.convert('RGB')
        width, height = stego_img.size
        img_array = np.array(stego_img, dtype=np.float32) - 128.0

        def extract_bits_with_range(k_start: int, k_end: int) -> str:
            bits = []
            for i in range(0, height - (height % 8), 8):
                for j in range(0, width - (width % 8), 8):
                    for channel in range(3):
                        block = img_array[i:i+8, j:j+8, channel]
                        dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
                        # Match encoder: quantize and zigzag before reading LSBs
                        quant_block = np.round(dct_block / QUANTIZATION_MATRIX).astype(int)
                        zigzag_coeffs = _zigzag_scan(quant_block)
                        for k in range(k_start, k_end):
                            coeff_int = int(zigzag_coeffs[k])
                            bits.append(str(coeff_int & 1))
            return "".join(bits)

        def looks_like_valid_metadata(payload_bits: str) -> bool:
            try:
                if len(payload_bits) < 16:
                    return False
                md_len = int(payload_bits[:16], 2)
                if md_len <= 0 or md_len > 16 * 24: # allow up to 384 bits (~48 chars) for magic + dims
                    return False
                if len(payload_bits) < 16 + md_len:
                    return False
                md_bin = payload_bits[16:16+md_len]
                md_text = bin_to_text(md_bin)
                # Accept either "w,h" or "DCT1|w,h"
                if '|' in md_text:
                    _, dims = md_text.split('|', 1)
                else:
                    dims = md_text
                if not re.match(r'^\d+,\d+$', dims):
                    return False
                width_str, height_str = dims.split(',')
                width = int(width_str); height = int(height_str)
                if width <= 0 or height <= 0 or width > 20000 or height > 20000:
                    return False
                return True
            except Exception:
                return False

        # Try multiple nearby coefficient ranges to account for small shifts
        candidate_ranges = [
            (10, 25), (9, 24), (11, 26),
            (8, 23), (12, 27), (13, 28)
        ]
        last_error = None
        for k_start, k_end in candidate_ranges:
            try:
                extracted_binary = extract_bits_with_range(k_start, k_end)
                # Fast path: look for out-of-band sync header: 16-bit magic + 16-bit w + 16-bit h + 32-bit len
                MAGIC = 0x53A7
                magic_bits = format(MAGIC, '016b')
                idx = extracted_binary.find(magic_bits)
                if idx != -1 and len(extracted_binary) >= idx + 16 + 16 + 16 + 32:
                    try:
                        w = int(extracted_binary[idx+16:idx+32], 2)
                        h = int(extracted_binary[idx+32:idx+48], 2)
                        payload_len_bits = int(extracted_binary[idx+48:idx+80], 2)
                        start = idx + 80
                        end = start + payload_len_bits
                        if end <= len(extracted_binary) and w > 0 and h > 0 and w <= 20000 and h <= 20000:
                            # After sync header we still have the in-band header before payload
                            if end - start >= (1+8+32):
                                is_encrypted = extracted_binary[start] == '1'
                                salt_len_bytes = int(extracted_binary[start+1:start+9], 2)
                                inner_payload_len = int(extracted_binary[start+9:start+41], 2)
                                payload_start = start + (1+8+32)
                                payload_end = payload_start + inner_payload_len
                                if payload_end <= len(extracted_binary):
                                    payload_binary = extracted_binary[payload_start:payload_end]
                                    payload_bytes = bin_to_bytes(payload_binary)
                                    if is_encrypted:
                                        if not password:
                                            continue
                                        if len(payload_bytes) < salt_len_bytes:
                                            continue
                                        salt = payload_bytes[:salt_len_bytes]
                                        enc = payload_bytes[salt_len_bytes:]
                                        dec_bytes = decrypt_data(enc, password, salt)
                                        final_binary_payload = bytes_to_bin(dec_bytes)
                                    else:
                                        final_binary_payload = payload_binary
                                    if looks_like_valid_metadata(final_binary_payload):
                                        _binary_to_image(final_binary_payload, output_path)
                                        return
                    except Exception:
                        pass
                # Try byte/bit alignment shifts 0..7 before header parsing
                shift_candidates = [extracted_binary[i:] for i in range(0, min(8, len(extracted_binary)))]
                parse_attempted = False
                parse_success = False
                for shifted in shift_candidates:
                    try:
                        # Proceed to parse header and payload with this shift
                        header_len = 1 + 8 + 32
                        if len(shifted) < header_len:
                            continue

                        is_encrypted = shifted[0] == '1'
                        salt_len_bytes = int(shifted[1:9], 2)
                        payload_len_bits = int(shifted[9:header_len], 2)
                        if payload_len_bits <= 0:
                            continue

                        total_bits_needed = header_len + payload_len_bits
                        working_binary = shifted

                        if len(working_binary) < total_bits_needed:
                            # Resync scan within the shifted stream
                            resynced = False
                            max_scan = min(65536, max(0, len(working_binary) - header_len))
                            for offset in range(1, max_scan):
                                try:
                                    cand_is_encrypted = working_binary[offset] == '1'
                                    cand_salt_len_bytes = int(working_binary[offset+1:offset+9], 2)
                                    cand_payload_len_bits = int(working_binary[offset+9:offset+header_len], 2)
                                    remaining_after_header = len(working_binary) - (offset + header_len)
                                    if 0 < cand_payload_len_bits <= remaining_after_header and 0 <= cand_salt_len_bytes <= 64:
                                        is_encrypted = cand_is_encrypted
                                        salt_len_bytes = cand_salt_len_bytes
                                        payload_len_bits = cand_payload_len_bits
                                        header_start = offset
                                        total_bits_needed = header_start + header_len + payload_len_bits
                                        resynced = True
                                        break
                                except Exception:
                                    continue
                            if not resynced:
                                continue
                            payload_start = header_start + header_len
                            payload_binary = working_binary[payload_start:payload_start+payload_len_bits]
                        else:
                            payload_start = header_len
                            payload_binary = working_binary[payload_start:payload_start+payload_len_bits]

                        payload_bytes = bin_to_bytes(payload_binary)

                        if is_encrypted:
                            if not password:
                                # if encrypted but no password, skip this shift
                                continue
                            # First try direct decryption
                            try:
                                if len(payload_bytes) < salt_len_bytes:
                                    continue
                                salt = payload_bytes[:salt_len_bytes]
                                encrypted_data = payload_bytes[salt_len_bytes:]
                                decrypted_payload_bytes = decrypt_data(encrypted_data, password, salt)
                                final_binary_payload = bytes_to_bin(decrypted_payload_bytes)
                            except Exception:
                                # Try alternative header offsets inside shifted stream
                                dec_ok = False
                                max_scan = min(65536, max(0, len(working_binary) - header_len))
                                for offset in range(1, max_scan):
                                    try:
                                        if working_binary[offset] != '1':
                                            continue
                                        local_salt_len = int(working_binary[offset+1:offset+9], 2)
                                        local_pl_bits = int(working_binary[offset+9:offset+header_len], 2)
                                        rem = len(working_binary) - (offset + header_len)
                                        if local_pl_bits <= 0 or local_pl_bits > rem:
                                            continue
                                        local_payload = working_binary[offset+header_len: offset+header_len+local_pl_bits]
                                        local_bytes = bin_to_bytes(local_payload)
                                        if len(local_bytes) < local_salt_len:
                                            continue
                                        salt_local = local_bytes[:local_salt_len]
                                        enc_local = local_bytes[local_salt_len:]
                                        decrypted_payload_bytes = decrypt_data(enc_local, password, salt_local)
                                        final_binary_payload = bytes_to_bin(decrypted_payload_bytes)
                                        dec_ok = True
                                        break
                                    except Exception:
                                        continue
                                if not dec_ok:
                                    continue
                        else:
                            final_binary_payload = payload_binary

                        if not looks_like_valid_metadata(final_binary_payload):
                            # Try legacy framing: 32-bit length prefix before header within shifted stream
                            try:
                                if len(shifted) < 32:
                                    continue
                                legacy_len = int(shifted[:32], 2)
                                if legacy_len <= 0 or legacy_len > len(shifted) - 32:
                                    continue
                                legacy_payload = shifted[32:32+legacy_len]
                                # Inside legacy payload: same header (1+8) then encrypted bytes
                                if len(legacy_payload) < header_len:
                                    continue
                                leg_is_enc = legacy_payload[0] == '1'
                                leg_salt_len = int(legacy_payload[1:9], 2)
                                leg_body = legacy_payload[header_len:]
                                if leg_is_enc:
                                    leg_bytes = bin_to_bytes(leg_body)
                                    if len(leg_bytes) < leg_salt_len:
                                        continue
                                    if not password:
                                        # legacy encrypted but no password provided
                                        continue
                                    salt = leg_bytes[:leg_salt_len]
                                    enc = leg_bytes[leg_salt_len:]
                                    dec_bytes = decrypt_data(enc, password, salt)
                                    candidate_payload = bytes_to_bin(dec_bytes)
                                else:
                                    candidate_payload = leg_body
                                if not looks_like_valid_metadata(candidate_payload):
                                    continue
                                _binary_to_image(candidate_payload, output_path)
                                parse_success = True
                                break
                            except Exception:
                                continue

                        _binary_to_image(final_binary_payload, output_path)
                        parse_success = True
                        break
                    except Exception as e:
                        parse_attempted = True
                        continue
                if parse_success:
                    return
                else:
                    raise ValueError("Unable to parse header in all bit alignments for this coefficient range.")
            except Exception as e:
                last_error = e
                continue
        # If all ranges failed, write a small debug dump to help diagnose alignment
        try:
            debug_bits = extract_bits_with_range(10, 25)
            debug_snippet = debug_bits[:4096]
            debug_id = str(uuid.uuid4())
            debug_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'decoded_output', f'dct_debug{debug_id}.txt')
            os.makedirs(os.path.dirname(debug_path), exist_ok=True)
            with open(debug_path, 'w') as f:
                f.write('k_range=10-25\n')
                f.write(debug_snippet)
            print(f"DCT_IMAGE_DECODE DEBUG: Wrote header bits dump to {debug_path}")
        except Exception as _:
            pass
        if last_error:
            raise last_error