# backend/steganography/utils.py

import os
import base64
from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend

# --- Encryption/Decryption Utilities ---
def _derive_key(password: str, salt: bytes) -> bytes:
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
        backend=default_backend()
    )
    return base64.urlsafe_b64encode(kdf.derive(password.encode()))

def encrypt_data(data: bytes, password: str) -> tuple[bytes, bytes]:
    salt = os.urandom(16)
    key = _derive_key(password, salt)
    f = Fernet(key)
    return f.encrypt(data), salt

def decrypt_data(encrypted_data: bytes, password: str, salt: bytes) -> bytes:
    key = _derive_key(password, salt)
    f = Fernet(key)
    return f.decrypt(encrypted_data)

# --- Binary/Bytes Conversion Utilities ---
def text_to_bin(text: str) -> str:
    return ''.join(format(ord(c), '08b') for c in text)

def bin_to_text(binary_string: str) -> str:
    text = ""
    for i in range(0, len(binary_string), 8):
        byte = binary_string[i:i+8]
        if len(byte) == 8:
            text += chr(int(byte, 2))
    return text

def bytes_to_bin(data_bytes: bytes) -> str:
    return ''.join(format(byte, '08b') for byte in data_bytes)

def bin_to_bytes(binary_string: str) -> bytes:
    padded_binary_string = binary_string + '0' * ((8 - len(binary_string) % 8) % 8)
    return int(padded_binary_string, 2).to_bytes(len(padded_binary_string) // 8, byteorder='big')