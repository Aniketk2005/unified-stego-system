import os
import io
import numpy as np
from PIL import Image
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