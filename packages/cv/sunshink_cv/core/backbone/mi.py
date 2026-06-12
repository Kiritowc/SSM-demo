import io
import json
import os
import time
from pathlib import Path

import torch
from cryptography.fernet import Fernet


def _repo_root() -> Path:
    # packages/cv/sunshink_cv/core/backbone/mi.py -> parents[5] == repository root (contains cv/)
    return Path(__file__).resolve().parents[5]




class MoEn:

    def __init__(self, key="BOjhNzbx5waxYRqCHq0F9tLAWSv925QkvMh10Pbnq9g="):
        """
        初始化加密器
        """
        if key is None:
            self.key = Fernet.generate_key()
            with open("secret.key", "wb") as key_file:
                key_file.write(self.key)
        else:
            self.key = key.encode()
        self.cipher_suite = Fernet(self.key)

    def encrypt_model(self, input_file_path, output_file_path):
        """
        加密模型文件
        """
        in_path = Path(input_file_path)
        if not in_path.is_absolute():
            in_path = _repo_root() / in_path
        with open(in_path, "rb") as file:
            file_data = file.read()
        encrypted_data = self.cipher_suite.encrypt(file_data)
        with open(output_file_path, "wb") as encrypted_file:
            encrypted_file.write(encrypted_data)
        print(f"模型文件已存储到 {output_file_path}")

    def de_model_to_memory(self, input_file_path):
        """
        解密模型文件到内存
        """
        path = Path(input_file_path)
        if not path.is_absolute():
            path = _repo_root() / path
        with open(str(path), "rb") as encrypted_file:
            encrypted_data = encrypted_file.read()
        decrypted_data = self.cipher_suite.decrypt(encrypted_data)
        return decrypted_data


    @staticmethod
    def load_cipher_suite():
        """
        加载密钥并创建加密套件
        """
        key="BOjhNzbx5waxYRqCHq0F9tLAWSv925QkvMh10Pbnq9g="
        return Fernet(key.encode())
    
    def en_save_model(self, model_state_dict, output_file_path):
        """
        加密并保存模型权重
        """
        buffer = io.BytesIO()
        torch.save(model_state_dict, buffer)
        buffer.seek(0)
        model_bytes = buffer.read()
        encrypted_data = self.cipher_suite.encrypt(model_bytes)
        with open(output_file_path, "wb") as encrypted_file:
            encrypted_file.write(encrypted_data)
        print(f"模型权重已存储到 {output_file_path}")

    def en_save_model_buffer(self, model_data, output_file_path):
        """
        加密并保存模型数据
        """
        encrypted_data = self.cipher_suite.encrypt(model_data)
        with open(output_file_path, "wb") as encrypted_file:
            encrypted_file.write(encrypted_data)
        print(f"模型数据已存储到 {output_file_path}")

    def load_and_decrypt_model(self, input_file_path):
        """
        加载并解密模型权重
        """
        path = Path(input_file_path)
        if not path.is_absolute():
            path = _repo_root() / path
        with open(str(path), "rb") as encrypted_file:
            encrypted_data = encrypted_file.read()
        decrypted_data = self.cipher_suite.decrypt(encrypted_data)
        buffer = io.BytesIO(decrypted_data)
        buffer.seek(0)  
        model_state_dict = torch.load(buffer, map_location=torch.device('cpu'))
        return model_state_dict