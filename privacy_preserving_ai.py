import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import random
from cryptography.fernet import Fernet
import socket
import pickle


class SecureDataHandler:
    def __init__(self):
        self.key = Fernet.generate_key()
        self.cipher = Fernet(self.key)

    def encrypt(self, data):
        serialized_data = pickle.dumps(data)
        encrypted_data = self.cipher.encrypt(serialized_data)
        return encrypted_data

    def decrypt(self, encrypted_data):
        decrypted_data = self.cipher.decrypt(encrypted_data)
        return pickle.loads(decrypted_data)


class PrivacyPreservingAI:
    def __init__(self):
        self.model = LogisticRegression()
        self.secure_data_handler = SecureDataHandler()

    def load_data(self):
        data = load_iris()
        X = data.data
        y = data.target
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def preprocess_data(self, X, y):
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_normalized = (X - X_mean) / X_std
        return X_normalized, y

    def train_model(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, y_true, y_pred):
        accuracy = accuracy_score(y_true, y_pred)
        print(f"Model Accuracy: {accuracy * 100:.2f}%")

    def secure_model_training(self):
        X_train, X_test, y_train, y_test = self.load_data()
        X_train, y_train = self.preprocess_data(X_train, y_train)
        encrypted_data = self.secure_data_handler.encrypt((X_train, y_train))
        X_train_decrypted, y_train_decrypted = self.secure_data_handler.decrypt(encrypted_data)
        self.train_model(X_train_decrypted, y_train_decrypted)
        return X_test, y_test

    def run(self):
        X_test, y_test = self.secure_model_training()
        X_test_normalized, _ = self.preprocess_data(X_test, None)
        y_pred = self.predict(X_test_normalized)
        self.evaluate(y_test, y_pred)


class SecureDataServer:
    def __init__(self, host='localhost', port=65432):
        self.host = host
        self.port = port
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def start_server(self):
        self.s.bind((self.host, self.port))
        self.s.listen()
        print(f"Server listening on {self.host}:{self.port}")
        conn, addr = self.s.accept()
        print(f"Connection from {addr}")
        with conn:
            encrypted_data = conn.recv(4096)
            data = pickle.loads(encrypted_data)
            print("Received data:", data)

    def stop_server(self):
        self.s.close()


class SecureDataClient:
    def __init__(self, host='localhost', port=65432):
        self.host = host
        self.port = port

    def send_data(self, data):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((self.host, self.port))
            serialized_data = pickle.dumps(data)
            s.sendall(serialized_data)


if __name__ == "__main__":
    privacy_ai = PrivacyPreservingAI()
    privacy_ai.run()

    # Server implementation (to be run in separate terminal)
    # server = SecureDataServer()
    # server.start_server()

    # Client implementation (to be run in separate terminal)
    # client = SecureDataClient()
    # test_data = {'sample': 'data'}
    # client.send_data(test_data)