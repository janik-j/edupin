#!/usr/bin/env python3
import os
import sys
import uuid
import json
import base64
import argparse
from pathlib import Path
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding

class DeviceSetup:
    def __init__(self, device_name=None, keys_dir='device_keys'):
        self.keys_dir = Path(keys_dir)
        self.keys_dir.mkdir(exist_ok=True)
        self.device_name = device_name
        self.device_id = None
        self.private_key = None
        self.public_key_pem = None
    
    def generate_device_id(self):
        """Generate a unique device ID"""
        self.device_id = str(uuid.uuid4())
        id_file = self.keys_dir / 'device_id'
        id_file.write_text(self.device_id)
        return self.device_id
    
    def generate_keys(self):
        """Generate RSA key pair for device authentication"""
        # Generate private key
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        self.private_key = private_key
        
        # Save private key
        pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        key_file = self.keys_dir / 'private_key.pem'
        key_file.write_bytes(pem)
        
        # Generate and save public key
        public_key = private_key.public_key()
        pub_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        self.public_key_pem = pub_pem.decode('utf-8')
        (self.keys_dir / 'public_key.pem').write_bytes(pub_pem)
        
        return self.public_key_pem
    
    def save_device_name(self):
        """Save device name to a file"""
        if not self.device_name:
            self.device_name = input("Please enter the device name: ")
        name_file = self.keys_dir / 'device_name'
        name_file.write_text(self.device_name)
    
    def setup_device(self):
        """Complete device setup process"""
        print("\n=== Lightberry Device Setup ===\n")
        
        # Check if device ID already exists
        id_file = self.keys_dir / 'device_id'
        if id_file.exists():
            self.device_id = id_file.read_text().strip()
            print(f"Using existing device ID: {self.device_id}")
        else:
            self.device_id = self.generate_device_id()
            print(f"Generated new device ID: {self.device_id}")
        
        # Check if keys already exist
        key_file = self.keys_dir / 'private_key.pem'
        if key_file.exists():
            print("Using existing key pair")
            with open(self.keys_dir / 'public_key.pem', 'r') as f:
                self.public_key_pem = f.read()
        else:
            print("Generating new key pair...")
            self.public_key_pem = self.generate_keys()
            print("Key pair generated successfully")
        
        # Save device name
        self.save_device_name()
        if self.device_name:
            print(f"Device name set to: {self.device_name}")
        
        # Output registration information
        self.output_registration_info()
    
    def output_registration_info(self):
        """Output registration information for the web interface"""
        print("\n=== Device Registration Information ===\n")
        print("Use this information to register your device in the web interface:\n")
        print(f"Device ID: {self.device_id}")
        print(f"Device Name: {self.device_name or 'Not set'}")
        print("\nPublic Key:")
        print("--------- COPY FROM HERE ---------")
        print(self.public_key_pem)
        print("--------- TO HERE ---------")
        
        # Create JSON for easy copy-paste
        registration_data = {
            "device_id": self.device_id,
            "name": self.device_name,
            "public_key": self.public_key_pem
        }
        
        # Save registration data to a file for easy access
        reg_file = self.keys_dir / 'registration_info.json'
        with open(reg_file, 'w') as f:
            json.dump(registration_data, f, indent=2)
        
        print(f"\nRegistration data saved to: {reg_file}")
        print("\nYou can now register this device through the web interface at:")
        print("https://lightberry.yourserver.com/devices\n")

def parse_args():
    parser = argparse.ArgumentParser(description='Set up a Lightberry device')
    parser.add_argument('--name', '-n', help='Device name (optional)')
    parser.add_argument('--keys-dir', '-d', default='device_keys', 
                       help='Directory to store device keys and ID')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    setup = DeviceSetup(device_name=args.name, keys_dir=args.keys_dir)
    setup.setup_device() 