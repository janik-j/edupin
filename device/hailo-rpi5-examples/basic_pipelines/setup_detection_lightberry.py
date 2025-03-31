#!/usr/bin/env python3
import os
import sys
import subprocess
import argparse
from pathlib import Path

def setup_environment():
    """Setup environment variables and paths"""
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Find lightberry-client directory (assuming it's two directories up then into lightberry-client)
    lightberry_path = os.path.abspath(os.path.join(script_dir, '../../lightberry-client'))
    
    if not os.path.exists(lightberry_path):
        print(f"Error: Lightberry client not found at {lightberry_path}")
        print("Please make sure the lightberry-client repository is in the correct location.")
        return False
        
    print(f"Found lightberry-client at: {lightberry_path}")
    
    # Add lightberry-client to Python path
    sys.path.append(lightberry_path)
    
    return True

def setup_lightberry_device(device_name=None):
    """Setup Lightberry device if not already set up"""
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    lightberry_path = os.path.abspath(os.path.join(script_dir, '../../lightberry-client'))
    
    # Check if device keys directory exists
    device_keys_path = os.path.join(lightberry_path, 'device_keys')
    if not os.path.exists(device_keys_path):
        os.makedirs(device_keys_path, exist_ok=True)
        
    # Check if device is already set up (private key exists)
    private_key_path = os.path.join(device_keys_path, 'private_key.pem')
    if os.path.exists(private_key_path):
        print("Device already set up. Using existing keys.")
        return True
        
    # Run setup_device.py script
    setup_script = os.path.join(lightberry_path, 'setup_device.py')
    cmd = [sys.executable, setup_script]
    
    if device_name:
        cmd.extend(['--name', device_name])
        
    print("\nSetting up Lightberry device...")
    print(f"Running: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        print("\nDevice setup complete!")
        return True
    except subprocess.CalledProcessError:
        print("\nError: Failed to set up device")
        return False

def register_device(server, username, password, admin=False):
    """Register device with Lightberry server"""
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    lightberry_path = os.path.abspath(os.path.join(script_dir, '../../lightberry-client'))
    
    # Device registration info
    reg_file = os.path.join(lightberry_path, 'device_keys', 'registration_info.json')
    if not os.path.exists(reg_file):
        print(f"Error: Registration file not found at {reg_file}")
        print("Please run setup first.")
        return False
        
    # Run register_device_cli.py script
    register_script = os.path.join(lightberry_path, 'register_device_cli.py')
    cmd = [
        sys.executable, 
        register_script,
        '--file', reg_file,
        '--server', server,
        '--username', username,
        '--password', password
    ]
    
    if admin:
        cmd.append('--admin')
        
    print("\nRegistering device with Lightberry server...")
    print(f"Running: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        print("\nDevice registration complete!")
        return True
    except subprocess.CalledProcessError:
        print("\nError: Failed to register device")
        return False

def run_detection():
    """Run detection script with Lightberry integration"""
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    detection_script = os.path.join(script_dir, 'detection.py')
    
    if not os.path.exists(detection_script):
        print(f"Error: Detection script not found at {detection_script}")
        return False
        
    print("\nRunning detection with Lightberry integration...")
    cmd = [sys.executable, detection_script]
    
    try:
        subprocess.run(cmd)
        return True
    except KeyboardInterrupt:
        print("\nDetection stopped by user")
        return True
    except Exception as e:
        print(f"\nError running detection: {e}")
        return False

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Setup and run detection with Lightberry integration')
    
    # Setup options
    parser.add_argument('--setup', action='store_true', help='Set up Lightberry device')
    parser.add_argument('--device-name', help='Name for the Lightberry device')
    
    # Registration options
    parser.add_argument('--register', action='store_true', help='Register device with Lightberry server')
    parser.add_argument('--server', help='Lightberry server address')
    parser.add_argument('--username', help='Username for Lightberry server')
    parser.add_argument('--password', help='Password for Lightberry server')
    parser.add_argument('--admin', action='store_true', help='Register as admin')
    
    # Run options
    parser.add_argument('--run', action='store_true', help='Run detection with Lightberry integration')
    
    args = parser.parse_args()
    
    # If no arguments provided, show help
    if not any(vars(args).values()):
        parser.print_help()
        sys.exit(0)
        
    return args

def main():
    """Main function"""
    print("\n=== Detection with Lightberry Integration ===\n")
    
    # Parse arguments
    args = parse_args()
    
    # Setup environment
    if not setup_environment():
        return
    
    # Setup Lightberry device if requested
    if args.setup:
        if not setup_lightberry_device(args.device_name):
            return
    
    # Register device if requested
    if args.register:
        if not all([args.server, args.username, args.password]):
            print("Error: --server, --username, and --password are required for registration")
            return
            
        if not register_device(args.server, args.username, args.password, args.admin):
            return
    
    # Run detection if requested
    if args.run:
        run_detection()

if __name__ == "__main__":
    main() 