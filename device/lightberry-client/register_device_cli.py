#!/usr/bin/env python3
import os
import sys
import json
import argparse
import requests
import websockets
import asyncio
from pathlib import Path

# Token storage
TOKEN_FILE = '.lightberry_token'

def get_auth_token(server_url, username, password):
    """Get authentication token using user credentials"""
    try:
        # Ensure we have a proper URL
        if not server_url.startswith('http'):
            base_url = f"http://{server_url}"
        else:
            base_url = server_url
            
        # Remove trailing slash if present
        if base_url.endswith('/'):
            base_url = base_url[:-1]
            
        # Login endpoint
        login_url = f"{base_url}/api/token"
        
        print(f"Authenticating with server at {login_url}")
        
        # Make login request
        response = requests.post(
            login_url,
            data={"username": username, "password": password},
            timeout=10
        )
        
        if response.status_code == 200:
            token_data = response.json()
            access_token = token_data.get('access_token')
            
            if access_token:
                print("Authentication successful!")
                
                # Save token to file
                with open(TOKEN_FILE, 'w') as f:
                    json.dump(token_data, f)
                    
                return access_token
                
            else:
                print("Error: No access token in response")
                return None
                
        elif response.status_code == 401:
            print("Authentication failed. Please check your credentials.")
            return None
            
        else:
            print(f"Login failed with status code {response.status_code}: {response.text}")
            return None
            
    except Exception as e:
        print(f"Error during authentication: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_user_devices(server_url, token):
    """Get list of devices owned by the user"""
    try:
        # Ensure we have a proper URL
        if not server_url.startswith('http'):
            base_url = f"http://{server_url}"
        else:
            base_url = server_url
            
        # Remove trailing slash if present
        if base_url.endswith('/'):
            base_url = base_url[:-1]
            
        # Devices endpoint
        devices_url = f"{base_url}/api/devices"
        
        # Make request
        response = requests.get(
            devices_url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=10
        )
        
        if response.status_code == 200:
            devices = response.json()
            return devices
        else:
            print(f"Failed to get devices: {response.text}")
            return []
            
    except Exception as e:
        print(f"Error getting user devices: {e}")
        return []

def register_device_as_user(registration_data, server_url, token):
    """Register device with server using user credentials"""
    try:
        # Ensure we have a proper URL
        if not server_url.startswith('http'):
            base_url = f"http://{server_url}"
        else:
            base_url = server_url
            
        # Remove trailing slash if present
        if base_url.endswith('/'):
            base_url = base_url[:-1]
            
        # First, get current user devices to check the limit
        devices = get_user_devices(server_url, token)
        
        if len(devices) >= 10:
            print(f"Error: You already have {len(devices)} devices registered. The limit is 10 devices per user.")
            return False
            
        # Devices endpoint
        register_url = f"{base_url}/api/devices"
        
        print(f"Registering device with server at {register_url}")
        
        # Prepare data for API request
        api_data = {
            'device_id': registration_data['device_id'],
            'public_key': registration_data['public_key'],
            'name': registration_data.get('name')
        }
        
        # Make API request
        response = requests.post(
            register_url,
            json=api_data,
            headers={"Authorization": f"Bearer {token}"},
            timeout=10
        )
        
        # Check response
        if response.status_code in [200, 201]:
            print(f"Device registered successfully as user!")
            return True
        elif response.status_code == 401:
            print("Authentication failed. Your session may have expired.")
            return False
        else:
            print(f"Registration failed: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"Could not connect to {base_url}. Please check the server URL and try again.")
        return False
    except Exception as e:
        print(f"Error during registration: {e}")
        import traceback
        traceback.print_exc()
        return False

def register_device_via_api(registration_data, server_url, admin_username, admin_password):
    """Register device with server via REST API with admin credentials"""
    try:
        # Ensure we have a proper URL for the API endpoint
        if not server_url.startswith('http'):
            if server_url.startswith('ws://'):
                base_url = server_url.replace('ws://', 'http://')
            elif server_url.startswith('wss://'):
                base_url = server_url.replace('wss://', 'https://')
            else:
                base_url = f"http://{server_url}"
        else:
            base_url = server_url

        # Make sure we use the API endpoint
        if not base_url.endswith('/api/register_device'):
            # Remove trailing slash if present
            if base_url.endswith('/'):
                base_url = base_url[:-1]
            
            # Add API endpoint
            base_url = f"{base_url}/api/register_device"
        
        print(f"Registering device with server at {base_url}")
        
        # Prepare data for API request
        api_data = {
            'device_id': registration_data['device_id'],
            'public_key': registration_data['public_key'],
            'name': registration_data.get('name')
        }
        
        # Make API request
        response = requests.post(
            base_url,
            json=api_data,
            auth=(admin_username, admin_password),
            timeout=10
        )
        
        # Check response
        if response.status_code == 200:
            print(f"Device registered successfully through API: {response.json().get('message', 'Success')}")
            return True
        elif response.status_code == 401:
            print("Authentication failed. Please check your admin credentials.")
            return False
        else:
            print(f"API registration failed: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"Could not connect to {base_url}. Please check the server URL and try again.")
        return False
    except Exception as e:
        print(f"Error during API registration: {e}")
        import traceback
        traceback.print_exc()
        return False

async def register_device_via_websocket(registration_data, server_url, admin_username, admin_password):
    """Register device with server via WebSocket connection (admin only)"""
    try:
        # Ensure we have a proper WebSocket URL
        if not server_url.startswith('ws'):
            if server_url.startswith('http://'):
                websocket_url = server_url.replace('http://', 'ws://')
            elif server_url.startswith('https://'):
                websocket_url = server_url.replace('https://', 'wss://')
            else:
                websocket_url = f"ws://{server_url}"
        else:
            websocket_url = server_url
        
        # Make sure it includes port 8765 if not already specified
        if not ':' in websocket_url.split('//')[1]:
            websocket_url = f"{websocket_url}:8765"
            
        print(f"Connecting to WebSocket server at {websocket_url}")
        
        # Connect to WebSocket and register device
        async with websockets.connect(websocket_url) as websocket:
            # Prepare registration message
            registration_message = {
                "type": "register_device",
                "device_id": registration_data['device_id'],
                "public_key": registration_data['public_key'],
                "admin_username": admin_username,
                "admin_password": admin_password
            }
            
            if registration_data.get('name'):
                registration_message["name"] = registration_data['name']
            
            # Send registration message
            await websocket.send(json.dumps(registration_message))
            
            # Wait for response with timeout
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                response_data = json.loads(response)
                
                if response_data.get('status') == 'success':
                    print(f"Device registered successfully through WebSocket: {response_data.get('message', 'Success')}")
                    return True
                else:
                    print(f"WebSocket registration failed: {response_data.get('message', 'Unknown error')}")
                    return False
                    
            except asyncio.TimeoutError:
                print("Timed out waiting for server response")
                return False
                
    except Exception as e:
        print(f"Error during WebSocket registration: {e}")
        import traceback
        traceback.print_exc()
        return False

def register_device(registration_file, server_url, username, password, admin_mode=False, method='both'):
    """Register a device with the server using the specified method"""
    try:
        # Load registration data
        with open(registration_file, 'r') as f:
            registration_data = json.load(f)
        
        print(f"Loaded registration data for device: {registration_data.get('device_id')}")
        
        if admin_mode:
            print("Registering device as admin")
            # Admin registration
            if method in ['api', 'both']:
                # Try API registration first
                if register_device_via_api(registration_data, server_url, username, password):
                    return True
                elif method == 'api':
                    return False
            
            if method in ['websocket', 'both']:
                # If API failed or wasn't attempted, try WebSocket
                loop = asyncio.get_event_loop()
                success = loop.run_until_complete(
                    register_device_via_websocket(registration_data, server_url, username, password)
                )
                return success
        else:
            # Regular user registration
            print("Registering device as user")
            
            # Get auth token
            token = get_auth_token(server_url, username, password)
            if not token:
                return False
                
            # Register device with user token
            return register_device_as_user(registration_data, server_url, token)
                
        return False
            
    except FileNotFoundError:
        print(f"Registration file not found: {registration_file}")
        return False
    except json.JSONDecodeError:
        print(f"Invalid JSON in registration file: {registration_file}")
        return False
    except Exception as e:
        print(f"Error during registration: {e}")
        import traceback
        traceback.print_exc()
        return False

def parse_args():
    parser = argparse.ArgumentParser(description='Register a Lightberry device with a server')
    parser.add_argument('--file', '-f', default='device_keys/registration_info.json',
                       help='Path to registration info JSON file (default: device_keys/registration_info.json)')
    parser.add_argument('--server', '-s', required=True,
                       help='Server URL (e.g., example.com or http://example.com)')
    parser.add_argument('--username', '-u', required=True,
                       help='Username for authentication')
    parser.add_argument('--password', '-p', required=True,
                       help='Password for authentication')
    parser.add_argument('--admin', '-a', action='store_true',
                       help='Use admin authentication instead of regular user')
    parser.add_argument('--method', '-m', choices=['api', 'websocket', 'both'], default='both',
                       help='Registration method to use (admin mode only, default: both)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    print("\n=== Lightberry Device Registration ===\n")
    
    success = register_device(
        args.file,
        args.server,
        args.username,
        args.password,
        args.admin,  # Admin mode flag
        args.method
    )
    
    if success:
        print("\nDevice registration completed successfully!\n")
        sys.exit(0)
    else:
        print("\nDevice registration failed. Please check the errors above.\n")
        sys.exit(1) 