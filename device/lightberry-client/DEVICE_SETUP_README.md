# Lightberry Device Setup

This guide explains how to set up and register your Lightberry voice assistant device.

## Requirements

- Python 3.7+
- The following Python packages:
  - cryptography
  - requests
  - websockets
  - asyncio

Install dependencies with:
```bash
pip install cryptography requests websockets asyncio
```

## Device Setup Process

Setting up a Lightberry device involves two main steps:

1. **Initial Device Setup**: Generate keys and device ID
2. **Registration**: Register the device with the Lightberry server

## Step 1: Initial Device Setup

The `setup_device.py` script handles the initial device setup by generating:
- A unique device ID
- RSA key pair for secure authentication
- Optional device name

### Usage:

```bash
python setup_device.py --name "My Kitchen Assistant"
```

Options:
- `--name` or `-n`: Specify a friendly name for your device (optional)
- `--keys-dir` or `-d`: Specify a directory to store device keys (default: `device_keys`)

The script will:
1. Generate a unique device ID if one doesn't exist
2. Generate RSA keys if they don't exist
3. Save the device name if provided
4. Output the registration information needed for the web interface
5. Save all registration data to a JSON file for easy access

## Step 2: Device Registration

You can register your device in several ways:

### Option A: Web Interface (For Users)

1. Log in to your Lightberry account at `https://your-lightberry-server.com`
2. Navigate to the "Devices" section
3. Click "Register New Device"
4. Enter the device information from the setup script:
   - Device ID
   - Device Name (optional)
   - Public Key

> **Note**: Each user can register up to 10 devices.

### Option B: Command Line Registration (For Users)

Use the `register_device_cli.py` script to register as a regular user:

```bash
python register_device_cli.py --server your-lightberry-server.com --username your.email@example.com --password yourpassword
```

The script will:
1. Authenticate with your user credentials
2. Check if you've reached the 10-device limit
3. Register the device and associate it with your account

### Option C: Admin Registration (For Administrators)

Administrators can register devices using:

```bash
python register_device_cli.py --server your-lightberry-server.com --username admin --password adminpassword --admin
```

Options for all registration methods:
- `--file` or `-f`: Path to registration JSON file (default: `device_keys/registration_info.json`)
- `--server` or `-s`: Server URL (required)
- `--username` or `-u`: Username for authentication (required)
- `--password` or `-p`: Password for authentication (required)
- `--admin` or `-a`: Use admin authentication instead of regular user (optional)
- `--method` or `-m`: Registration method for admin mode: `api`, `websocket`, or `both` (default: `both`)

## Device Limits

- Regular users can register up to 10 devices
- Administrators have no limit on device registration

## Next Steps

After successful registration, your device is ready to use:

1. Make sure the Lightberry server is running
2. Run the audio client:
   ```bash
   python audio_client.py --server your-lightberry-server.com
   ```

3. Follow the prompts to interact with your voice assistant

## Troubleshooting

If you encounter issues during registration:

1. **User Registration Issues**:
   - Check that your user account is active
   - Verify you haven't reached the 10-device limit
   - Make sure your credentials are correct

2. **API Registration Fails**:
   - Check that the server is running and accessible
   - Ensure the server has the right endpoints enabled

3. **WebSocket Registration Fails** (Admin only):
   - Check that the WebSocket server is running on the default port (8765)
   - Verify your admin credentials

4. **Authentication Issues**:
   - For users: Check that your account is active and credentials are correct
   - For admins: The default admin username may be 'admin@lightberry.local'

5. **Connection Issues**:
   - Verify your network connection
   - Check that the server URL is correct
   - Ensure firewalls aren't blocking the connection 