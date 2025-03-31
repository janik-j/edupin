# Lightberry Client

Voice assistant client for the Lightberry project. This client handles audio recording, wake word detection, audio streaming to server, and playback of server responses.

## Features

- Wake word detection using Picovoice Porcupine
- Real-time audio streaming over WebSockets
- Cross-platform compatibility (macOS, Windows, Raspberry Pi)
- Automatic reconnection to server
- High-quality audio playback
- Public/private key device authentication

## Installation

### macOS / Windows

1. Clone this repository
2. Install the requirements:
   ```
   pip install -r requirements.txt
   ```
3. Install platform-specific dependencies:
   - macOS: `pip install pynput`

### Raspberry Pi

1. Clone this repository
2. Install dependencies:
   ```
   sudo apt-get update
   sudo apt-get install python3-pyaudio
   pip3 install -r requirements.txt
   ```

## Configuration

1. Create a `.env` file with your configuration:
   ```
   AUDIO_SERVER_IP=your_server_ip
   AUDIO_SERVER_PORT=8765
   PICOVOICE_API_KEY=your_picovoice_key
   ```

2. Optional: If you've created a custom wake word model, place it in the `models` directory.

## Usage

### Standard Mode

```
python audio_client.py
```

### With Custom Server

```
python audio_client.py --server your_server_ip --port 8765
```

### With Custom Wake Word

```
python audio_client.py --wake-word models/your_wake_word_model.ppn
```

### Debug Mode (Test Audio Files)

```
python audio_client.py --debug --test-dir test_audio
```

## How It Works

1. Client listens for wake word (default: "Hey Berry")
2. Once wake word is detected, client starts recording audio
3. Audio is streamed in real-time to the server
4. Server processes audio, generates response, and streams back
5. Client plays the response audio

## Troubleshooting

- Use `--debug` flag to test with pre-recorded audio files
- Check connection using `ping` to ensure server is reachable
- Run `python audio_client.py` with `--register` flag to register device with server

## License

[MIT License](LICENSE) 