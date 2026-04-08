## Netra: Real-Time Driver Drowsiness Detection

Netra is a real-time driver drowsiness detection system. A Python application running on a PC processes a webcam or video stream using OpenCV and dlib (68 facial landmarks), computes the Eye Aspect Ratio (EAR), and sends drowsiness alerts over UART to an ESP32 microcontroller. The ESP32 then drives a 5V active buzzer.

### Repository Layout

- `netra_vision.py`: Main real-time vision app (webcam or .mp4 file) with serial signaling to ESP32.
- `evaluate_datasets.py`: Headless evaluation script to run the same pipeline on datasets and compute confusion-matrix metrics.
- `netra_common.py`: Shared utilities for loading dlib models, EAR computation, and drowsiness state logic.
- `esp32/esp32_buzzer.ino`: ESP32 firmware that drives a buzzer based on serial bytes from the PC.
- `models/shape_predictor_68_face_landmarks.dat`: Dlib facial landmark model (not tracked in git; you must download it).
- `datasets/`: Local-only folder containing evaluation datasets (NITYMED, SUST, etc.).
- `requirements.txt`: Python dependencies.

### Python Environment Setup

1. Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
.venv\Scripts\activate  # on Windows
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download the dlib 68 facial landmark model:

- Download `shape_predictor_68_face_landmarks.dat` from the official dlib model repository (e.g. the dlib website).
- Create a `models` folder at the project root (if it does not exist).
- Place the file at:

```text
models/shape_predictor_68_face_landmarks.dat
```

### Running the Real-Time Application

Webcam input with ESP32 connected (replace `COM3` with your serial port):

```bash
python netra_vision.py --source webcam --camera-index 0 --serial-port COM3
```

Testing from a video file without hardware:

```bash
python netra_vision.py --source file --video-path path/to/video.mp4 --no-serial
```

Press `q` in the video window to quit.

### ESP32 Firmware (Buzzer Control)

The firmware in `esp32/esp32_buzzer.ino`:

- Initializes `Serial` at 9600 baud.
- Sets a GPIO pin as output for the 5V active buzzer.
- Reads bytes from serial:
  - `'1'` → buzzer pin HIGH (buzzer ON).
  - `'0'` → buzzer pin LOW (buzzer OFF).

You can flash the sketch to your ESP32 using the Arduino IDE:

- Select the appropriate ESP32 board and COM port.
- Open `esp32/esp32_buzzer.ino`.
- Click **Upload**.

Make sure the buzzer is connected between the selected GPIO pin (through a suitable driver or transistor if needed) and ground/5V according to your hardware design.

### Dataset Structure and Evaluation

Place your evaluation datasets under a `datasets` directory with the following structure:

- `datasets/NITYMED/microsleeps/*.mp4` (videos with yawning/microsleeps → drowsy, label 1)
- `datasets/NITYMED/normal/*.mp4` (normal/awake driving → label 0)
- `datasets/SUST/drowsy/*.mp4` (videos labeled drowsy → label 1)
- `datasets/SUST/normal/*.mp4` (videos labeled not drowsy → label 0)

Run the headless evaluation as:

```bash
python evaluate_datasets.py --root-dir datasets
```

This will:

- Run the same EAR + drowsiness state machine on each `.mp4` file.
- Treat a video as positive if it ever enters the drowsy state.
- Print a basic confusion matrix (TP, FN, FP, TN) and simple derived metrics.

### Notes

- If the ESP32 is not connected or the serial port cannot be opened, `netra_vision.py` will continue to run in a test mode and log what it would send instead of failing.
- The drowsiness logic uses:
  - `EAR_THRESHOLD = 0.25`
  - `CONSEC_FRAMES_DROWSY = 20`

