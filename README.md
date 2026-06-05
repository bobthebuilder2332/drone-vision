# Drone Vision Project

This is a very simple project that utilizes computer vision to control a DJI Tello Talent drone

## Features

* Keyboard control
* Hand gesture control
* ArUco tag control
* Self centering landing using ArUco tag
* Live video feed from drone

## How to setup and use (Bash)

1. Install latest version of Python 3
2. ```bash
   git clone https://github.com/bobthebuilder2332/drone-vision.git
   ```
3. ```bash
   cd drone-vision
   ```
4. ```bash
   python -m venv .venv
   ```
5. ```bash
   source .venv/bin/activate
   ```
   * Use this if in Windows Command Prompt
     ```bash
     venv\Scripts\activate.bat
     ```
6. ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
7. Run the main.py file


Distributed under MIT license
