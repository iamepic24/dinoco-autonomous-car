#!/bin/bash
# One-time setup on a freshly flashed Pi. Run from inside ~/uamc_pi:  bash setup.sh
set -e
sudo apt update
sudo apt install -y python3-opencv python3-serial python3-numpy tesseract-ocr
pip3 install --break-system-packages pytesseract
sudo usermod -aG dialout,video "$USER"
sudo mkdir -p /var/log/journal && sudo systemctl restart systemd-journald
python3 -c "import cv2, serial, pytesseract; print('OpenCV', cv2.__version__, '| Tesseract', pytesseract.get_tesseract_version())"
python3 -m py_compile *.py
echo "Setup done. Log out and back in (or reboot) once, then run: python3 check.py ports"
