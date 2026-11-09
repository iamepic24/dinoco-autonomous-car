UAMC car code, full version as of 8 Nov 2026 (everything since uamc_pi_v2 included).

Install on a fresh Pi:
  cp /boot/firmware/uamc_pi_final.zip ~/ && cd ~ && unzip -o uamc_pi_final.zip && cd uamc_pi && bash setup.sh
Then reboot once.

Settings saved here are the last ones used outdoors (white lines on dark road):
  LINE_COLOR ""  (white lines)      ROW_NEAR/FAR 0.40/0.20    BASE_THROTTLE 0.35
  CRUISE_SPEED_MPS 0.12 (guess)     TURN_DELAY_M 0.10         COAST_M 0.15
  CAMERA_F_PX 700   WORD_HEIGHT_M 0.05 (measure your printed letters)
  SIGN_WAIT_FOR_MOVE False   SEARCH_TURN_S 6 (turns right when lane is lost)
For red tape indoors: LINE_COLOR = "red", ROW_NEAR 0.50, ROW_FAR 0.22.

After any check, the USB ports in config.py must match `python3 check.py ports`.
