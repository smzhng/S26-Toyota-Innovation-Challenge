import serial
import sys
import tty
import termios
import select
import time

PORT = '/dev/tty.usbserial-AQ00YYW4'
BAUD = 9600

ser = serial.Serial(PORT, BAUD)

print("""
============ TELEOP TEST ============
  w  = forward       s = backward
  a  = turn left     d = turn right
  1  = open gripper  2 = close gripper
  SPACE / x = stop    q = quit
=====================================
Hold a key to move continuously.
""")

LABELS = {
    'w': 'forward', 's': 'backward',
    'a': 'turn left', 'd': 'turn right',
    '1': 'gripper OPEN', '2': 'gripper CLOSE',
    'x': 'stopped',
}

INITIAL_HOLD = 0.70   # covers macOS key-repeat initial delay (~500ms)
REPEAT_HOLD  = 0.15   # window between key repeats (~30-50ms on macOS)

fd = sys.stdin.fileno()
old_settings = termios.tcgetattr(fd)

current_cmd   = 'x'
last_key_time = 0.0
key_first_time = 0.0
last_label    = ''

try:
    tty.setraw(fd)
    while True:
        now = time.time()

        if select.select([sys.stdin], [], [], 0)[0]:
            ch = sys.stdin.read(1)
            if ch in ('q', '\x03'):
                break
            elif ch in (' ', 'x'):
                current_cmd    = 'x'
                last_key_time  = 0.0
                key_first_time = 0.0
            elif ch in LABELS:
                if ch != current_cmd:
                    key_first_time = now
                current_cmd   = ch
                last_key_time = now

        # Decide whether key is still held
        since_first = now - key_first_time
        since_last  = now - last_key_time

        if current_cmd != 'x' and (since_first < INITIAL_HOLD or since_last < REPEAT_HOLD):
            cmd_to_send = current_cmd
        else:
            cmd_to_send = 'x'
            current_cmd = 'x'

        label = LABELS.get(cmd_to_send, 'stopped')
        if label != last_label:
            sys.stdout.write(f'\r  >> {label:<20}')
            sys.stdout.flush()
            last_label = label

        ser.write(cmd_to_send.encode())
        time.sleep(0.04)

finally:
    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    ser.write(b'x')
    ser.close()
    print("\nStopped.")
