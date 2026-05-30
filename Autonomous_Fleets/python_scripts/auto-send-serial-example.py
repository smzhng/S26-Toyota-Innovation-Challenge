import serial
import sys
import tty
import termios
import select
import time

ser = serial.Serial('/dev/tty.usbserial-AQ00YYW4', 9600)

print("w=forward  s=backward  a=left  d=right")
print("1=open gripper  2=close gripper")
print("SPACE or x = stop    q = quit")

fd = sys.stdin.fileno()
old_settings = termios.tcgetattr(fd)
current_cmd = b'x'

try:
    tty.setraw(fd)
    while True:
        # Check for new keypress (non-blocking)
        if select.select([sys.stdin], [], [], 0)[0]:
            ch = sys.stdin.read(1)
            if ch in ('q', '\x03'):
                break
            elif ch in (' ', 'x'):
                current_cmd = b'x'
            elif ch in ('w', 'a', 's', 'd', '1', '2'):
                current_cmd = ch.encode()

        ser.write(current_cmd)
        time.sleep(0.04)
finally:
    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    ser.write(b'x')
    ser.close()
    print("\nStopped.")
