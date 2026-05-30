import serial
import keyboard
import time

ser = serial.Serial('COM6', 9600)

while True:
    if keyboard.is_pressed('w'):
        ser.write(b'w')
    elif keyboard.is_pressed('a'):
        ser.write(b'a')
    elif keyboard.is_pressed('s'):
        ser.write(b's')
    elif keyboard.is_pressed('d'):
        ser.write(b'd')
    elif keyboard.is_pressed('x'):
        ser.write(b'x')
    elif keyboard.is_pressed('1'):
        ser.write(b'1')
    elif keyboard.is_pressed('2'):
        ser.write(b'2')

    time.sleep(0.05)