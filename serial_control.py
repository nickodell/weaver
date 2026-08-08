import serial

SERIAL_PORT = "/dev/cu.usbmodem1301"
BAUD_RATE = 115200


def init_serial(port=SERIAL_PORT, baudrate=BAUD_RATE):
    return serial.Serial(port=port, baudrate=baudrate, timeout=1)


def send_pin(ser, pin):
    ser.write(f"pin {pin}\n".encode("utf-8"))
    while True:
        raw_data = ser.readline()
        if raw_data == b"READY\r\n":
            break
