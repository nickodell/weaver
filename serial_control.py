import serial

SERIAL_PORT = "/dev/cu.usbmodem1301"
BAUD_RATE = 115200


def init_serial(port=SERIAL_PORT, baudrate=BAUD_RATE):
    return serial.Serial(port=port, baudrate=baudrate, timeout=1)


def send_command(ser, command):
    ser.write(f"{command}\n".encode("utf-8"))
    while True:
        raw_data = ser.readline()
        if raw_data == b"READY\r\n":
            break


def send_pin(ser, pin):
    send_command(ser, f"pin {pin}")


def send_home(ser):
    send_command(ser, "home")
