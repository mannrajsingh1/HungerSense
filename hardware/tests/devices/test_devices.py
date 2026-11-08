#!/usr/bin/env python3

import time
from smbus2 import SMBus

BUS = 1
HAT_ADDRESS = 0x04

REG_RAW_DATA_START = 0x10
REG_VOLTAGE_START = 0x20

# Physical Grove analog sockets on this HAT
CHANNELS = [0, 2, 4, 6]


def read_u16(bus, register):
    data = bus.read_i2c_block_data(HAT_ADDRESS, register, 2)
    return (data[1] << 8) | data[0]


def main():
    print("Grove Base HAT tester")
    print("---------------------")

    with SMBus(BUS) as bus:
        try:
            # Simple communication test with ADC channel 0
            read_u16(bus, REG_RAW_DATA_START)
            print("✓ Grove Base HAT detected at I2C address 0x04")
        except Exception as e:
            print("✗ Grove Base HAT not responding")
            print(e)
            return

        print("\nReading analog ports. Ctrl+C to stop.\n")

        try:
            while True:
                values = []

                for channel in CHANNELS:
                    raw = read_u16(
                        bus,
                        REG_RAW_DATA_START + channel
                    )

                    millivolts = read_u16(
                        bus,
                        REG_VOLTAGE_START + channel
                    )

                    values.append(
                        f"A{channel}: raw={raw:4d}  voltage={millivolts:4d} mV"
                    )

                print(" | ".join(values))
                time.sleep(0.25)

        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
