echo Toggle MistyCarrier Debug Pad pin TP42
gpioset --mode=time --sec=1 `gpiofind "P8_2"`=1
