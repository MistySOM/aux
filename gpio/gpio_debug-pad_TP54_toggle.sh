echo Toggle MistyCarrier Debug Pad pin TP54
gpioset --mode=time --sec=1 `gpiofind "P14_0"`=1
