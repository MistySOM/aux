echo Toggle MistyCarrier Debug Pad pin TP66
gpioset --mode=time --sec=1 `gpiofind "P7_0"`=1
