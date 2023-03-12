echo Toggle MistyCarrier Debug Pad pin TP56
gpioset --mode=time --sec=1 `gpiofind "P15_0"`=1
