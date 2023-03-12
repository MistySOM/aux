echo Toggle MistyCarrier Debug Pad pin TP41
gpioset --mode=time --sec=1 `gpiofind "P8_1"`=1
