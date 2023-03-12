echo Toggle MistyCarrier Debug Pad pin TP61
gpioset --mode=time --sec=1 `gpiofind "P17_1"`=1
