echo Toggle MistyCarrier Debug Pad pin TP60
gpioset --mode=time --sec=1 `gpiofind "P17_0"`=1
