echo Toggle MistyCarrier Debug Pad pin TP43
gpioset --mode=time --sec=1 `gpiofind "P9_0"`=1
