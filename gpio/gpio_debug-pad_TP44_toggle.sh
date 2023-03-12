echo Toggle MistyCarrier Debug Pad pin TP44
gpioset --mode=time --sec=1 `gpiofind "P9_1"`=1
