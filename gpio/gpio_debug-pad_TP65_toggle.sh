echo Toggle MistyCarrier Debug Pad pin TP65
gpioset --mode=time --sec=1 `gpiofind "P6_1"`=1
