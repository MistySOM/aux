echo Toggle MistyCarrier Debug Pad pin TP57
gpioset --mode=time --sec=1 `gpiofind "P15_1"`=1
