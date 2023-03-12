echo Toggle MistyCarrier Debug Pad pin TP52
gpioset --mode=time --sec=1 `gpiofind "P13_1"`=1
