echo Toggle MistyCarrier Debug Pad pin TP48
gpioset --mode=time --sec=1 `gpiofind "P11_1"`=1
