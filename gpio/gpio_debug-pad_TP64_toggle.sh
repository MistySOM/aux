echo Toggle MistyCarrier Debug Pad pin TP64
gpioset --mode=time --sec=1 `gpiofind "P7_1"`=1
