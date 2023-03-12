echo Toggle MistyCarrier Debug Pad pin TP55
gpioset --mode=time --sec=1 `gpiofind "P14_1"`=1
