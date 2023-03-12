echo Toggle MistyCarrier Debug Pad pin TP59
gpioset --mode=time --sec=1 `gpiofind "P16_1"`=1
