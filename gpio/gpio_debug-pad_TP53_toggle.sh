echo Toggle MistyCarrier Debug Pad pin TP53
gpioset --mode=time --sec=1 `gpiofind "P13_2"`=1
