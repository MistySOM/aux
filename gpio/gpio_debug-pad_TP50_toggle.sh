echo Toggle MistyCarrier Debug Pad pin TP50
gpioset --mode=time --sec=1 `gpiofind "P12_1"`=1
