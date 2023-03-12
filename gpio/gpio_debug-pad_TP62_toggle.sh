echo Toggle MistyCarrier Debug Pad pin TP62
gpioset --mode=time --sec=1 `gpiofind "P17_2"`=1
