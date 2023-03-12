echo Toggle MistyCarrier Debug Pad pin TP39
gpioset --mode=time --sec=1 `gpiofind "P7_2"`=1
