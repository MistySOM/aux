echo Toggle MistyCarrier Debug Pad pin TP45
gpioset --mode=time --sec=1 `gpiofind "P10_0"`=1
