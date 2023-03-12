echo Toggle MistyCarrier Debug Pad pin TP47
gpioset --mode=time --sec=1 `gpiofind "P11_0"`=1
