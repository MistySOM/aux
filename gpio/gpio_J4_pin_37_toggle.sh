echo Toggle MistyCarrier J4 pin 37
gpioset --mode=time --sec=1 `gpiofind "P1_0"`=1
