echo Toggle MistyCarrier J4 pin 33
gpioset --mode=time --sec=1 `gpiofind "P0_0"`=1
