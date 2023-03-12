echo Toggle MistyCarrier J4 pin 35
gpioset --mode=time --sec=1 `gpiofind "P0_1"`=1
