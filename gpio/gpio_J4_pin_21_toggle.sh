echo Toggle MistyCarrier J4 pin 21
gpioset --mode=time --sec=1 `gpiofind "P48_1"`=1
