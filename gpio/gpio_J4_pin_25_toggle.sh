echo Toggle MistyCarrier J4 pin 25
gpioset --mode=time --sec=1 `gpiofind "P48_4"`=1
