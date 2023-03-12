echo Toggle MistyCarrier J4 pin 40
gpioset --mode=time --sec=1 `gpiofind "P48_3"`=1
