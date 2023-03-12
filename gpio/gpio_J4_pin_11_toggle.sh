echo Toggle MistyCarrier J4 pin 11
gpioset --mode=time --sec=1 `gpiofind "P42_3"`=1
