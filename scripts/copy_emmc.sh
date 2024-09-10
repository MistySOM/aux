# Script to copy the root partition from SDcard to eMMC memory
# date 08/02/2024
# author: Matin

#!/bin/bash

mount /dev/mmcblk0p1 /mnt
rsync -aAHXx --partial --info=progress2 --numeric-ids / /mnt
