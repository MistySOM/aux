# Script to copy the root partition from SDcard to eMMC memory
# date 08/02/2024
# author: Matin

#!/bin/bash

rsync -aAHXx --partial --info=progress2 --numeric-ids / /mnt
