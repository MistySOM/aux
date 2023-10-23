#!/bin/sh


date > tick_start
while true
	do
	date > tick_last
	cat /tmp/tick
	sleep 10
done
