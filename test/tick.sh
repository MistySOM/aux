#!/bin/sh


date > /tmp/tick_start
while true
	do
	date > /tmp/tick
	cat /tmp/tick
	sleep 10
done
