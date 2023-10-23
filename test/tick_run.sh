#!/bin/sh


date > tick_start
while true
	do
	date > tick_last
	cat tick_last
	sleep 10
done
