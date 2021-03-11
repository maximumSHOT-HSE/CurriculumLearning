#!/bin/bash

for ((i=231857; i<231873; i+=1))
do
	echo "$i"
	scancel "$i"
done
