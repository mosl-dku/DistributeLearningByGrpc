#!/bin/sh

a=0

while [ $a -lt 5 ]
do
    python3 cifar.py
    a=`expr $a + 1`
done