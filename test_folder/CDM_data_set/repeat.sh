#!/bin/sh

a=0

while [ $a -lt 10 ]
do
    python3 local.py
    a=`expr $a + 1`
done