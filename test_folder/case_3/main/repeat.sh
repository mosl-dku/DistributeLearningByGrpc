#!/bin/sh

a=0

while [ $a -lt 50 ]
do
    python3 request.py
    a=`expr $a + 1`
done