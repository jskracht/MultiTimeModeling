#!/bin/bash
while read FILE; do
	echo $FILE
	sed -e 's/,\./,/g' -i $FILE
done <./FRED2/index