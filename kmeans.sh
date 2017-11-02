#! /usr/bin/env bash
case $1 in 
-s) python serialK.py;;	
-p) mpiexec -n $2 python parallelK.py;;
*) echo error: command not defined;;
esac 