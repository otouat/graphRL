#!/bin/sh

for file in /config;do
  python run_exp.py -c $file
done