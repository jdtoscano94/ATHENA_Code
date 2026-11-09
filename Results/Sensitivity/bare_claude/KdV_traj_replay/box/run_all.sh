#!/bin/bash
# Reproduce the main result (~38 min on one H100) and its evaluation.
python march.py 10 '{}' A > march_A.log 2>&1
python evaluate.py A .
# optional: second seed
# python march.py 10 '{"seed":7}' A_s7 > march_A_s7.log 2>&1; python evaluate.py A_s7 .
