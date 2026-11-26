PYTHONIOENCODING=utf-8 python -u tabpfn_exp.py --max-train 8000 --caja < /dev/null 2>&1 | grep -v -i warn | tail -2
PYTHONIOENCODING=utf-8 python -u tabpfn_exp.py --max-train 8000 --grid grid_l2.0 --tag tabpfn_g2 < /dev/null 2>&1 | grep -v -i warn | tail -2
PYTHONIOENCODING=utf-8 python -u tabpfn_exp.py --max-train 8000 --grid grid_l2.0 --tag tabpfn_g2 --caja < /dev/null 2>&1 | grep -v -i warn | tail -2
