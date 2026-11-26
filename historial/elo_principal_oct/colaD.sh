PYTHONIOENCODING=utf-8 python -u tabpfn_exp.py --max-train 12000 --estimadores 8 --grid grid_l2.0 --tag tabpfn_g2_grande < /dev/null 2>&1 | grep -v -i warn | tail -2
PYTHONIOENCODING=utf-8 python -u grilla.py --tag grid_l4.0 --tau 4.0 2>&1 | tail -1
PYTHONIOENCODING=utf-8 python gb_grid.py grid_l4.0 cat_huber 2>&1 | grep -v Warn
