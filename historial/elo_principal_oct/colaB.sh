for t in 1.0 2.0; do
  PYTHONIOENCODING=utf-8 python -u grilla.py --tag grid_l$t --tau $t 2>&1 | tail -1
  PYTHONIOENCODING=utf-8 python gb_grid.py grid_l$t cat_huber 2>&1 | grep -v Warn
done
PYTHONIOENCODING=utf-8 python -u tabpfn_exp.py --max-train 8000 < /dev/null 2>&1 | grep -v -i warn | tail -3
PYTHONIOENCODING=utf-8 python -u grilla.py --tag grid_l4.0 --tau 4.0 2>&1 | tail -1
PYTHONIOENCODING=utf-8 python gb_grid.py grid_l4.0 cat_huber 2>&1 | grep -v Warn
