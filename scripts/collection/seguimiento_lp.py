#!/usr/bin/env python3
"""
scripts/collection/seguimiento_lp.py
====================================
Historial de LP medido por nosotros, para validar la interpolación del elo al revés con
LP real en el medio (PROJECT_SUMMARY.md §9.S.21, "Seguimiento de LP propio"). Reemplaza a lolpros.gg, que
corta con 429 a los ~10 pedidos y pide ~5 min de espera.

Dos fuentes, en `data/external/lp_seguimiento/` (dataset aparte: no toca `lol_las.db`):

* `apex.csv`: Challenger, Grandmaster y Master completos de cada región (3 pedidos por
  región), cada `--cada-apex` minutos. Se guarda una fila solo cuando cambia el estado
  (tier, división, LP, W, L) de una cuenta, y una fila `FUERA` cuando sale de Master+.
  El estado entre dos filas es el de la anterior: lo que garantiza `pasadas.csv`.
* `alto_volumen.csv`: las `--seguidos` cuentas por región que más partidas juegan,
  consultadas una por una (league-v4 entries/by-puuid) cada `--cada-seguidos` minutos.
  Con una foto cada 10 min y partidas de ~30 min, casi cada partida queda aislada
  (ΔW+ΔL = 1): el cambio de LP es el de esa partida. Se siguen aunque bajen de Master.
  La lista (`seguidos.csv`) se rehace cada `--reelegir-horas`: por partidas por día
  medidas en `apex.csv` cuando hay `MIN_HORAS_RITMO` de historia; antes, por W+L de la
  temporada.
* `pasadas.csv`: cada pedido de ladder o ronda de seguidos, con su hora y cuántas cuentas
  respondieron. Sin pasada no hay dato: un hueco ahí es un hueco en la medición.

Todo con la key de P0: los PUUID vienen cifrados por proyecto y mezclar keys partiría
cada cuenta en dos. Toda petición pasa por `RiotRateLimiter` (AGENTS.md §3.B); cada región
tiene su propia cuota en Riot, así que EUW y NA no le quitan nada al crawler de LAS.

Uso:
    python scripts/collection/seguimiento_lp.py                  # sin fin; Ctrl+C y retomar
    python scripts/collection/seguimiento_lp.py --horas 72
    python scripts/collection/seguimiento_lp.py --una-pasada     # prueba: una vuelta de cada cosa
"""

import argparse
import csv
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.config import APEX_TIERS, DATA_DIR, RIOT_API_KEYS, project_index_for_key  # noqa: E402

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

OUT = DATA_DIR / "external" / "lp_seguimiento"
REGIONES = ("la2", "euw1", "na1")
CAMPOS = ["region", "puuid", "visto_at", "tier", "rank", "league_points", "wins", "losses"]
ESTADO = ["tier", "rank", "league_points", "wins", "losses"]
FUERA = "FUERA"                 # salió de Master+ (o se dejó de ver en el ladder)
MIN_HORAS_RITMO = 24            # historia mínima de apex.csv para elegir por partidas/día
VENTANA_RITMO = timedelta(days=3)


def ahora_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


class Registro:
    """CSV de solo agregar con el último estado de cada (región, puuid): escribe solo cambios.

    Al abrir lee el CSV existente, así que cortar y volver a correr no duplica filas.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.ultimo = {}
        if self.path.exists():
            with self.path.open(newline="", encoding="utf-8") as f:
                for fila in csv.DictReader(f):
                    self.ultimo[(fila["region"], fila["puuid"])] = self._estado(fila)

    @staticmethod
    def _estado(fila) -> tuple:
        return tuple("" if fila.get(c) is None else str(fila[c]) for c in ESTADO)

    def anotar(self, region: str, filas, visto_at: str) -> int:
        """Agrega las filas cuyo estado cambió. `filas`: dicts con puuid y ESTADO."""
        nuevas = []
        for f in filas:
            clave = (region, f["puuid"])
            est = self._estado(f)
            if self.ultimo.get(clave) != est:
                self.ultimo[clave] = est
                nuevas.append({"region": region, "puuid": f["puuid"], "visto_at": f.get("visto_at") or visto_at,
                               **dict(zip(ESTADO, est))})
        self._escribir(nuevas)
        return len(nuevas)

    def anotar_salidas(self, region: str, presentes: set, visto_at: str) -> int:
        """Una fila FUERA por cada cuenta de la región que estaba en el ladder y ya no aparece."""
        fuera = [{"puuid": p, "tier": FUERA, "rank": "", "league_points": "", "wins": "", "losses": ""}
                 for (r, p), est in self.ultimo.items() if r == region and est[0] != FUERA and p not in presentes]
        return self.anotar(region, fuera, visto_at)

    def _escribir(self, filas) -> None:
        if not filas:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        nuevo = not self.path.exists()
        with self.path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=CAMPOS)
            if nuevo:
                w.writeheader()
            w.writerows(filas)


def agregar_csv(path: Path, fila: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nuevo = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fila))
        if nuevo:
            w.writeheader()
        w.writerow(fila)


def partidas_por_dia(cambios: pd.DataFrame, desde: pd.Timestamp, hasta: pd.Timestamp) -> pd.Series:
    """Partidas por día de cada (región, puuid) entre `desde` y `hasta`, desde `apex.csv`.

    El conteo al inicio es el último estado visto hasta `desde` (el estado se mantiene
    entre filas); si la cuenta entró al ladder después, su primera fila en la ventana.
    """
    c = cambios[cambios["tier"] != FUERA].copy()
    c["visto_at"] = pd.to_datetime(c["visto_at"])
    c = c[c["visto_at"] <= hasta].sort_values("visto_at")
    c["n"] = pd.to_numeric(c["wins"]) + pd.to_numeric(c["losses"])
    claves = ["region", "puuid"]
    fin = c.groupby(claves)["n"].last()
    inicio = c[c["visto_at"] <= desde].groupby(claves)["n"].last()
    inicio = inicio.reindex(fin.index).fillna(c[c["visto_at"] > desde].groupby(claves)["n"].first())
    dias = max((hasta - desde) / pd.Timedelta(days=1), 1e-9)
    return ((fin - inicio) / dias).rename("partidas_dia")


def elegir_alto_volumen(cambios: pd.DataFrame, primera_pasada: dict, ahora: pd.Timestamp, k: int) -> pd.DataFrame:
    """Las `k` cuentas de cada región que más juegan, con el criterio usado.

    `primera_pasada[región]`: hora de la primera pasada de ladder completa. Con menos de
    `MIN_HORAS_RITMO` de historia se ordena por W+L de la temporada (proxy grueso: premia
    a quien jugó mucho al principio y dejó).
    """
    elegidos = []
    for region, g in cambios[cambios["tier"] != FUERA].groupby("region"):
        inicio = primera_pasada.get(region)
        if inicio is not None and ahora - inicio >= pd.Timedelta(hours=MIN_HORAS_RITMO):
            valor = partidas_por_dia(g, max(inicio, ahora - VENTANA_RITMO), ahora)
            criterio = "partidas_dia"
        else:
            u = g.sort_values("visto_at").groupby("puuid").last()
            valor = pd.to_numeric(u["wins"]) + pd.to_numeric(u["losses"])
            valor.index = pd.MultiIndex.from_product([[region], valor.index], names=["region", "puuid"])
            criterio = "partidas_temporada"
        top = valor.sort_values(ascending=False, kind="stable").head(k)
        elegidos.append(pd.DataFrame({"region": region, "puuid": top.index.get_level_values("puuid"),
                                      "criterio": criterio, "valor": top.values}))
    return pd.concat(elegidos, ignore_index=True) if elegidos else pd.DataFrame(
        columns=["region", "puuid", "criterio", "valor"])


class Seguimiento:
    def __init__(self, regiones, k: int, out: Path = OUT):
        from src.riot_client import RiotAPIClient
        p0 = next((key for key in RIOT_API_KEYS if project_index_for_key(key) == 0), None)
        if not p0:
            sys.exit("[X] Hace falta la key de P0 en el .env (los PUUID quedan cifrados con P0).")
        self.clientes = {r: RiotAPIClient(api_keys=[p0], project_indices=[0], platform=r) for r in regiones}
        self.k = k
        self.out = Path(out)
        self.apex = Registro(self.out / "apex.csv")
        self.alto = Registro(self.out / "alto_volumen.csv")
        self.seguidos = self._seguidos_vigentes()

    def _seguidos_vigentes(self) -> dict:
        f = self.out / "seguidos.csv"
        if not f.exists():
            return {}
        s = pd.read_csv(f)
        s = s[s["elegido_at"] == s.groupby("region")["elegido_at"].transform("max")]
        return {r: list(g["puuid"]) for r, g in s.groupby("region")}

    def pasada_apex(self) -> None:
        for region, cli in self.clientes.items():
            presentes, completa, t0, n_cambios = {}, True, time.time(), 0
            for tier in APEX_TIERS:
                visto = ahora_utc()
                filas = cli.get_apex_league(tier)
                agregar_csv(self.out / "pasadas.csv", {"visto_at": visto, "region": region, "fuente": tier,
                                                       "consultadas": 1, "respondidas": len(filas)})
                if not filas:          # Master nunca está vacío: lista vacía es falla del pedido
                    completa = False
                    continue
                for f in filas:
                    f["visto_at"] = visto
                    presentes[f["puuid"]] = f
            n_cambios += self.apex.anotar(region, presentes.values(), ahora_utc())
            if completa:
                n_cambios += self.apex.anotar_salidas(region, set(presentes), ahora_utc())
            print(f"  [{ahora_utc()}] {region}: {len(presentes):,} en Master+, {n_cambios:,} cambios"
                  f"{'' if completa else ' (pasada incompleta)'} ({time.time() - t0:.0f} s)", flush=True)

    def reelegir(self) -> None:
        f = self.out / "apex.csv"
        if not f.exists():
            return
        cambios = pd.read_csv(f, dtype={"rank": str})
        p = pd.read_csv(self.out / "pasadas.csv")
        p = p[(p["fuente"].isin(APEX_TIERS)) & (p["respondidas"] > 0)]
        primera = {r: pd.to_datetime(g["visto_at"]).min() for r, g in p.groupby("region")}
        ahora = pd.Timestamp(ahora_utc())
        e = elegir_alto_volumen(cambios[cambios["region"].isin(list(self.clientes))], primera, ahora, self.k)
        e["elegido_at"] = str(ahora)
        e.to_csv(self.out / "seguidos.csv", mode="a", header=not (self.out / "seguidos.csv").exists(), index=False)
        self.seguidos = {r: list(g["puuid"]) for r, g in e.groupby("region")}
        for r, g in e.groupby("region"):
            print(f"  [{ahora_utc()}] {r}: {len(g)} seguidos por {g['criterio'].iat[0]} "
                  f"(de {g['valor'].min():.1f} a {g['valor'].max():.1f})", flush=True)

    def pasada_seguidos(self) -> None:
        for region, puuids in self.seguidos.items():
            cli = self.clientes.get(region)
            if cli is None:
                continue
            t0, filas = ahora_utc(), []
            for p in puuids:
                r = cli.get_soloq_entry_by_puuid({0: p})
                if r is not None:
                    filas.append({**r, "visto_at": ahora_utc()})
            n = self.alto.anotar(region, filas, ahora_utc())
            agregar_csv(self.out / "pasadas.csv", {"visto_at": t0, "region": region, "fuente": "seguidos",
                                                   "consultadas": len(puuids), "respondidas": len(filas)})
            print(f"  [{ahora_utc()}] {region}: {len(filas)}/{len(puuids)} seguidos, {n} cambios", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Historial de LP de Master+ y de cuentas de alto volumen")
    ap.add_argument("--regiones", default=",".join(REGIONES))
    ap.add_argument("--cada-apex", type=float, default=60, help="minutos entre fotos del ladder completo")
    ap.add_argument("--cada-seguidos", type=float, default=10, help="minutos entre rondas de alto volumen")
    ap.add_argument("--seguidos", type=int, default=50, help="cuentas de alto volumen por región")
    ap.add_argument("--reelegir-horas", type=float, default=6)
    ap.add_argument("--horas", type=float, default=0, help="duración; 0 = sin fin")
    ap.add_argument("--una-pasada", action="store_true", help="ladder, elección y una ronda de seguidos; y sale")
    args = ap.parse_args()

    s = Seguimiento([r.strip().lower() for r in args.regiones.split(",") if r.strip()], args.seguidos)
    fin = time.time() + args.horas * 3600 if args.horas else float("inf")
    prox = {"apex": 0.0, "elegir": 0.0, "seguidos": 0.0}
    print(f"[*] Seguimiento de LP en {', '.join(s.clientes)} -> {s.out}", flush=True)
    try:
        while time.time() < fin:
            t = time.time()
            if t >= prox["apex"]:
                s.pasada_apex()
                prox["apex"] = t + args.cada_apex * 60
            if t >= prox["elegir"] or not s.seguidos:
                s.reelegir()
                prox["elegir"] = t + args.reelegir_horas * 3600
            if t >= prox["seguidos"]:
                s.pasada_seguidos()
                prox["seguidos"] = t + args.cada_seguidos * 60
            if args.una_pasada:
                break
            time.sleep(max(1.0, min(min(prox.values()), fin) - time.time()))
    except KeyboardInterrupt:
        print("\n[i] Cortado: lo medido ya está en disco; volver a correr retoma.")


if __name__ == "__main__":
    main()
