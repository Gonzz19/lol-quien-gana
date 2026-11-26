#!/usr/bin/env python3
"""
scripts/collection/lolpros_historial.py
=======================================
Historial de LP de cuentas seguidas por lolpros.gg, para validar la interpolación del
elo al revés con LP real en el medio (PROJECT_SUMMARY.md §9.S.21, TAREAS.md).

lolpros.gg actualiza cada cuenta 4 veces por día (cada 6 h) y guarda, por foto, tier,
LP, victorias y derrotas. No es LAS: son pros y semipros, casi todo Master+ de EUW y NA.
Sirve para medir el *método* (una pasada desde cada punta, recta, suavizado doble),
no el modelo entrenado en LAS.

Pasos (cada respuesta se guarda cruda en `data/external/lolpros/raw/`; cortar y volver a
correr retoma sin repetir pedidos):

1. `ladder`: páginas del ladder de lolpros (20 por página) → jugadores y su cuenta principal.
2. (`--todas-las-cuentas`) perfil de cada jugador → todas sus cuentas con rango.
3. `historial`: historial de cada cuenta, ~1 foto por día con tier, LP, W y L →
   `data/external/lolpros/historial.csv`.

Sin límites publicados: a 1 pedido/s corta con 429 a los ~9 pedidos, así que va a uno
cada 4 s y ante un 429 espera cada vez más (hasta 10 min). 1.000 jugadores ≈ 70 min.

La API no está documentada; los endpoints salen de lo que usa el sitio y de
`lolpros_parser`. Si alguno cambió, `--probar` muestra la estructura de cada respuesta
para ajustarlo.

Uso:
    python scripts/collection/lolpros_historial.py --probar
    python scripts/collection/lolpros_historial.py --max-jugadores 500
    python scripts/collection/lolpros_historial.py --solo-csv      # rearma el CSV desde lo crudo
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "external" / "lolpros"
RAW = OUT / "raw"

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

API = "https://api.lolpros.gg"
LADDER = API + "/es/ladder?page={page}&sort=rank&order=desc"
PERFIL = API + "/es/profiles/{slug}"
RANKINGS = API + "/lol/riot-accounts/{uuid}/rankings?months={meses}"   # el que usa el sitio (js de /player)
PAUSA_S = 4.0      # sin límites publicados ni cabeceras; a 1 pedido/s corta con 429 a los ~9 pedidos
UA = "ProyectoFinalML (UdeSA, proyecto academico; historial de LP para validacion)"


def pedir(url: str, sesion, forzar: bool = False):
    """GET con caché en disco: la misma URL no se pide dos veces. Ante 429 espera cada vez más."""
    f = RAW / (hashlib.sha1(url.encode()).hexdigest() + ".json")
    if f.exists() and not forzar:
        return json.loads(f.read_text(encoding="utf-8"))["datos"]
    for intento in range(8):
        time.sleep(PAUSA_S)
        try:
            r = sesion.get(url, timeout=60)
        except Exception as ex:   # red caída, timeout
            print(f"  [!] {url}: {ex}", flush=True)
            time.sleep(30)
            continue
        if r.status_code == 404:
            datos = None
            break
        if r.status_code in (429, 500, 502, 503, 504):
            espera = min(30 * 2 ** intento, 600)
            print(f"  [!] {r.status_code} en {url}; espero {espera} s", flush=True)
            time.sleep(espera)
            continue
        r.raise_for_status()
        datos = r.json()
        break
    else:
        raise RuntimeError(f"sin respuesta de {url}")
    f.write_text(json.dumps({"url": url, "datos": datos}, ensure_ascii=False), encoding="utf-8")
    return datos


def buscar(obj, cond):
    """Todos los dicts anidados en `obj` que cumplen `cond` (la forma exacta de la API no es fija)."""
    if isinstance(obj, dict):
        if cond(obj):
            yield obj
        for v in obj.values():
            yield from buscar(v, cond)
    elif isinstance(obj, list):
        for v in obj:
            yield from buscar(v, cond)


def esqueleto(obj, nivel=0, max_nivel=4):
    """Estructura de una respuesta (claves y tipos), para `--probar`."""
    pad = "  " * nivel
    if nivel > max_nivel:
        return pad + "..."
    if isinstance(obj, dict):
        return "\n".join(f"{pad}{k}: {type(v).__name__}" + ("\n" + esqueleto(v, nivel + 1) if isinstance(v, (dict, list)) and v else
                                                            f" = {str(v)[:40]!r}" if not isinstance(v, (dict, list)) else "")
                         for k, v in list(obj.items())[:25])
    if isinstance(obj, list):
        return f"{pad}[{len(obj)} elementos]" + ("\n" + esqueleto(obj[0], nivel + 1) if obj else "")
    return pad + repr(obj)[:60]


def slugs_de(ladder) -> list:
    """Slugs de una página del ladder: una lista de jugadores (los equipos anidados también tienen slug)."""
    return [d["slug"] for d in (ladder or []) if isinstance(d, dict) and isinstance(d.get("slug"), str)]


def cuentas_de(perfil) -> list:
    """Cuentas con rango de `league_player.accounts` (el perfil trae además equipos con uuid y servidor)."""
    cuentas = ((perfil or {}).get("league_player") or {}).get("accounts") or []
    return [c for c in cuentas if isinstance(c.get("uuid"), str) and c.get("rank")]


def filas_de(hist, cuenta: dict, slug: str) -> list:
    nombre = cuenta.get("summoner_name") or cuenta.get("riot_id") or ""
    filas = []
    for d in buscar(hist, lambda d: "league_points" in d and ("created_at" in d or "date" in d)):
        filas.append({"player_slug": slug, "account_uuid": cuenta["uuid"], "server": cuenta.get("server"),
                      "summoner_name": nombre, "created_at": d.get("created_at") or d.get("date"),
                      "tier": d.get("tier"), "rank": d.get("rank") or d.get("division"),
                      "league_points": d.get("league_points"), "wins": d.get("wins"), "losses": d.get("losses"),
                      "score": d.get("score"), "season": d.get("season")})
    return filas


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--probar", action="store_true", help="un pedido de cada tipo y su estructura; no baja nada más")
    ap.add_argument("--max-jugadores", type=int, default=500)
    ap.add_argument("--max-paginas", type=int, default=200)
    ap.add_argument("--meses", type=int, default=12, help="ventana del historial por cuenta (~1 foto por día)")
    ap.add_argument("--solo-csv", action="store_true", help="no pide nada: rearma el CSV con lo ya bajado")
    ap.add_argument("--todas-las-cuentas", action="store_true",
                    help="pide el perfil de cada jugador y baja todas sus cuentas con rango (no solo la principal)")
    args = ap.parse_args()

    try:   # certificados del sistema operativo en vez de los de certifi (antivirus/proxy con CA propia)
        import truststore
        truststore.inject_into_ssl()
    except ImportError:
        pass
    import requests
    RAW.mkdir(parents=True, exist_ok=True)
    s = requests.Session()
    s.headers.update({"User-Agent": UA, "Accept": "application/json"})

    if args.probar:
        lad = pedir(LADDER.format(page=1), s, forzar=True)
        print("== ladder página 1\n" + esqueleto(lad))
        sl = slugs_de(lad)
        print(f"\n-> {len(sl)} slugs: {sl[:5]}")
        if not sl:
            return
        per = pedir(PERFIL.format(slug=sl[0]), s, forzar=True)
        print(f"\n== perfil {sl[0]}\n" + esqueleto(per))
        cu = cuentas_de(per)
        print(f"\n-> {len(cu)} cuentas: {[(c['uuid'], c.get('server')) for c in cu]}")
        if not cu:
            return
        h = pedir(RANKINGS.format(uuid=cu[0]["uuid"], meses=args.meses), s, forzar=True)
        print(f"\n== historial {cu[0]['uuid']}\n" + esqueleto(h))
        f = filas_de(h, cu[0], sl[0])
        print(f"\n-> {len(f)} fotos; primeras: {f[:3]}")
        return

    import pandas as pd
    t0 = time.time()
    filas = []
    if args.solo_csv:   # todo lo que esté en la caché: cuentas del ladder y de los perfiles, con su historial
        cuentas, hist = {}, {}
        for f in RAW.glob("*.json"):
            j = json.loads(f.read_text(encoding="utf-8"))
            if "/ladder" in j["url"]:
                cuentas.update({d["account"]["uuid"]: (d["slug"], d["account"]) for d in (j["datos"] or [])
                                if isinstance(d, dict) and d.get("account")})
            elif "/profiles/" in j["url"]:
                cuentas.update({c["uuid"]: (j["url"].rsplit("/", 1)[1], c) for c in cuentas_de(j["datos"])})
            elif "/riot-accounts/" in j["url"]:
                hist[j["url"].split("/riot-accounts/")[1].split("/")[0]] = j["datos"]
        for uuid, (slug, c) in cuentas.items():
            if uuid in hist:
                filas += filas_de(hist[uuid] or [], c, slug)
    else:
        # Jugadores del ladder; cada entrada ya trae su cuenta principal (sin pedir el perfil).
        # Página por página: el historial de cada jugador se baja apenas aparece en el ladder.
        vistos = set()
        for page in range(1, args.max_paginas + 1):
            pag = [d for d in (pedir(LADDER.format(page=page), s) or []) if isinstance(d, dict) and d.get("slug")]
            nuevos = [d for d in pag if d["slug"] not in vistos][:args.max_jugadores - len(vistos)]
            if not nuevos:
                break
            for d in nuevos:
                vistos.add(d["slug"])
                if args.todas_las_cuentas:
                    cs = cuentas_de(pedir(PERFIL.format(slug=d["slug"]), s))
                else:
                    cs = [d["account"]] if d.get("account") else []
                for c in cs:
                    filas += filas_de(pedir(RANKINGS.format(uuid=c["uuid"], meses=args.meses), s) or [], c, d["slug"])
            print(f"  página {page}: {len(vistos)} jugadores, {len(filas):,} fotos ({time.time() - t0:.0f} s)", flush=True)
            if len(vistos) >= args.max_jugadores:
                break

    if not filas:
        sys.exit("[X] Ninguna foto: correr antes con --probar y revisar los endpoints.")
    df = pd.DataFrame(filas).drop_duplicates(["account_uuid", "created_at"])
    df.to_csv(OUT / "historial.csv", index=False)
    print(f"[+] {OUT / 'historial.csv'}: {len(df):,} fotos de {df['account_uuid'].nunique() if len(df) else 0:,} cuentas "
          f"({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
