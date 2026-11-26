"""Atributos de campeones de fuentes externas a Riot API (§9.S.37), en data/externos/.

  * Meraki Analytics (cdn.merakianalytics.com): un JSON con roles finos, tipo de ataque,
    rango y attributeRatings de los ~170 campeones;
  * CommunityDragon (raw.communitydragon.org): champion-summary.json y, por campeón,
    tacticalInfo (tipo de daño) y playstyleInfo (daño, durabilidad, control, movilidad,
    utilidad) del cliente.
Ambas son públicas, sin key. No pasa por RiotRateLimiter porque no es Riot API; igual va
de a 8 pedidos en paralelo. Se bajan las versiones "latest": son atributos que casi no
cambian entre parches.

Uso: python scripts/collection/bajar_campeones_externos.py   (~1 min)"""
import json, socket, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EXT = REPO / "data" / "externos"
CD = "https://raw.communitydragon.org/latest/plugins/rcp-be-lol-game-data/global/default/v1"
# Solo IPv4: en la notebook la ruta IPv6 a Cloudflare queda colgada y cada pedido esperaba
# ~20 s antes de caer a IPv4 (2026-11-14).
_gai = socket.getaddrinfo
socket.getaddrinfo = lambda host, *a, **k: [r for r in _gai(host, *a, **k) if r[0] == socket.AF_INET] or _gai(host, *a, **k)
MERAKI = "https://cdn.merakianalytics.com/riot/lol/resources/latest/en-US/champions.json"


def get(url, intentos=3):
    for i in range(intentos):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ml-academic-script"})
            return json.load(urllib.request.urlopen(req, timeout=90))
        except Exception as e:
            print(f"  reintento {i + 1} {url}: {e}")
            time.sleep(2)
    return None


def main():
    (EXT / "meraki").mkdir(parents=True, exist_ok=True)
    (EXT / "cdragon").mkdir(parents=True, exist_ok=True)
    json.dump(get(MERAKI), open(EXT / "meraki" / "champions.json", "w", encoding="utf-8"), ensure_ascii=False)
    resumen = get(f"{CD}/champion-summary.json")
    json.dump(resumen, open(EXT / "cdragon" / "champion-summary.json", "w", encoding="utf-8"), ensure_ascii=False)
    ids = [c["id"] for c in resumen if c["id"] > 0]
    with ThreadPoolExecutor(max_workers=8) as ex:
        res = list(ex.map(lambda i: (i, get(f"{CD}/champions/{i}.json")), ids))
    out = {i: {k: d.get(k) for k in ("name", "alias", "roles", "tacticalInfo", "playstyleInfo")} for i, d in res if d}
    print(f"sin respuesta: {[i for i, d in res if not d]}")
    json.dump(out, open(EXT / "cdragon" / "champions_tactical.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"Meraki y CommunityDragon: {len(out)} campeones en {EXT}")


if __name__ == "__main__":
    main()
