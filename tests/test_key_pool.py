"""
Pruebas del pool multi-clave: rotación en caliente por ranura, agotamiento no
fatal del pool y supervisión desde `run.py`.

Sin red: todas las claves son cadenas de prueba y ninguna petición sale.
"""

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import run
from src.collector import DataCollector
from src.riot_client import RiotAPIClient
from src.rate_limiter import RiotRateLimiter

K0, K1 = "RGAPI-vieja-0000", "RGAPI-vieja-1111"
NUEVA0 = "RGAPI-nueva-0000"


def cliente() -> RiotAPIClient:
    return RiotAPIClient(api_keys=[K0, K1], project_indices=[0, 1])


# --------------------------------------------------------------------------
# Rotación en caliente
# --------------------------------------------------------------------------
def test_refresh_pool_reemplaza_por_ranura_y_conserva_el_proyecto():
    """La ranura es el proyecto de Riot: renovar la clave no puede moverla."""
    c = cliente()
    cambios = c.refresh_pool([NUEVA0, K1])
    assert cambios["reemplazadas"] == [0]
    assert c.key_pool[0].api_key == NUEVA0
    assert c.key_pool[0].key_index == 0
    assert c.key_pool[0].session.headers["X-Riot-Token"] == NUEVA0
    assert c.key_pool[1].api_key == K1


def test_refresh_pool_reactiva_una_ranura_caida_con_la_misma_clave():
    c = cliente()
    c.key_pool[0].is_active = False
    c.key_pool[0].deactivated_reason = "HTTP 401"
    cambios = c.refresh_pool([K0, K1])
    assert cambios["reactivadas"] == [0]
    assert c.key_pool[0].is_active and c.key_pool[0].deactivated_reason is None


def test_refresh_pool_no_inventa_ranuras_nuevas_sin_permiso():
    """Una ranura nueva es un proyecto sin registrar: mezclaría PUUIDs."""
    c = cliente()
    assert c.refresh_pool([K0, K1, "RGAPI-tercera"])["agregadas"] == []
    assert len(c.key_pool) == 2
    assert c.refresh_pool([K0, K1, "RGAPI-tercera"], allow_new_slots=True)["agregadas"] == [2]
    assert len(c.key_pool) == 3 and c.key_pool[2].key_index == 2


def test_reemplazar_la_clave_reactiva_el_pool_agotado():
    c = cliente()
    for ctx in c.key_pool:
        ctx.is_active = False
    assert c._select_key_and_acquire() is None
    assert c.pool_exhausted.is_set()

    c.refresh_pool([NUEVA0, K1])
    assert not c.pool_exhausted.is_set()
    assert c._select_key_and_acquire() is not None


# --------------------------------------------------------------------------
# Agotamiento no fatal
# --------------------------------------------------------------------------
def test_pool_sin_claves_no_mata_el_proceso():
    """Antes esto era sys.exit(): desde un hilo trabajador solo mataba ese hilo."""
    c = cliente()
    for ctx in c.key_pool:
        ctx.is_active = False
    assert c._select_key_and_acquire() is None
    assert c._request("https://ejemplo.invalido/x") is None
    assert c.pool_exhausted.is_set()


def test_pool_status_reporta_carga_por_clave():
    c = cliente()
    c.key_pool[0].note("requests")
    c.key_pool[0].note("ok")
    c.key_pool[1].note("requests")
    c.key_pool[1].note("rate_limited")
    c.key_pool[1].is_active = False
    c.key_pool[1].deactivated_reason = "HTTP 401"

    filas = c.pool_status()
    assert [f["slot"] for f in filas] == [0, 1]
    assert filas[0]["requests"] == 1 and filas[0]["ok"] == 1 and filas[0]["activa"]
    assert filas[1]["rate_limited"] == 1 and filas[1]["motivo_baja"] == "HTTP 401"
    assert K0 not in filas[0]["key"]  # la clave va enmascarada


# --------------------------------------------------------------------------
# Supervisor (run.py)
# --------------------------------------------------------------------------
def supervisor(c, tmp_path, **kwargs):
    return run.KeyPoolSupervisor(c, threading.Event(),
                                 env_path=tmp_path / ".env", **kwargs)


def escribir_env(tmp_path, *keys):
    (tmp_path / ".env").write_text("RIOT_API_KEYS=" + ",".join(keys), encoding="utf-8")


def test_supervisor_corta_la_recoleccion_si_muere_todo_el_pool(tmp_path):
    c = cliente()
    s = supervisor(c, tmp_path, reload_env=False)
    s.revisar()
    assert not s.stop_event.is_set()

    for ctx in c.key_pool:
        ctx.is_active = False
        ctx.deactivated_reason = "HTTP 401"
    s.revisar()
    assert s.stop_event.is_set()
    assert len(s.bajas) == 2


def test_supervisor_revive_el_pool_leyendo_el_env(tmp_path):
    c = cliente()
    s = supervisor(c, tmp_path)
    for ctx in c.key_pool:
        ctx.is_active = False
    s.revisar()
    assert s.stop_event.is_set()

    # El usuario renueva las claves mientras la corrida sigue viva.
    escribir_env(tmp_path, NUEVA0, K1)
    s.revisar()
    assert not s.stop_event.is_set()
    assert c.key_pool[0].api_key == NUEVA0
    assert s.rotaciones == 2  # ranura 0 reemplazada, ranura 1 reactivada


def test_supervisor_ignora_placeholders_del_env(tmp_path):
    c = cliente()
    s = supervisor(c, tmp_path)
    escribir_env(tmp_path, "RGAPI-TU-CLAVE-AQUI", "RGAPI-XXXXXXXX")
    assert s.keys_del_env() == []
    s.revisar()
    assert c.key_pool[0].api_key == K0  # no se pisó nada


def test_supervisor_sin_indices_explicitos_no_mueve_el_proyecto(tmp_path):
    c = RiotAPIClient(api_keys=[K1], project_indices=[1])
    s = supervisor(c, tmp_path)
    escribir_env(tmp_path, NUEVA0, K1)
    s.revisar()
    assert c.key_pool[0].api_key == K1


def test_preflight_sin_p0_impide_sincronizar_el_padron(temp_db, monkeypatch):
    import src.config as config

    monkeypatch.setattr(config, "RIOT_API_KEYS", [K0, K1])
    monkeypatch.setattr(config, "RIOT_API_PROJECT_INDICES", [0, 1])
    monkeypatch.setattr(run, "check_keys", lambda keys: {K0: 401, K1: 200})
    fetch = MagicMock(return_value=[{"puuid": "p1", "tier": "MASTER"}])
    monkeypatch.setattr(RiotAPIClient, "fetch_all_diamond_plus_players", fetch)
    client = RiotAPIClient(api_keys=run.preflight_keys([K0, K1]))
    assert client.key_pool[0].project_index == 1
    assert DataCollector(client, temp_db).sync_players() == 0
    fetch.assert_not_called()
    assert temp_db.get_all_players() == []


def test_p0_ambiguo_por_clave_ajena_al_env_no_sincroniza(temp_db):
    client = RiotAPIClient(api_keys=["KEY-SIN-PROYECTO-ASIGNADO"])
    assert DataCollector(client, temp_db)._canonical_client() is None


def test_supervisor_sin_recarga_solo_informa(tmp_path):
    c = cliente()
    s = supervisor(c, tmp_path, reload_env=False)
    escribir_env(tmp_path, NUEVA0, K1)
    s.revisar()
    assert c.key_pool[0].api_key == K0 and s.rotaciones == 0


# --------------------------------------------------------------------------
# Señal de corte en el collector
# --------------------------------------------------------------------------
def test_el_crawler_respeta_la_senal_de_corte(temp_db):
    """Con el pool muerto, el recorrido no debe marcar jugadores como visitados."""
    temp_db.save_players([
        {"puuid": f"p{i}", "summoner_id": "s", "tier": "DIAMOND", "rank": "I",
         "league_points": 0, "wins": 0, "losses": 0} for i in range(5)
    ])
    mock_client = MagicMock()
    mock_client.num_active_keys = 1
    mock_client.get_telemetry.return_value = {"total_requests": 0, "status_200": 0,
                                              "status_429": 0, "timeouts": 0, "http_errors": 0}
    collector = DataCollector(riot_client=mock_client, db_manager=temp_db)

    corte = threading.Event()
    corte.set()
    collector.collect_matches(tier_filter="DIAMOND", stop_event=corte)

    mock_client.get_match_ids_by_puuid.assert_not_called()
    with temp_db._get_connection() as conn:
        crawleados = conn.execute(
            "SELECT COUNT(*) FROM players WHERE last_crawled_at IS NOT NULL;").fetchone()[0]
    assert crawleados == 0


def test_el_backfill_respeta_la_senal_de_corte(temp_db):
    with temp_db._get_connection() as conn:
        conn.execute("INSERT INTO matches (match_id, game_duration, queue_id) "
                     "VALUES ('LA2_1', 1800, 420);")
        conn.commit()
    mock_client = MagicMock()
    collector = DataCollector(riot_client=mock_client, db_manager=temp_db)

    corte = threading.Event()
    corte.set()
    assert collector.backfill_timelines(stop_event=corte) == 0
    mock_client.get_match_timeline.assert_not_called()


# --------------------------------------------------------------------------
# Penalizacion explicita de Riot (HTTP 429)
# --------------------------------------------------------------------------
def test_la_penalizacion_del_429_es_visible_para_los_demas_hilos(monkeypatch):
    """Sin esto, los historiales vaciados hacen creer que hay cuota libre."""
    from src.rate_limiter import RiotRateLimiter
    limiter = RiotRateLimiter()
    monkeypatch.setattr("src.rate_limiter.time.sleep", lambda _s: None)

    limiter.handle_rate_limit_exceeded(retry_after_sec=30)
    espera = limiter.get_wait_time()
    assert 30.0 < espera <= 31.0
    assert limiter.short_history == [] and limiter.long_history == []


def test_la_penalizacion_caduca_sola():
    from src.rate_limiter import RiotRateLimiter
    limiter = RiotRateLimiter()
    limiter.penalty_until = 0.0
    assert limiter.get_wait_time() == 0.0


def test_las_ventanas_ignoran_saltos_del_reloj_de_pared(monkeypatch):
    import src.rate_limiter as rate_limiter

    clock = SimpleNamespace(wall=1000.0, monotonic_value=10.0)
    monkeypatch.setattr(rate_limiter, "time", SimpleNamespace(
        time=lambda: clock.wall, monotonic=lambda: clock.monotonic_value, sleep=lambda _: None))
    limiter = RiotRateLimiter(short_limit=1)
    limiter.acquire_slot()
    clock.wall += 3600
    assert limiter.get_wait_time() == 1.0
    limiter.handle_rate_limit_exceeded(30)
    clock.wall -= 7200
    clock.monotonic_value += 1
    assert limiter.get_wait_time() == 30.0
    clock.monotonic_value += 31
    assert limiter.get_wait_time() == 0.0


@pytest.mark.parametrize("status", [200, 401, 403, 429])
def test_sondeo_usa_el_limitador_y_conserva_el_estado(monkeypatch, status):
    from scripts.ops.check_api_keys import probe_key

    calls = []
    get = MagicMock(return_value=SimpleNamespace(status_code=status, headers={"Retry-After": "17"}))
    monkeypatch.setattr("requests.get", get)
    monkeypatch.setattr("requests.Session.get", get)
    acquire = RiotRateLimiter.acquire_slot

    def acquire_traced(self):
        calls.append("slot")
        acquire(self)

    monkeypatch.setattr(RiotRateLimiter, "acquire_slot", acquire_traced)
    monkeypatch.setattr("src.rate_limiter.time.sleep", lambda seconds: calls.append(seconds))
    assert probe_key(K0, timeout=3.0) == status
    assert calls == (["slot", 18.0] if status == 429 else ["slot"])
    assert get.call_count == 1 and get.call_args.kwargs["timeout"] == 3.0


def test_el_puuid_de_otro_proyecto_queda_contabilizado(monkeypatch):
    """El 400 'decrypting' no es error ni exito: sin contarlo, la sesion es opaca."""
    class RespuestaFalsa:
        status_code = 400
        text = '{"status":{"message":"Bad Request - Exception decrypting ABC"}}'
        headers: dict = {}

    c = cliente()
    for ctx in c.key_pool:
        monkeypatch.setattr(ctx.session, "get", lambda *a, **kw: RespuestaFalsa())

    assert c._request("https://ejemplo.invalido/x") is None
    assert c.get_telemetry()["puuid_mismatch"] >= 2
    assert sum(f["puuid_mismatch"] for f in c.pool_status()) >= 2


def test_el_pool_agotado_dispara_una_revision_urgente(tmp_path):
    """Esperar el intervalo entero dejaria al crawler recorriendo el padron en falso."""
    c = cliente()
    s = supervisor(c, tmp_path, reload_env=False, interval=300.0)
    ahora, proxima = 1000.0, 9999.0
    assert s.toca_revisar(ahora, proxima) is False

    c.pool_exhausted.set()
    assert s.toca_revisar(ahora, proxima) is True

    # Ya cortada la recoleccion, la urgencia se apaga: vuelve al turno periodico.
    s.stop_event.set()
    assert s.toca_revisar(ahora, proxima) is False
    assert s.toca_revisar(ahora, ahora) is True


# --------------------------------------------------------------------------
# Respaldo a Google Drive (--respaldar)
# --------------------------------------------------------------------------
def test_sin_cambios_no_se_sube_nada_a_drive(monkeypatch):
    """Reescribir 687 MB en Drive por una sesion que no guardo nada no es gratis."""
    llamadas = []
    import scripts.backup.backup_to_drive as backup
    monkeypatch.setattr(backup, "run_backup", lambda *a, **kw: llamadas.append(1))

    assert run.respaldar_a_drive(0) is False
    assert run.respaldar_a_drive(-3) is False
    assert llamadas == []


def test_con_cambios_se_respalda(monkeypatch):
    import scripts.backup.backup_to_drive as backup
    monkeypatch.setattr(backup, "run_backup",
                        lambda *a, **kw: {"integrity_ok": True, "uploaded_latest": True,
                                          "csvs": ["full_dataset.csv"]})
    assert run.respaldar_a_drive(1200) is True


def test_la_integridad_rota_aborta_el_respaldo(monkeypatch):
    """Politica 4.W: si quick_check falla, no se pisa la copia sana de la nube."""
    import scripts.backup.backup_to_drive as backup
    monkeypatch.setattr(backup, "run_backup",
                        lambda *a, **kw: {"integrity_ok": False, "uploaded_latest": False,
                                          "csvs": []})
    assert run.respaldar_a_drive(1200) is False


def test_un_respaldo_que_explota_no_tumba_la_corrida(monkeypatch):
    import scripts.backup.backup_to_drive as backup

    def explota(*a, **kw):
        raise OSError("Drive no montado")

    monkeypatch.setattr(backup, "run_backup", explota)
    assert run.respaldar_a_drive(1200) is False


# --------------------------------------------------------------------------- #
# Límites leídos de Riot (keys permanentes, 2026-10-21)                       #
# --------------------------------------------------------------------------- #
def test_el_limitador_adopta_el_limite_real_que_informa_riot():
    limiter = RiotRateLimiter()
    limiter.update_from_headers({"X-App-Rate-Limit": "20:1,100:120",
                                 "X-App-Rate-Limit-Count": "1:1,1:120"})
    assert (limiter.short_limit, limiter.short_window) == (19, 1.0)
    assert (limiter.long_limit, limiter.long_window) == (99, 120.0)


def test_una_key_con_mas_cuota_se_aprovecha_sola():
    limiter = RiotRateLimiter()
    limiter.update_from_headers({"X-App-Rate-Limit": "500:10,30000:600",
                                 "X-App-Rate-Limit-Count": "1:10,1:600"})
    assert (limiter.short_limit, limiter.short_window) == (499, 10.0)
    assert (limiter.long_limit, limiter.long_window) == (29999, 600.0)


def test_el_conteo_de_riot_cubre_peticiones_de_otro_proceso():
    limiter = RiotRateLimiter()
    # Este proceso no pidió nada, pero Riot ya contó 99 en la ventana larga.
    limiter.update_from_headers({"X-App-Rate-Limit": "20:1,100:120",
                                 "X-App-Rate-Limit-Count": "0:1,99:120"})
    assert limiter.get_wait_time() > 100


def test_cabeceras_ausentes_o_rotas_no_cambian_nada():
    limiter = RiotRateLimiter()
    limiter.update_from_headers({})
    limiter.update_from_headers({"X-App-Rate-Limit": "basura"})
    assert (limiter.short_limit, limiter.long_limit) == (18, 95)


def test_al_arrancar_sale_una_sola_peticion_hasta_la_primera_respuesta():
    limiter = RiotRateLimiter(warmup=True)
    assert limiter.get_wait_time() == 0.0
    limiter.acquire_slot()
    assert limiter.get_wait_time() > 0          # el resto de los hilos espera
    limiter.update_from_headers({"X-App-Rate-Limit": "20:1,100:120",
                                 "X-App-Rate-Limit-Count": "1:1,1:120"})
    assert limiter.get_wait_time() == 0.0       # calibrado con el conteo real


def test_la_espera_de_arranque_no_bloquea_si_no_llega_respuesta():
    limiter = RiotRateLimiter(warmup=True)
    limiter.acquire_slot()
    limiter._first_sent_at -= RiotRateLimiter.WARMUP_TIMEOUT + 1
    assert limiter.get_wait_time() == 0.0


def test_las_keys_del_cliente_arrancan_de_a_una():
    client = RiotAPIClient(api_keys=["KEY_P0_AAAAAAAA"], project_indices=[0])
    client.key_pool[0].rate_limiter.acquire_slot()
    assert client.key_pool[0].rate_limiter.get_wait_time() > 0
