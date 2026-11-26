"""
Controlador de Rate Limit para la API de Riot Games.
"""

import time
import threading
from typing import Dict, Optional


class RiotRateLimiter:
    """
    Controla la cadencia de peticiones a la API de Riot Games para evitar errores 429 (Rate Limit Exceeded).
    Gestiona límites en ventana corta (20 req / 1s) y ventana larga (100 req / 120s) con márgenes de seguridad.
    """

    # Tope de la espera de calibración: si la primera respuesta no llega (red
    # caída), no se bloquea para siempre.
    WARMUP_TIMEOUT = 5.0

    def __init__(self, short_limit: int = 18, short_window: float = 1.0,
                 long_limit: int = 95, long_window: float = 120.0,
                 warmup: bool = False):
        # `warmup`: hasta recibir la primera respuesta (que trae el conteo real
        # de Riot) sale una sola petición. Riot cuenta la cuota por key, no por
        # proceso: al relanzar, la ventana de 120 s puede seguir llena por la
        # corrida anterior y una ráfaga de hilos con historial vacío termina en
        # 429 en cadena (medido: 6 seguidos, Retry-After de 42 a 17 s).
        self._calibrated = not warmup
        self._first_sent_at: Optional[float] = None
        self.short_limit = short_limit
        self.short_window = short_window
        self.long_limit = long_limit
        self.long_window = long_window

        self.short_history = []
        self.long_history = []
        # Instante hasta el que Riot nos declaró penalizados por un 429 explícito.
        # Vive fuera de los historiales porque la penalización los invalida a
        # ambos: es un veto externo, no una cuota consumida.
        self.penalty_until = 0.0
        self.lock = threading.Lock()

    def get_wait_time(self) -> float:
        """
        Calcula el tiempo de espera necesario (en segundos) hasta que haya una ranura libre, sin bloquear.
        Retorna 0.0 si hay ranura disponible de inmediato.
        """
        with self.lock:
            now = time.monotonic()
            penalty_wait = max(0.0, self.penalty_until - now)
            # Limpiar timestamps expirados
            self.short_history = [t for t in self.short_history if now - t < self.short_window]
            self.long_history = [t for t in self.long_history if now - t < self.long_window]

            short_wait = 0.0
            if len(self.short_history) >= self.short_limit:
                oldest_short = self.short_history[0]
                short_wait = max(0.0, (oldest_short + self.short_window) - now)

            long_wait = 0.0
            if len(self.long_history) >= self.long_limit:
                oldest_long = self.long_history[0]
                long_wait = max(0.0, (oldest_long + self.long_window) - now)

            warmup_wait = 0.0
            if not self._calibrated and self._first_sent_at is not None:
                pending = self.WARMUP_TIMEOUT - (now - self._first_sent_at)
                warmup_wait = min(max(0.0, pending), 0.1)

            return max(short_wait, long_wait, penalty_wait, warmup_wait)

    def acquire_slot(self) -> None:
        """Registra el consumo inmediato de una ranura de petición."""
        with self.lock:
            now = time.monotonic()
            if not self._calibrated and self._first_sent_at is None:
                self._first_sent_at = now
            self.short_history.append(now)
            self.long_history.append(now)

    def wait_for_slot(self) -> None:
        """Bloquea la ejecución hasta que haya una ranura disponible respetando los límites."""
        while True:
            wait_time = self.get_wait_time()
            if wait_time <= 0.001:
                self.acquire_slot()
                return
            time.sleep(wait_time + 0.02)


    # Margen sobre el límite real: cubre las peticiones en vuelo que Riot todavía
    # no contó. Con el conteo de Riot reconciliado en cada respuesta, 1 alcanza.
    HEADER_MARGIN = 1

    @staticmethod
    def _parse_rate_header(value: Optional[str]) -> Dict[int, int]:
        """`"20:1,100:120"` -> `{1: 20, 120: 100}` (ventana en s -> cantidad)."""
        out: Dict[int, int] = {}
        for part in (value or "").split(","):
            amount, _, window = part.strip().partition(":")
            if amount.strip().isdigit() and window.strip().isdigit():
                out[int(window)] = int(amount)
        return out

    def update_from_headers(self, headers: Dict[str, str]) -> None:
        """
        Adopta el límite real de la key y el conteo que lleva Riot (`X-App-Rate-Limit`
        y `X-App-Rate-Limit-Count`).

        * **Límite:** los valores por defecto (18/1 s, 95/120 s) son de una key de
          desarrollo con margen fijo. Con el límite que informa Riot se usa la cuota
          real menos `HEADER_MARGIN` (99/120 s en vez de 95: +4 % de requests), y una
          key con más cuota (producción) se aprovecha sola, sin tocar código.
        * **Conteo:** si Riot contó más peticiones de las que este proceso registró
          (otro proceso usando la misma key: un notebook, `check_api_keys.py`), se
          agregan al historial para no chocar contra un 429.

        Antes, esta función solo dormía el hilo 2 s al pasar de 96, bloqueando un
        trabajador sin liberar cuota; ahora la espera la calcula `get_wait_time()`.
        """
        with self.lock:
            self._calibrated = True  # llegó una respuesta: termina el arranque de a una
        limits = self._parse_rate_header(headers.get("X-App-Rate-Limit"))
        if not limits:
            return
        counts = self._parse_rate_header(headers.get("X-App-Rate-Limit-Count"))
        short_w, long_w = min(limits), max(limits)
        with self.lock:
            now = time.monotonic()
            self.short_window = float(short_w)
            self.short_limit = max(1, limits[short_w] - self.HEADER_MARGIN)
            self.long_window = float(long_w)
            self.long_limit = max(1, limits[long_w] - self.HEADER_MARGIN)
            for window, attr in ((short_w, "short_history"), (long_w, "long_history")):
                hist = [t for t in getattr(self, attr) if now - t < window]
                riot = counts.get(window, 0)
                if riot > len(hist):
                    hist.extend([now] * (riot - len(hist)))
                setattr(self, attr, hist)

    def handle_rate_limit_exceeded(self, retry_after_sec: Optional[int] = None) -> None:
        """
        Maneja una respuesta HTTP 429 explícita pausando el tiempo requerido por Riot.

        La penalización se **publica** en `penalty_until` antes de dormir, no
        después: los historiales se vacían al salir, así que sin este dato el
        resto de los hilos vería cuota libre durante toda la pausa y volvería a
        golpear la API (más 429 en cadena). Publicarla además vuelve la espera
        visible para quien mire `get_wait_time()` desde afuera — es lo que le
        permite a `run.py` cosechar la pausa validando el dataset ya descargado.
        """
        wait_time = (retry_after_sec or 5) + 1.0
        with self.lock:
            self.penalty_until = max(self.penalty_until, time.monotonic() + wait_time)
        print(f"\n[!] Rate limit alcanzado (HTTP 429). Pausando {wait_time:.1f} segundos...")
        time.sleep(wait_time)
        with self.lock:
            self.short_history.clear()
            self.long_history.clear()

