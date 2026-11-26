"""
Herramienta de diagnostico manual (interactiva) para la API de Riot Games.
=============================================
Este script realiza las siguientes pruebas:
1. Valida el formato y la presencia de la API Key.
2. Comprueba el estado del servidor y la validez de la API Key (Status V4).
3. Busca un jugador por su Riot ID (GameName#TagLine) usando Account-V1.
4. Obtiene información del invocador (Nivel, Icono, etc.) usando Summoner-V4.
5. Obtiene las últimas partidas del jugador usando Match-V5.

Uso:
  python tools/check_riot_api.py
"""

import os
import sys
from pathlib import Path

# Intentar cargar variables de entorno desde .env
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).resolve().parents[1] / ".env"
    load_dotenv(dotenv_path=env_path)
except ImportError:
    pass

try:
    import requests
except ImportError:
    print("\n[!] Error: La librería 'requests' no está instalada.")
    print("    Instalala ejecutando: pip install -r requirements.txt\n")
    sys.exit(1)


# Mapeo de plataformas a enrutamiento regional
PLATFORM_TO_REGION = {
    "la1": "americas",   # LAN
    "la2": "americas",   # LAS
    "na1": "americas",   # NA
    "br1": "americas",   # BR
    "euw1": "europe",    # EUW
    "eun1": "europe",    # EUNE
    "tr1": "europe",     # TR
    "ru": "europe",      # RU
    "kr": "asia",        # KR
    "jp1": "asia",       # JP
    "oc1": "sea",        # OCE
    "ph2": "sea",        # PH
    "sg2": "sea",        # SG
    "th2": "sea",        # TH
    "tw2": "sea",        # TW
    "vn2": "sea"         # VN
}


def get_api_key() -> str:
    """Obtiene la API Key desde variables de entorno o la solicita al usuario."""
    api_key = os.getenv("RIOT_API_KEY", "").strip()

    # Si está vacía o es un placeholder
    if not api_key or "TU-CLAVE" in api_key or "XXXXXXXX" in api_key:
        print("\n" + "="*60)
        print("  CONFIGURACIÓN DE LA RIOT API KEY")
        print("="*60)
        print("No se encontró una API Key válida en el archivo .env.")
        print("Puedes obtener una Development API Key en: https://developer.riotgames.com/")
        print("Nota: Las Development API Keys duran 24 horas y comienzan con 'RGAPI-'.\n")
        api_key = input("Introduce tu Riot API Key para probar ahora (o presiona Enter para cancelar): ").strip()
        if not api_key:
            print("\n[!] Operación cancelada. Configura RIOT_API_KEY en el archivo .env.")
            sys.exit(0)

    return api_key


def handle_api_error(response: requests.Response, endpoint_desc: str) -> None:
    """Imprime un diagnóstico detallado según el código de estado HTTP."""
    status_code = response.status_code
    print(f"\n[X] Error al consultar {endpoint_desc} (HTTP {status_code})")
    
    if status_code == 401:
        print("    -> No autorizado: La API Key no fue proporcionada o es inválida.")
    elif status_code == 403:
        print("    -> Prohibido (Forbidden):")
        print("       - La API Key probablemente ha expirado (las Development Keys expiran cada 24 horas).")
        print("       - O no tienes permisos para este endpoint.")
        print("       -> Renueva tu clave en https://developer.riotgames.com/")
    elif status_code == 404:
        print("    -> Recurso no encontrado: Verifica los datos ingresados (ej. nombre de invocador o ID).")
    elif status_code == 429:
        print("    -> Límite de peticiones alcanzado (Rate Limit Exceeded). Espera unos segundos e intenta nuevamente.")
    elif status_code in (500, 502, 503, 504):
        print("    -> Error en los servidores de Riot Games. Intenta de nuevo más tarde.")
    else:
        print(f"    -> Detalle: {response.text}")


def check_platform_status(api_key: str, platform: str) -> bool:
    """Prueba 1: Valida la API Key y la conexión al servidor de Riot (Status V4)."""
    print(f"\n--- [1/3] Probando conexión y validación de API Key en '{platform}' ---")
    url = f"https://{platform}.api.riotgames.com/lol/status/v4/platform-data"
    headers = {"X-Riot-Token": api_key}

    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"[✓] Conexión exitosa!")
            print(f"    - Servidor: {data.get('name', platform)} ({data.get('id', platform)})")
            
            maintenances = len(data.get('maintenances', []))
            incidents = len(data.get('incidents', []))
            print(f"    - Estado de la plataforma: {maintenances} mantenimientos activos, {incidents} incidentes.")
            return True
        else:
            handle_api_error(response, "Platform Status V4")
            return False
    except requests.exceptions.RequestException as e:
        print(f"[X] Error de red / conexión: {e}")
        return False


def check_account_and_summoner(api_key: str, game_name: str, tag_line: str, platform: str, regional_routing: str):
    """Prueba 2 y 3: Busca cuenta por Riot ID (Account-V1) y Summoner por PUUID (Summoner-V4)."""
    print(f"\n--- [2/3] Buscando cuenta Riot ID: '{game_name}#{tag_line}' en región '{regional_routing}' ---")
    
    # 1. Account-V1
    account_url = f"https://{regional_routing}.api.riotgames.com/riot/account/v1/accounts/by-riot-id/{game_name}/{tag_line}"
    headers = {"X-Riot-Token": api_key}

    try:
        res_acc = requests.get(account_url, headers=headers, timeout=10)
        if res_acc.status_code != 200:
            handle_api_error(res_acc, f"Account-V1 ({game_name}#{tag_line})")
            return None
        
        acc_data = res_acc.json()
        puuid = acc_data.get("puuid")
        print(f"[✓] Cuenta encontrada:")
        print(f"    - Riot ID: {acc_data.get('gameName')}#{acc_data.get('tagLine')}")
        print(f"    - PUUID: {puuid[:12]}... (oculto por brevedad)")

        # 2. Summoner-V4
        print(f"\n--- [3/3] Obteniendo datos de Invocador y Partidas en '{platform}' ---")
        summoner_url = f"https://{platform}.api.riotgames.com/lol/summoner/v4/summoners/by-puuid/{puuid}"
        res_summ = requests.get(summoner_url, headers=headers, timeout=10)
        
        if res_summ.status_code == 200:
            summ_data = res_summ.json()
            print(f"[✓] Datos de invocador obtenidos:")
            print(f"    - Nivel de invocador: {summ_data.get('summonerLevel')}")
            print(f"    - Icono de perfil ID: {summ_data.get('profileIconId')}")
        else:
            print(f"[!] Nota: No se encontraron datos de invocador en '{platform}'. (¿El jugador juega en otro servidor?)")

        # 3. Match-V5
        match_url = f"https://{regional_routing}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids?count=5"
        res_match = requests.get(match_url, headers=headers, timeout=10)
        
        if res_match.status_code == 200:
            match_ids = res_match.json()
            print(f"[✓] Historial de partidas obtenido con éxito:")
            print(f"    - Últimas {len(match_ids)} partidas encontradas:")
            for idx, match_id in enumerate(match_ids, 1):
                print(f"      {idx}. {match_id}")
        else:
            print(f"[!] No se pudo obtener el historial de partidas (HTTP {res_match.status_code}).")

        return puuid

    except requests.exceptions.RequestException as e:
        print(f"[X] Error de red: {e}")
        return None


def main():
    print("="*60)
    print("      RIOT GAMES API - TEST SCRIPT")
    print("="*60)
    
    api_key = get_api_key()
    platform = os.getenv("DEFAULT_PLATFORM", "la2").lower()
    regional_routing = os.getenv("DEFAULT_REGIONAL_ROUTING", PLATFORM_TO_REGION.get(platform, "americas")).lower()

    print(f"\nConfiguración actual:")
    print(f"  - Plataforma / Servidor: {platform}")
    print(f"  - Enrutamiento Regional: {regional_routing}")
    print(f"  - API Key: {api_key[:10]}...{api_key[-4:] if len(api_key) > 14 else ''}")

    # Paso 1: Test de status y API Key
    status_ok = check_platform_status(api_key, platform)
    if not status_ok:
        print("\n[!] La prueba de conexión falló. Revisa tu API Key y configuración antes de continuar.")
        return

    # Paso 2: Test opcional de cuenta
    print("\n" + "-"*60)
    print("¿Deseas probar la búsqueda de un jugador específico?")
    print("Ingresa el Riot ID en formato 'Nombre#TAG' (ejemplo: 'MiNombre#LAS' o 'Faker#KR1').")
    print("O simplemente presiona Enter para omitir la búsqueda de jugador.")
    user_input = input("\nRiot ID: ").strip()

    if user_input and "#" in user_input:
        parts = user_input.split("#", 1)
        game_name = parts[0].strip()
        tag_line = parts[1].strip()
        check_account_and_summoner(api_key, game_name, tag_line, platform, regional_routing)
    elif user_input:
        print("[!] Formato inválido. Debe contener '#' (ejemplo: Nombre#TAG).")
    else:
        print("\n[i] Búsqueda de jugador omitida.")

    print("\n" + "="*60)
    print("      PRUEBA FINALIZADA CON ÉXITO")
    print("="*60)


if __name__ == "__main__":
    main()

