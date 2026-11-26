# 📋 TABLERO DE TAREAS Y DESPACHO (TASK_DISPATCH.md)

> **PROPÓSITO:** Archivo maestro compartido entre **Antigravity** y **Muse (OpenCode)** para coordinar la división del trabajo, consensuar decisiones técnicas y registrar autorizaciones.

---

## ⚙️ MODOS DE OPERACIÓN DEL REPOSITORIO (DIRECTIVA DEL USUARIO)

### 1. MODO ESTÁNDAR (HABITUAL - UN SOLO AGENTE ACTIVO)
* **El 90% del trabajo se realizará con un solo agente activo a la vez** (ya sea Antigravity o Muse).
* **Autonomía:** En modo estándar, el agente activo trabaja de forma autónoma, sin necesidad de esperar confirmación de un segundo agente ausente.
* **Protocolo:** Registra sus avances, decisiones y métricas en `AGENTS.md` y `PROJECT_SUMMARY.md` como dicta el protocolo del proyecto.
* **Independencia:** Todo el código, scripts y herramientas deben ser 100% autocontenidos y funcionales sin requerir la presencia simultánea de dos agentes.

### 2. MODO DÚO INTENSIVO (EXCEPCIONAL - AMBOS AGENTES SIMULTÁNEOS)
* **Sesiones intensivas puntuales y poco frecuentes**, reservadas exclusivamente para tareas pesadas de investigación, auditorías forenses, refactorizaciones de arquitectura o debugging complejo (como la sesión actual).
* **Regla Suprema de Consenso:** Solo durante estas sesiones en conjunto, **ningún agente puede modificar código o base de datos sin la confirmación explícita del otro agente en `COMMUNICATION.md` (`[APROBADO POR...]`) o del usuario**.

---

## 🎯 ESTADO DE TAREAS Y OBJETIVOS ACTIVOS (2026-09-04)

### ✅ Hitos Completados y Consolidados
1. **Infraestructura Multi-Proyecto en SQLite:** Tablas `api_projects`, `player_identities`, `player_puuid_map` y `v_puuid_canonical` operativas.
2. **Saneamiento y Unificación de Apex:** Challenger (200) y Grandmaster (500) unificados al 100% con doble key.
3. **Deduplicación Estricta:** 268/268 filas dobles unificadas ($71,602 \times 10 = 716,020$ exacto en `match_participants`).
4. **Hardening de Backup en la Nube:** Script de Drive con `PRAGMA quick_check;` preventivo, manifiestos JSON y snapshots inmutables.
5. **Suite de Pruebas Automatizadas:** 39 de 39 tests aprobados en verde (`tests/`).

---

### 🔄 Tareas Activas en Curso
1. **MUSE (Pase Activo en `COMMUNICATION.md`):**
   - Evaluar e implementar la propuesta del usuario de **Resolución Inversa por Match-ID** (`Match-V5`) para atribuir PUUIDs de partidas locales sin depender de solapamientos recientes.
   - Integración coordinada de Key 2 (Tercer Proyecto) si se decide combinar ambos métodos.
2. **ANTIGRAVITY (Completado y en Standby):**
   - Vigilancia watchdog concluida (27/27 ciclos OK).
   - Re-exportación completa de los 4 datasets CSV y sincronización a Drive y GitHub efectuada.
   - Pase emitido con evidencia empírica verificada en vivo.


---

## 💬 CANAL DE DISCUSIÓN ACTIVO
Toda la discusión técnica, propuestas y aprobaciones cruzadas deben registrarse en:
📄 [**`COMMUNICATION.md`**](file:///c:/Users/DarkA/OneDrive/Escritorio/Facultad/ML/ProyectoFinalML/COMMUNICATION.md)

---

## 🔁 PROTOCOLO DE ALTERNANCIA OBLIGATORIA (orden del USUARIO, 2026-09-03)
> [!IMPORTANT]
> **Ningún agente puede invocar directamente al otro: el pase se hace POR ESCRITO en `COMMUNICATION.md`.**
> 1. Cada vez que un agente termine su tanda de trabajo, DEBE cerrar su turno escribiendo en `COMMUNICATION.md` un mensaje de pase explícito dirigido al otro agente (`[PASE A MUSE]` o `[PASE A ANTIGRAVITY]`), con: qué hizo, qué deja pendiente y qué necesita del otro.
> 2. Cuando un agente inicie un turno, DEBE leer primero `AGENTS.md`, `PROJECT_SUMMARY.md`, `COMMUNICATION.md` y este archivo, atender el último `[PASE A ...]` dirigido a él, ejecutar lo pendiente y cerrar con un nuevo pase al otro agente.
> 3. Si no hay nada pendiente para el otro, el pase debe decirlo explícitamente (`sin pendientes`) en vez de omitirse.
> 4. **Criterio de terminación:** el ciclo alternado continúa hasta que (a) todo lo despachado en este archivo esté implementado y aprobado por ambos o el usuario, y (b) ningún agente encuentre errores pendientes en su área. Recién entonces se escribe `[FIN DE CICLO]` en `COMMUNICATION.md`.
> 5. El USUARIO actúa como relay entre turnos: basta con decir "seguí" / "pasalo al otro" para que el siguiente agente retome donde quedó el pase.
