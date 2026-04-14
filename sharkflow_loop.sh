#!/bin/bash
# ============================================
# SharkFlow Autonomous Loop Runner
# Carlos David Donoso Cordero (ddchack)
# ============================================
# Ejecuta Claude Code en loop autónomo por 4 horas
# Uso: chmod +x sharkflow_loop.sh && ./sharkflow_loop.sh

HOURS=4
END_TIME=$(($(date +%s) + HOURS * 3600))
CYCLE=1
LOG_FILE="loop_runner.log"

echo "🦈 SharkFlow Autonomous Loop — Iniciando $(date)" | tee -a "$LOG_FILE"
echo "⏰ Finalizará a las $(date -d "+${HOURS} hours" 2>/dev/null || date -v+${HOURS}H)" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"

while [ $(date +%s) -lt $END_TIME ]; do
    REMAINING=$(( (END_TIME - $(date +%s)) / 60 ))
    echo "" | tee -a "$LOG_FILE"
    echo "🔄 CICLO $CYCLE — $(date) — ${REMAINING}min restantes" | tee -a "$LOG_FILE"
    
    claude --dangerously-skip-permissions -p \
        "Ciclo $CYCLE de SharkFlow autónomo. Lee CLAUDE.md. NO pidas confirmación. Ejecuta mejoras y reporta en improvement_log.md. Minutos restantes: $REMAINING" \
        2>&1 | tee -a "$LOG_FILE"
    
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -ne 0 ]; then
        echo "⚠️ Ciclo $CYCLE terminó con error (código: $EXIT_CODE). Continuando..." | tee -a "$LOG_FILE"
        sleep 10
    fi
    
    CYCLE=$((CYCLE + 1))
    
    # Pausa entre ciclos para rate limits
    echo "⏸️ Pausa de 30s entre ciclos..." | tee -a "$LOG_FILE"
    sleep 30
done

echo "" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
echo "✅ Loop completado — $((CYCLE-1)) ciclos en ${HOURS}h — $(date)" | tee -a "$LOG_FILE"
echo "📋 Ver resultados: cat improvement_log.md" | tee -a "$LOG_FILE"
