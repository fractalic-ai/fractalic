#!/bin/bash

# Fractalic Chat Agent Test Script (enhanced)
# Запускает UI Server, устанавливает диагностические переменные окружения и открывает чат интерфейс
# Options:
#   --port <n>        Порт сервера (default 8000)
#   --host <h>        Хост (default 0.0.0.0)
#   --debug           Включить расширенный лог исполнения (FRACTALIC_DEBUG_EXEC=1)
#   --dev             Включить режим разработки (отключить кэширование статики)
#   --background      Запуск сервера в фоне (не открывать браузер, просто tail логи)
#   --no-browser      Не открывать браузер автоматически
#   --help            Показать помощь

set -euo pipefail

PORT=8000
HOST=0.0.0.0
DEBUG=0
DEV_MODE=0
BACKGROUND=0
OPEN_BROWSER=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --port)
            PORT="$2"; shift 2;;
        --host)
            HOST="$2"; shift 2;;
        --debug)
            DEBUG=1; shift;;
        --dev)
            DEV_MODE=1; shift;;
        --background)
            BACKGROUND=1; shift;;
        --no-browser)
            OPEN_BROWSER=0; shift;;
        --help|-h)
            grep '^#' "$0" | sed 's/^# //'; exit 0;;
        *) echo "Неизвестный аргумент: $1"; exit 1;;
    esac
done

echo "🚀 Запуск Fractalic Chat Agent... (host=$HOST port=$PORT debug=$DEBUG background=$BACKGROUND)"
echo "=================================="

# Проверяем, что мы в корневой директории fractalic
if [ ! -f "fractalic.py" ]; then
    echo "❌ Ошибка: Запустите этот скрипт из корневой директории fractalic"
    exit 1
fi

# Проверяем наличие файла чата
if [ ! -f "fractalic_chat.html" ] && [ ! -f "web/index.html" ]; then
    echo "❌ Ошибка: Не найден ни fractalic_chat.html ни web/index.html"
    exit 1
fi

echo "📋 Проверка зависимостей..."

# Активируем виртуальное окружение если есть
if [ -d ".venv" ]; then
    echo "🐍 Активируем виртуальное окружение..."
    source .venv/bin/activate
fi

# Проверяем Python пакеты (добавлен aiohttp для WS sink)
python3 - <<'PY'
import importlib, sys
missing = []
for pkg in ("fastapi", "websockets", "uvicorn", "aiohttp"):
    try:
        importlib.import_module(pkg)
    except Exception:
        missing.append(pkg)
if missing:
    print("MISSING:" + ",".join(missing))
PY
NEED=$(python3 - <<'PY'
import importlib
req=["fastapi","websockets","uvicorn","aiohttp"]
miss=[m for m in req if not importlib.util.find_spec(m)]
print(' '.join(miss))
PY
)
if [ -n "$NEED" ]; then
  echo "❌ Не установлены пакеты: $NEED — устанавливаем..."
  pip install $NEED
fi

echo "🔧 Настройка..."

# Убиваем процессы на целевом порту если есть
echo "🧹 Освобождаем порт $PORT..."
lsof -ti:"$PORT" | xargs kill -9 2>/dev/null || true

# Экспортируем переменные окружения для сервера
export SERVER_HOST="${HOST}"
export SERVER_PORT="${PORT}"
export PORT="${PORT}"  # некоторые платформы читают PORT
if [ "$DEBUG" = "1" ]; then
    export FRACTALIC_DEBUG_EXEC=1
    echo "🛠  Debug режим включен (FRACTALIC_DEBUG_EXEC=1)"
fi
if [ "$DEV_MODE" = "1" ]; then
    export FRACTALIC_DEV_MODE=1
    echo "⚡ Dev режим включен (кэширование статики отключено)"
fi

# Запускаем UI Server
echo "🚀 Запуск UI Server на порту $PORT..."
UVICORN_ARGS=(--host "$HOST" --port "$PORT")
if [ "$DEV_MODE" = "1" ]; then
    UVICORN_ARGS+=(--reload --reload-exclude "**/.fractalic/**")
fi
# Note: Auto-reload is disabled by default when --reload is not specified
# This prevents server restarts during LLM streaming

if [ "$BACKGROUND" = "1" ]; then
    nohup python3 -m uvicorn core.ui_server.server:app "${UVICORN_ARGS[@]}" > server.out 2>&1 &
    SERVER_PID=$!
    echo $SERVER_PID > .server_pid
    echo "💤 Сервер запущен в фоне (PID=$SERVER_PID). Логи: tail -f server.out"
else
    python3 -m uvicorn core.ui_server.server:app "${UVICORN_ARGS[@]}" &
    SERVER_PID=$!
fi

# Ждем запуска сервера
echo "⏳ Ждем запуска сервера..."
sleep 5

# Проверяем доступность сервера
if curl -s "http://localhost:${PORT}/health" > /dev/null; then
    echo "✅ Сервер запущен успешно!"
    echo ""
    echo "🌐 Чат интерфейс доступен по адресу:"
    echo "   http://localhost:${PORT}/chat"
    echo ""
    echo "📊 API информация:"
    echo "   http://localhost:${PORT}/info"
    echo ""
    echo "🔍 Состояние здоровья:"
    echo "   http://localhost:${PORT}/health"
    echo ""
    echo "💬 WebSocket endpoint:"
    echo "   ws://localhost:${PORT}/ws/chat"
    echo ""
    echo "=================================="
    echo "Для остановки нажмите Ctrl+C"
    echo "=================================="
    
    # Пытаемся открыть браузер (macOS)
    if [ "$OPEN_BROWSER" = "1" ] && [ "$BACKGROUND" = "0" ]; then
      if command -v open >/dev/null 2>&1; then
          echo "🌐 Открываем браузер..."
          open "http://localhost:${PORT}/chat"
      fi
    fi
    
    # Ждем прерывания
        if [ "$BACKGROUND" = "1" ]; then
            echo "📌 Для остановки: kill $(cat .server_pid) или ./test_chat.sh --stop"
            exit 0
        else
            trap "echo '🛑 Останавливаем сервер...'; lsof -ti:$PORT | xargs kill -9 2>/dev/null || true; exit 0" INT
            wait $SERVER_PID
        fi
    
else
    echo "❌ Сервер не запустился. Проверьте логи."
    kill $SERVER_PID 2>/dev/null
    exit 1
fi
