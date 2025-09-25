#!/bin/bash

# Fractalic Chat Agent Test Script
# Запускает UI Server и открывает чат интерфейс

echo "🚀 Запуск Fractalic Chat Agent..."
echo "=================================="

# Проверяем, что мы в корневой директории fractalic
if [ ! -f "fractalic.py" ]; then
    echo "❌ Ошибка: Запустите этот скрипт из корневой директории fractalic"
    exit 1
fi

# Проверяем наличие файла чата
if [ ! -f "fractalic_chat.html" ]; then
    echo "❌ Ошибка: Файл fractalic_chat.html не найден"
    exit 1
fi

echo "📋 Проверка зависимостей..."

# Активируем виртуальное окружение если есть
if [ -d ".venv" ]; then
    echo "🐍 Активируем виртуальное окружение..."
    source .venv/bin/activate
fi

# Проверяем Python пакеты
python3 -c "import fastapi, websockets" 2>/dev/null || {
    echo "❌ Не установлены необходимые пакеты. Устанавливаем..."
    pip install fastapi websockets uvicorn
}

echo "🔧 Настройка..."

# Убиваем процессы на порту 8000 если есть
echo "🧹 Освобождаем порт 8000..."
lsof -ti:8000 | xargs kill -9 2>/dev/null || true

# Запускаем UI Server
echo "🚀 Запуск UI Server на порту 8000..."
python3 -m uvicorn core.ui_server.server:app --host 0.0.0.0 --port 8000 &
SERVER_PID=$!

# Ждем запуска сервера
echo "⏳ Ждем запуска сервера..."
sleep 5

# Проверяем доступность сервера
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ Сервер запущен успешно!"
    echo ""
    echo "🌐 Чат интерфейс доступен по адресу:"
    echo "   http://localhost:8000/chat"
    echo ""
    echo "📊 API информация:"
    echo "   http://localhost:8000/info"
    echo ""
    echo "🔍 Состояние здоровья:"
    echo "   http://localhost:8000/health"
    echo ""
    echo "💬 WebSocket endpoint:"
    echo "   ws://localhost:8000/ws/chat"
    echo ""
    echo "=================================="
    echo "Для остановки нажмите Ctrl+C"
    echo "=================================="
    
    # Пытаемся открыть браузер (macOS)
    if command -v open >/dev/null 2>&1; then
        echo "🌐 Открываем браузер..."
        open http://localhost:8000/chat
    fi
    
    # Ждем прерывания
    trap "echo '🛑 Останавливаем сервер...'; kill $SERVER_PID; exit 0" INT
    wait $SERVER_PID
    
else
    echo "❌ Сервер не запустился. Проверьте логи."
    kill $SERVER_PID 2>/dev/null
    exit 1
fi
