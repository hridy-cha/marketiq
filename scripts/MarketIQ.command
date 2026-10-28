#!/bin/zsh
# Double-click to start MarketIQ. Close this window (or press Ctrl-C) to stop it.
cd "$HOME/marketiq/repo" || { echo "MarketIQ is not installed at ~/marketiq/repo"; read -k1; exit 1; }
source .venv/bin/activate
pkill -f "marketiq serve" 2>/dev/null; pkill -f "next-server" 2>/dev/null; sleep 1
echo "Starting MarketIQ backend..."
marketiq serve > /tmp/marketiq-api.log 2>&1 &
API=$!
for i in {1..30}; do curl -s localhost:8000/api/health >/dev/null 2>&1 && break; sleep 1; done
if ! curl -s localhost:8000/api/health >/dev/null 2>&1; then
  echo ""; echo "Backend failed to start. Log:"; echo "------------------------------"; cat /tmp/marketiq-api.log; echo "------------------------------"
  echo "Press any key to close."; read -k1; exit 1
fi
echo "Starting dashboard..."
(cd web && NODE_ENV=production npm run start > /tmp/marketiq-web.log 2>&1) &
WEB=$!
trap "kill $API $WEB 2>/dev/null; pkill -f next-server 2>/dev/null; pkill -f 'marketiq serve' 2>/dev/null" EXIT INT TERM
for i in {1..30}; do curl -s localhost:3000 >/dev/null 2>&1 && break; sleep 1; done
open http://localhost:3000
echo ""
echo "  MarketIQ is running at http://localhost:3000"
echo "  Keep this window open. Close it to stop the app."
echo ""
wait
