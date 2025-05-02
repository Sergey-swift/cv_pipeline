#!/bin/bash

# Exit on error
set -e

echo "🔧 Creating virtual environment in .venv ..."
python3 -m venv .venv

echo "📦 Activating .venv ..."
source .venv/bin/activate

# ⬇️  Make project root importable
export PYTHONPATH="$PWD:$PYTHONPATH"

echo "⬇️ Installing dependencies from requirements.txt ..."
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Dependencies installed."

echo "🧪 Running tests ..."
pytest -q

echo "📌 Freezing exact versions to dev-lock.txt ..."
pip freeze > dev-lock.txt

echo "✅ Done. Environment is ready and test passed."
echo "💡 Activate later with: source .venv/bin/activate"
