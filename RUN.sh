#!/bin/bash
# Run script voor Aircraft Landing Scheduling
# Gebruik: ./RUN.sh

echo "========================================================================"
echo "Aircraft Landing Scheduling - Run Script"
echo "========================================================================"
echo ""

# Ga naar de juiste directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "✓ Working directory: $SCRIPT_DIR"
echo ""

# Activeer virtual environment
if [ -d "Operations_Guusje/bin" ]; then
    echo "✓ Activating virtual environment..."
    source Operations_Guusje/bin/activate
else
    echo "✗ Virtual environment not found!"
    echo "  Please create it first:"
    echo "  python3 -m venv Operations_Guusje"
    echo "  source Operations_Guusje/bin/activate"
    echo "  pip install -r aircraft_landing_scheduling/requirements.txt"
    exit 1
fi

echo ""
echo "✓ Starting program..."
echo ""

# Run het programma
python aircraft_landing_scheduling/code/main.py "$@"

echo ""
echo "✓ Done!"
echo ""
echo "Results saved in: $SCRIPT_DIR/results/"
