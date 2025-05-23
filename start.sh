#!/bin/bash

# Set project directory
PROJECT_DIR="/home/reeves/7th Sem Project/Stock Predictor"
ERROR_LOG="$PROJECT_DIR/startup.log"

# Navigate to project directory
cd "$PROJECT_DIR" || { echo "Directory not found!"; exit 1; }

# Clear previous error log
: > "$ERROR_LOG"

# Run Flask app in background and redirect errors to error.log
FLASK_APP=app.py flask run > /dev/null 2>>"$ERROR_LOG" &
FLASK_PID=$!

# Wait a few seconds to allow Flask to initialize
sleep 3

# Check if Flask is still running
if ps -p $FLASK_PID > /dev/null
then
    echo "Flask started successfully. Opening Brave browser..."
    brave http://localhost:5000 &
    wait $FLASK_PID
else
    echo "Flask failed to start. See startup.log for details."
    exit 1
fi
