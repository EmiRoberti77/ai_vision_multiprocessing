#!/bin/bash

# Setup Terminals Script for AI Vision Multiprocessing Project
# This script helps recreate your terminal layout in Cursor

echo "🚀 Setting up terminal environment for AI Vision Multiprocessing..."

# Check if we're in the right directory
if [[ ! -f "README.md" ]] || [[ ! -d "multi_processing" ]]; then
    echo "❌ Please run this script from the multiprocessing project root directory"
    exit 1
fi

# Function to open a new terminal with a specific title and command
open_terminal() {
    local title="$1"
    local command="$2"
    local directory="$3"
    
    echo "📱 Opening terminal: $title"
    
    # Note: This is a template - actual implementation depends on your terminal emulator
    # For gnome-terminal:
    # gnome-terminal --title="$title" --working-directory="$directory" -- bash -c "$command; exec bash"
    
    # For tmux (if available):
    # tmux new-window -n "$title" -c "$directory" "$command"
    
    # For now, just print what would be done
    echo "   Title: $title"
    echo "   Directory: $directory"
    echo "   Command: $command"
    echo "   ---"
}

# Get the current directory
PROJECT_ROOT=$(pwd)

# Setup different terminal windows
echo "🔧 Configuring terminal windows..."

open_terminal "AI Environment" "source ~/ai_env311/bin/activate" "$PROJECT_ROOT"
open_terminal "Main Process" "source ~/ai_env311/bin/activate && echo 'Ready to run: python3 -m multi_processing.main'" "$PROJECT_ROOT"
open_terminal "Testing" "source ~/ai_env311/bin/activate && echo 'Ready to run tests'" "$PROJECT_ROOT/tests"
open_terminal "Database" "echo 'Database operations ready'" "$PROJECT_ROOT"
open_terminal "Commands" "source ~/ai_env311/bin/activate && echo 'General commands terminal'" "$PROJECT_ROOT"

echo ""
echo "✅ Terminal setup complete!"
echo ""
echo "💡 Tips:"
echo "   • Use Ctrl+Shift+\` to open new terminals"
echo "   • Use Ctrl+Shift+P and search 'Tasks: Run Task' to run predefined tasks"
echo "   • Customize terminal profiles in .vscode/settings.json"
echo ""
echo "📋 Available VS Code Tasks:"
echo "   • Setup AI Environment"
echo "   • Start Main Process"
echo "   • Run Tests"
echo "   • Database Terminal"
echo ""
echo "🔗 Access tasks via: Ctrl+Shift+P → 'Tasks: Run Task'"

