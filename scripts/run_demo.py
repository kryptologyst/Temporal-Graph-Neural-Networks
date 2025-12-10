#!/usr/bin/env python3
"""Demo runner script for temporal graph neural networks."""

import subprocess
import sys
from pathlib import Path


def run_demo():
    """Run the Streamlit demo."""
    demo_path = Path(__file__).parent.parent / "demo" / "app.py"
    
    if not demo_path.exists():
        print(f"Demo file not found: {demo_path}")
        return False
    
    try:
        print("Starting Temporal Graph Neural Networks Demo...")
        print("The demo will open in your web browser.")
        print("Press Ctrl+C to stop the demo.")
        
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(demo_path), "--server.port", "8501"
        ])
        return True
    except KeyboardInterrupt:
        print("\nDemo stopped by user.")
        return True
    except Exception as e:
        print(f"Error running demo: {e}")
        return False


def main():
    """Main function."""
    print("Temporal Graph Neural Networks Demo Runner")
    print("=" * 50)
    
    # Check if streamlit is installed
    try:
        import streamlit
        print(f"Streamlit version: {streamlit.__version__}")
    except ImportError:
        print("Streamlit not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "streamlit"])
    
    # Run demo
    success = run_demo()
    
    if success:
        print("Demo completed successfully!")
    else:
        print("Demo failed to run.")
        sys.exit(1)


if __name__ == "__main__":
    main()
