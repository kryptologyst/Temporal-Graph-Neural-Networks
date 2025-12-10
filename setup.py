#!/usr/bin/env python3
"""Setup script for temporal graph neural networks project."""

import os
import subprocess
import sys
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"Running: {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def setup_project():
    """Setup the temporal graph neural networks project."""
    print("Setting up Temporal Graph Neural Networks Project")
    print("=" * 60)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        return False
    
    print(f"Python version: {sys.version}")
    
    # Install requirements
    requirements_file = Path(__file__).parent / "requirements.txt"
    if requirements_file.exists():
        success = run_command(
            f"{sys.executable} -m pip install -r {requirements_file}",
            "Installing Python dependencies"
        )
        if not success:
            return False
    else:
        print("Warning: requirements.txt not found")
    
    # Create necessary directories
    directories = [
        "data",
        "checkpoints", 
        "assets",
        "logs",
        "assets/evaluations"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    # Make scripts executable
    scripts_dir = Path(__file__).parent / "scripts"
    if scripts_dir.exists():
        for script in scripts_dir.glob("*.py"):
            os.chmod(script, 0o755)
            print(f"✓ Made executable: {script}")
    
    # Test imports
    print("\nTesting imports...")
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        
        import torch_geometric
        print(f"✓ PyTorch Geometric version: {torch_geometric.__version__}")
        
        try:
            import torch_geometric_temporal
            print(f"✓ PyTorch Geometric Temporal version: {torch_geometric_temporal.__version__}")
        except ImportError:
            print("⚠ PyTorch Geometric Temporal not found - some features may not work")
        
        import streamlit
        print(f"✓ Streamlit version: {streamlit.__version__}")
        
        import numpy
        print(f"✓ NumPy version: {numpy.__version__}")
        
        import pandas
        print(f"✓ Pandas version: {pandas.__version__}")
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    
    # Test basic functionality
    print("\nTesting basic functionality...")
    try:
        sys.path.append(str(Path(__file__).parent / "src"))
        from src.utils import set_seed, get_device
        from src.models import EvolveGCN
        
        set_seed(42)
        device = get_device()
        print(f"✓ Device detection: {device}")
        
        model = EvolveGCN(in_channels=10, out_channels=2, hidden_channels=32)
        print(f"✓ Model creation: {sum(p.numel() for p in model.parameters())} parameters")
        
    except Exception as e:
        print(f"✗ Functionality test failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run the demo: python scripts/run_demo.py")
    print("2. Train a model: python scripts/train.py")
    print("3. Explore the notebooks in the notebooks/ directory")
    print("4. Read the README.md for detailed documentation")
    
    return True


def main():
    """Main function."""
    success = setup_project()
    
    if not success:
        print("\nSetup failed. Please check the errors above.")
        sys.exit(1)
    
    print("\nSetup completed successfully!")


if __name__ == "__main__":
    main()
