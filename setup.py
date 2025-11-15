#!/usr/bin/env python3
"""
Setup script for Sentiment Analysis Pipeline
Handles Git and DVC initialization properly
"""

import os
import subprocess
import sys

def run_command(cmd, ignore_errors=False):
    """Run shell command and return success status"""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        if result.stdout:
            print(result.stdout.strip())
        return True
    except subprocess.CalledProcessError as e:
        if not ignore_errors:
            print(f"⚠️ Warning: {e}")
            if e.stderr:
                print(e.stderr.strip())
        return False

def setup_git():
    """Initialize Git repository"""
    print("\n" + "="*60)
    print("📚 Setting up Git")
    print("="*60)
    
    if os.path.exists('.git'):
        print("✅ Git already initialized")
        return True
    
    print("🔧 Initializing Git repository...")
    if run_command("git init"):
        run_command('git config user.email "sentiment@analysis.com"', ignore_errors=True)
        run_command('git config user.name "Sentiment Analysis"', ignore_errors=True)
        print("✅ Git initialized successfully")
        return True
    else:
        print("❌ Git initialization failed")
        return False

def setup_dvc():
    """Initialize DVC"""
    print("\n" + "="*60)
    print("📦 Setting up DVC")
    print("="*60)
    
    if os.path.exists('.dvc'):
        print("✅ DVC already initialized")
        return True
    
    print("🔧 Initializing DVC...")
    
    # Try with SCM first
    if run_command("dvc init", ignore_errors=True):
        print("✅ DVC initialized with Git integration")
        return True
    
    # Try without SCM
    print("🔧 Trying DVC without SCM...")
    if run_command("dvc init --no-scm"):
        print("✅ DVC initialized without SCM")
        return True
    
    print("❌ DVC initialization failed")
    return False

def setup_dvc_remote():
    """Setup DVC remote storage"""
    print("\n🗄️ Setting up DVC remote storage...")
    
    # Create storage directory
    storage_path = "./dvc_storage"
    os.makedirs(storage_path, exist_ok=True)
    
    # Add remote
    if run_command(f"dvc remote add -d myremote {storage_path}", ignore_errors=True):
        print(f"✅ DVC remote configured: {storage_path}")
        return True
    
    # Try to modify if already exists
    if run_command(f"dvc remote modify myremote url {storage_path}", ignore_errors=True):
        print(f"✅ DVC remote updated: {storage_path}")
        return True
    
    print("⚠️ Could not configure DVC remote")
    return False

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating project directories...")
    
    dirs = [
        'data/raw',
        'data/processed',
        'saved_models',
        'artifacts',
        'plots',
        'mlruns',
        'dvc_storage'
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    print(f"✅ Created {len(dirs)} directories")

def create_gitignore():
    """Create .gitignore file"""
    print("\n📝 Creating .gitignore...")
    
    gitignore_content = """# Python
__pycache__/
*.py[cod]
.Python
venv/
env/

# Data (tracked by DVC)
/data/raw/*.csv
/data/processed/*.pkl

# Models (tracked by DVC)
/saved_models/*.h5

# MLflow
/mlruns/
/mlflow_artifacts/

# DVC
/dvc_storage/

# Artifacts
/artifacts/*.pkl

# Plots
/plots/*.png

# Environment
.env
"""
    
    with open('.gitignore', 'w') as f:
        f.write(gitignore_content)
    
    print("✅ .gitignore created")

def main():
    """Main setup function"""
    print("="*60)
    print("🚀 Sentiment Analysis Pipeline Setup")
    print("="*60)
    
    # Create directories
    create_directories()
    
    # Create .gitignore
    create_gitignore()
    
    # Setup Git
    git_success = setup_git()
    
    # Setup DVC
    dvc_success = setup_dvc()
    
    if dvc_success:
        setup_dvc_remote()
    
    # Summary
    print("\n" + "="*60)
    print("📋 Setup Summary")
    print("="*60)
    print(f"Git: {'✅ Initialized' if git_success else '❌ Failed'}")
    print(f"DVC: {'✅ Initialized' if dvc_success else '❌ Failed'}")
    print("Directories: ✅ Created")
    print(".gitignore: ✅ Created")
    
    print("\n" + "="*60)
    print("✅ Setup Complete!")
    print("="*60)
    
    print("\n📝 Next Steps:")
    print("1. Run pipeline: python main.py")
    print("2. Or skip DVC: python main.py --no-dvc")
    print("3. View MLflow UI: mlflow ui")
    
    if not dvc_success:
        print("\n⚠️ DVC setup failed. Options:")
        print("   a) Install Git and run setup.py again")
        print("   b) Run pipeline without DVC: python main.py --no-dvc")

if __name__ == "__main__":
    main()