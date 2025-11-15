import os
import subprocess
import yaml

class DVCManager:
    """Manage DVC for data and model versioning"""
    
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.dvc_config = self.config['dvc']
    
    def run_command(self, command):
        """Run shell command"""
        try:
            result = subprocess.run(
                command,
                shell=True,
                check=True,
                capture_output=True,
                text=True
            )
            print(f"✅ {command}")
            if result.stdout:
                print(result.stdout)
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Error: {e}")
            if e.stderr:
                print(e.stderr)
            return False
    
    def initialize_git(self):
        """Initialize Git repository if not exists"""
        print("\n🔧 Initializing Git repository...")
        
        if os.path.exists('.git'):
            print("✅ Git already initialized")
            return True
        
        if not self.run_command("git init"):
            return False
        
        # Configure git if needed
        self.run_command('git config user.email "sentiment@analysis.com"')
        self.run_command('git config user.name "Sentiment Analysis"')
        
        return True
    
    def initialize_dvc(self):
        """Initialize DVC repository"""
        print("\n🔧 Initializing DVC...")
        
        if os.path.exists('.dvc'):
            print("✅ DVC already initialized")
            return True
        
        # Initialize with --no-scm if git init fails
        if not self.run_command("dvc init"):
            print("⚠️ Trying DVC init with --no-scm...")
            return self.run_command("dvc init --no-scm")
        
        return True
    
    def setup_remote_storage(self):
        """Setup DVC remote storage"""
        print("\n📦 Setting up DVC remote storage...")
        
        # Create storage directory
        os.makedirs(self.dvc_config['remote_url'], exist_ok=True)
        
        # Add remote
        remote_name = self.dvc_config['remote_name']
        remote_url = self.dvc_config['remote_url']
        
        self.run_command(f"dvc remote add -d {remote_name} {remote_url}")
        
        print(f"✅ Remote storage configured: {remote_url}")
    
    def track_data(self, data_path='data/raw/sentiment_data.csv'):
        """Track data file with DVC"""
        print(f"\n📊 Tracking data: {data_path}")
        
        if not os.path.exists(data_path):
            print(f"❌ File not found: {data_path}")
            return False
        
        # Add to DVC
        self.run_command(f"dvc add {data_path}")
        
        # Git add the .dvc file
        dvc_file = f"{data_path}.dvc"
        if os.path.exists(dvc_file):
            self.run_command(f"git add {dvc_file} .gitignore")
        
        print(f"✅ Data tracked with DVC")
        return True
    
    def track_models(self):
        """Track saved models with DVC"""
        print("\n🤖 Tracking models...")
        
        model_dir = 'saved_models'
        if not os.path.exists(model_dir):
            print(f"❌ Directory not found: {model_dir}")
            return False
        
        # Add to DVC
        self.run_command(f"dvc add {model_dir}")
        
        # Git add the .dvc file
        dvc_file = f"{model_dir}.dvc"
        if os.path.exists(dvc_file):
            self.run_command(f"git add {dvc_file} .gitignore")
        
        print(f"✅ Models tracked with DVC")
        return True
    
    def track_processed_data(self):
        """Track processed data with DVC"""
        print("\n⚙️ Tracking processed data...")
        
        processed_path = self.config['data']['processed_path']
        if not os.path.exists(processed_path):
            print(f"❌ File not found: {processed_path}")
            return False
        
        self.run_command(f"dvc add {processed_path}")
        
        dvc_file = f"{processed_path}.dvc"
        if os.path.exists(dvc_file):
            self.run_command(f"git add {dvc_file} .gitignore")
        
        print(f"✅ Processed data tracked with DVC")
        return True
    
    def push_to_remote(self):
        """Push tracked files to remote storage"""
        print("\n📤 Pushing to remote storage...")
        return self.run_command("dvc push")
    
    def pull_from_remote(self):
        """Pull tracked files from remote storage"""
        print("\n📥 Pulling from remote storage...")
        return self.run_command("dvc pull")
    
    def show_status(self):
        """Show DVC status"""
        print("\n📋 DVC Status:")
        self.run_command("dvc status")
    
    def create_pipeline(self):
        """Create DVC pipeline for reproducibility"""
        print("\n🔄 Creating DVC pipeline...")
        
        # Stage 1: Data preprocessing
        self.run_command("""
            dvc stage add -n preprocess \
                -d data/raw/sentiment_data.csv \
                -d preprocessing.py \
                -o data/processed/processed_data.pkl \
                -o artifacts/tokenizer.pkl \
                -o artifacts/label_encoder.pkl \
                python preprocessing.py
        """)
        
        # Stage 2: Model training
        models = ['lstm', 'gru', 'textcnn']
        for model in models:
            self.run_command(f"""
                dvc stage add -n train_{model} \
                    -d data/processed/processed_data.pkl \
                    -d train.py \
                    -d models.py \
                    -o saved_models/{model}_model.h5 \
                    python train.py --model {model}
            """)
        
        print("✅ DVC pipeline created!")
        print("Run 'dvc dag' to visualize the pipeline")

def setup_complete_dvc():
    """Complete DVC setup"""
    print("="*60)
    print("🚀 Setting up DVC for Sentiment Analysis Project")
    print("="*60)
    
    manager = DVCManager()
    
    # Initialize
    manager.initialize_dvc()
    
    # Setup remote
    manager.setup_remote_storage()
    
    # Track data
    manager.track_data()
    
    # Show status
    manager.show_status()
    
    print("\n✅ DVC setup complete!")
    print("\n📝 Next steps:")
    print("1. Run preprocessing: python preprocessing.py")
    print("2. Run training: python train.py")
    print("3. Track models: python -c 'from dvc_setup import DVCManager; DVCManager().track_models()'")
    print("4. Push to remote: dvc push")
    print("5. Commit changes: git add . && git commit -m 'Add DVC tracking'")

if __name__ == "__main__":
    setup_complete_dvc()