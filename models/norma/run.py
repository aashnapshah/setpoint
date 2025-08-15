import subprocess
import sys

experiments = [
    {"lambda_align": 0.1, "adaptive_weight": False},
    {"lambda_align": 0.1, "adaptive_weight": True},
    {"lambda_align": 0.01, "adaptive_weight": False}
]

for i, exp in enumerate(experiments, 1):
    print(f"\n{'='*50}")
    print(f"Running Experiment {i}/{len(experiments)}")
    print(f"lambda_align={exp['lambda_align']}, adaptive_weight={exp['adaptive_weight']}")
    print(f"{'='*50}")
    
    cmd = [
        "python", "train.py",
        "--epochs", "15",
        "--lambda_align", str(exp['lambda_align']),
        "--adaptive_weight", str(exp['adaptive_weight'])
    ]
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"✅ Experiment {i} completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Experiment {i} failed with error: {e}")
        
print("\n🎉 All experiments completed!")
