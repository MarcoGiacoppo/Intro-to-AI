import subprocess
import os

# Config
algorithms = ['DFS', 'BFS', 'GBFS', 'AS', 'CUS1', 'CUS2']
test_files = [f"Test_{i + 1}.txt" for i in range(20)]
output_file = "batch_results.txt"

with open(output_file, "w") as log:
    for test_file in test_files:
        if not os.path.exists(test_file):
            log.write(f"❌ {test_file} not found. Skipping.\n")
            continue

        for method in algorithms:
            log.write(f"🔍 Running: {test_file} with {method}\n")
            try:
                result = subprocess.run(
                    ["python3", "search.py", test_file, method],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                log.write(result.stdout + "\n")
                if result.stderr:
                    log.write("⚠️ STDERR:\n" + result.stderr + "\n")
            except subprocess.TimeoutExpired:
                log.write(f"⏱️ Timeout: {test_file} with {method}\n")
            except Exception as e:
                log.write(f"💥 Error running {test_file} with {method}: {e}\n")

            log.write("-" * 60 + "\n")

print(f"✅ All results saved to {output_file}")
