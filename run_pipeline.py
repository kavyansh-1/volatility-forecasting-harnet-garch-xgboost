"""
Master Execution Script. Runs the pipeline sequentially from Day 1 to Day 7.
"""
import subprocess
import sys
import os

scripts_to_run = [
    "day01/src/day01_setup.py",
    "day02/src/day02_returns.py",
    "day02/src/day02_volatility.py",
    "day03/src/day03_stationarity.py",
    "day04/src/day04_features.py",
    "day04/src/day04_har.py",
    "day06/src/day06_fetch_news.py",
    "day06/src/day06_sentiment.py",
    "day07/src/day07_merge_sentiment.py",
    "day07/src/day07_har_sentiment.py"
]

def main():
    print("Starting 60-Day Volatility Forecasting Pipeline...\n" + "="*50)
    for script in scripts_to_run:
        print(f"Executing: {script}")
        if not os.path.exists(script):
            print(f"File missing: {script}. Skipping.")
            continue
            
        # Run the script and stream the output to console
        result = subprocess.run([sys.executable, script], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"\n[ERROR] {script} failed with error:\n{result.stderr}")
            print("Pipeline Halted.")
            return
        else:
            print(result.stdout.strip())
            print("-" * 50)
            
    print("Pipeline execution completed successfully.")

if __name__ == "__main__":
    main()
