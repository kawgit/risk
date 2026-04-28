import os
import subprocess
from collections import defaultdict

def get_cs440_users(directory="/usr4/cs440"):
    """Scans the target directory and returns a set of directory names."""
    try:
        # os.scandir is fast and lets us explicitly filter for directories
        with os.scandir(directory) as entries:
            return {entry.name for entry in entries if entry.is_dir()}
    except (FileNotFoundError, PermissionError) as e:
        print(f"Warning: Could not read {directory} -> {e}")
        return set()

def get_qstat_data(valid_users):
    try:
        result = subprocess.run(
            ['qstat', '-u', '*'], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            universal_newlines=True, 
            check=True
        )
        lines = result.stdout.strip().split('\n')
    except subprocess.CalledProcessError as e:
        print(f"Error: qstat command failed. {e.stderr}")
        return None, None

    running_cores = defaultdict(int)
    total_cores = defaultdict(int)

    for line in lines:
        parts = line.split()
        
        if not parts or not parts[0].isdigit():
            continue
            
        try:
            user = parts[3]
            
            # THE FILTER: Skip this job if the user isn't in the cs440 directory
            if user not in valid_users:
                continue

            state = parts[4].lower()
            
            slots = 1
            for p in parts[7:]:
                if p.isdigit():
                    slots = int(p)
                    break

            total_cores[user] += slots
            if 'r' in state:
                running_cores[user] += slots
                
        except (ValueError, IndexError):
            continue
            
    return running_cores, total_cores

def print_leaderboards():
    # 1. Grab the allowed users first
    valid_users = get_cs440_users()
    if not valid_users:
        print("No users found to filter against. Exiting.")
        return

    # 2. Pass them to the parser
    running, total = get_qstat_data(valid_users)
    if running is None: return

    # 3. Sort and slice for Top 5
    top_running = sorted(running.items(), key=lambda x: x[1], reverse=True)[:5]
    top_total = sorted(total.items(), key=lambda x: x[1], reverse=True)[:5]

    # --- Table 1: Running Cores ---
    print("\n" + "="*40)
    print(f"{'CS 440 TOP 5: CURRENTLY RUNNING':^40}")
    print("="*40)
    print(f"{'USER':<20} | {'SLOTS':<10}")
    print("-" * 40)
    for user, count in top_running:
        print(f"{user:<20} | {count:<10}")
    if not top_running:
        print(f"{'No active running jobs.':^40}")

    # --- Table 2: Total Cores ---
    print("\n" + "="*40)
    print(f"{'CS 440 TOP 5: TOTAL REQUESTED':^40}")
    print("="*40)
    print(f"{'USER':<20} | {'SLOTS':<10}")
    print("-" * 40)
    for user, count in top_total:
        print(f"{user:<20} | {count:<10}")
    if not top_total:
        print(f"{'No active or queued jobs.':^40}")

if __name__ == "__main__":
    print_leaderboards()