import time
import csv
import numpy as np
import datetime
from pathlib import Path

class MetricLogger:
    """
    Logs training metrics to a file and generates plots.
    """
    def __init__(self, save_dir: Path, resume: bool = False):
        """
        Initialize the MetricLogger.

        Args:
            save_dir (Path): Directory where logs and plots will be saved.
            resume (bool): If True, appends to existing log file. Otherwise, overwrites.
        """
        self.save_dir = save_dir
        self.save_log = save_dir / "log.csv"
        
        # Headers for CSV
        self.headers = ["Episode", "Step", "MeanReward", "MeanLength", "MeanLoss", "MeanQValue", "TimeDelta", "Time"]
        
        mode = "a" if resume and self.save_log.exists() else "w"
        
        # If writing new file, write headers
        write_header = not (resume and self.save_log.exists())
        
        with open(self.save_log, mode, newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(self.headers)

        self.ep_rewards_plot = save_dir / "reward_plot.jpg"
        self.ep_lengths_plot = save_dir / "length_plot.jpg"
        self.ep_avg_losses_plot = save_dir / "loss_plot.jpg"
        self.ep_avg_qs_plot = save_dir / "q_plot.jpg"

        # History metrics
        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_avg_losses = []
        self.ep_avg_qs = []

        # Current episode metric
        self.init_episode()

        # Timing
        self.record_time = time.time()
        
    def init_episode(self) -> None:
        """Reset current episode metrics."""
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_loss = 0.0
        self.curr_ep_q = 0.0
        self.curr_ep_loss_length = 0

    def log_step(self, reward: float, loss: float | None, q_value: float | None) -> None:
        """Log metrics for a single step."""
        self.curr_ep_reward += reward
        self.curr_ep_length += 1
        if loss is not None:
            self.curr_ep_loss += loss
            self.curr_ep_q += q_value
            self.curr_ep_loss_length += 1

    def log_episode(self) -> None:
        """Mark end of episode and record stats."""
        self.ep_rewards.append(self.curr_ep_reward)
        self.ep_lengths.append(self.curr_ep_length)
        if self.curr_ep_loss_length == 0:
            ep_avg_loss = 0
            ep_avg_q = 0
        else:
            ep_avg_loss = self.curr_ep_loss / self.curr_ep_loss_length
            ep_avg_q = self.curr_ep_q / self.curr_ep_loss_length
        self.ep_avg_losses.append(ep_avg_loss)
        self.ep_avg_qs.append(ep_avg_q)

        self.init_episode()

    def get_last_episode(self) -> int:
        """Parses the log file to find the last episode number."""
        if not self.save_log.exists():
            return 0
        try:
            with open(self.save_log, "r", newline='') as f:
                reader = csv.reader(f)
                rows = list(reader)
                if not rows or len(rows) < 2: return 0
                
                # Header is row 0
                last_row = rows[-1]
                if not last_row: return 0
                
                return int(last_row[0])
        except Exception as e:
            print(f"Error parsing log file: {e}")
            return 0

    def record(self, episode: int, step: int) -> None:
        """Print metrics to console and append to log file."""
        mean_ep_reward = np.round(np.mean(self.ep_rewards[-100:]), 3)
        mean_ep_length = np.round(np.mean(self.ep_lengths[-100:]), 3)
        mean_ep_loss = np.round(np.mean(self.ep_avg_losses[-100:]), 3)
        mean_ep_q = np.round(np.mean(self.ep_avg_qs[-100:]), 3)
        
        last_record_time = self.record_time
        self.record_time = time.time()
        time_since_last_record = np.round(self.record_time - last_record_time, 3)

        print(
            f"Episode {episode} - "
            f"Step {step} - "
            f"Mean Reward {mean_ep_reward} - "
            f"Mean Length {mean_ep_length} - "
            f"Mean Loss {mean_ep_loss} - "
            f"Mean Q Value {mean_ep_q} - "
            f"Time Delta {time_since_last_record} - "
            f"Time {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}"
        )

        with open(self.save_log, "a", newline='') as f:
             writer = csv.writer(f)
             writer.writerow([
                 episode,
                 step,
                 mean_ep_reward,
                 mean_ep_length,
                 mean_ep_loss,
                 mean_ep_q,
                 time_since_last_record,
                 datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
             ])

    def log_eval(self, step: int, avg_reward: float) -> None:
        """Log evaluation results to a separate file."""
        eval_log_path = self.save_dir / "eval_log.csv"
        write_header = not eval_log_path.exists()
        
        with open(eval_log_path, "a", newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["Step", "AvgReward"])
            writer.writerow([step, avg_reward])

def plot_metrics(log_path):
    """
    Plot metrics from log file using matplotlib.
    Lazy imports matplotlib to avoid overhead during training.
    """
    import matplotlib.pyplot as plt
    
    log_path = Path(log_path)
    if not log_path.exists():
        print(f"Error: Log file not found at {log_path}")
        return

    data = {
        "Episode": [],
        "Step": [],
        "MeanReward": [],
        "MeanLength": [],
        "MeanLoss": [],
        "MeanQValue": [],
    }

    try:
        with open(log_path, "r", newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    data["Episode"].append(int(row["Episode"]))
                    data["Step"].append(int(row["Step"]))
                    data["MeanReward"].append(float(row["MeanReward"]))
                    data["MeanLength"].append(float(row["MeanLength"]))
                    data["MeanLoss"].append(float(row["MeanLoss"]))
                    data["MeanQValue"].append(float(row["MeanQValue"]))
                except ValueError:
                    continue
    except Exception as e:
         print(f"Error reading log file currently: {e}")
         print("Attempting legacy fallback (space-separated)...")
         # Fallback to try reading old space separated format
         try:
            with open(log_path, "r") as f:
                lines = f.readlines()
                for line in lines[1:]: # Skip header
                    parts = line.split()
                    if len(parts) >= 6:
                        data["Episode"].append(int(parts[0]))
                        data["Step"].append(int(parts[1]))
                        data["MeanReward"].append(float(parts[2]))
                        data["MeanLength"].append(float(parts[3]))
                        data["MeanLoss"].append(float(parts[4]))
                        data["MeanQValue"].append(float(parts[5]))
         except Exception as e2:
             print(f"Legacy fallback failed: {e2}")
             return

    if not data["Episode"]:
        print("Error: No valid data found in log file.")
        return

    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f"Training Progress - {log_path.parent.name}", fontsize=16)

    # Reward Plot
    axs[0, 0].plot(data["Episode"], data["MeanReward"], label="Mean Reward (last 100)")
    axs[0, 0].set_title("Mean Reward")
    axs[0, 0].set_xlabel("Episode")
    axs[0, 0].set_ylabel("Reward")
    axs[0, 0].grid(True)
    axs[0, 0].legend()

    # Loss Plot
    axs[0, 1].plot(data["Episode"], data["MeanLoss"], label="Mean Loss (last 100)", color="orange")
    axs[0, 1].set_title("Mean Loss")
    axs[0, 1].set_xlabel("Episode")
    axs[0, 1].set_ylabel("Loss")
    axs[0, 1].grid(True)
    axs[0, 1].legend()

    # Q-Value Plot
    axs[1, 0].plot(data["Episode"], data["MeanQValue"], label="Mean Q-Value (last 100)", color="green")
    axs[1, 0].set_title("Mean Q-Value")
    axs[1, 0].set_xlabel("Episode")
    axs[1, 0].set_ylabel("Q-Value")
    axs[1, 0].grid(True)
    axs[1, 0].legend()

    # Length Plot (Filling the 4th slot)
    axs[1, 1].plot(data["Episode"], data["MeanLength"], label="Mean Length (last 100)", color="purple")
    axs[1, 1].set_title("Mean Episode Length")
    axs[1, 1].set_xlabel("Episode")
    axs[1, 1].set_ylabel("Steps")
    axs[1, 1].grid(True)
    axs[1, 1].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    plot_path = log_path.parent / "training_plots.png"
    plt.savefig(plot_path)
    print(f"Plots saved to {plot_path}")
    
    # Try to show the plot if possible
    try:
        plt.show()
    except Exception:
        pass
