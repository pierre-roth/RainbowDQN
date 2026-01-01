import argparse
import sys
from src.train import train, visualize
from src.utils import plot_metrics
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DQN Breakout (Rainbow)")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Start training the agent")
    train_parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    train_parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility (default: 42)")
    
    # Visualize command
    viz_parser = subparsers.add_parser("visualize", help="Visualize a trained agent")
    viz_parser.add_argument("checkpoint", type=str, nargs="?", default=None, help="Path to checkpoint file (optional)")
    
    # Plot command
    plot_parser = subparsers.add_parser("plot", help="Plot training metrics from log.csv")
    plot_parser.add_argument("log_path", type=str, nargs="?", default=None, help="Path to log.csv (optional)")
    
    args = parser.parse_args()
    
    if args.command == "train":
        train(resume_path=args.resume, seed=args.seed)
    elif args.command == "visualize":
        visualize(checkpoint_path=args.checkpoint)
    elif args.command == "plot":
        log_path = args.log_path
        if log_path is None:
            # Auto-discovery logic for plot
            checkpoints_root = Path("checkpoints")
            if checkpoints_root.exists():
                all_logs = list(checkpoints_root.glob("**/log.csv"))
                if all_logs:
                    # Sort by modification time (latest first)
                    latest_log = max(all_logs, key=lambda p: p.stat().st_mtime)
                    log_path = latest_log
                    print(f"No log path provided. Auto-found latest: {log_path}")
                else:
                    print("Error: No 'log.csv' files found in 'checkpoints/' directory.")
                    sys.exit(1)
            else:
                print("Error: 'checkpoints/' directory not found. Please train an agent first.")
                sys.exit(1)
        
        plot_metrics(log_path)
