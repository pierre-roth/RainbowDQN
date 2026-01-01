import pytest
import csv
from pathlib import Path
from src.utils import MetricLogger
import shutil

@pytest.fixture
def log_dir(tmp_path):
    d = tmp_path / "logs"
    d.mkdir()
    return d

def test_logger_init(log_dir):
    logger = MetricLogger(log_dir)
    assert logger.save_log.name == "log.csv"
    assert logger.save_log.exists()
    
    # Check header
    with open(logger.save_log, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        assert header == ["Episode", "Step", "MeanReward", "MeanLength", "MeanLoss", "MeanQValue", "TimeDelta", "Time"]

def test_logger_resume(log_dir):
    # First run
    logger = MetricLogger(log_dir)
    logger.log_episode() # Log one empty episode
    logger.record(1, 100) # Flush to file
    
    # Resume
    logger2 = MetricLogger(log_dir, resume=True)
    with open(logger2.save_log, "r") as f:
        reader = csv.reader(f)
        rows = list(reader)
        # Header + 1 data row
        assert len(rows) == 2

def test_logger_record(log_dir, capsys):
    logger = MetricLogger(log_dir)
    # Simulate some data
    logger.log_step(10, 0.5, 1.0)
    logger.log_episode()
    
    logger.record(1, 100)
    
    # Verify file content
    with open(logger.save_log, "r") as f:
        reader = csv.DictReader(f)
        row = next(reader)
        assert int(row["Episode"]) == 1
        assert int(row["Step"]) == 100
        assert float(row["MeanReward"]) == 10.0
        assert float(row["MeanLoss"]) == 0.5

def test_get_last_episode(log_dir):
    logger = MetricLogger(log_dir)
    assert logger.get_last_episode() == 0
    
    logger.log_step(10, 0.5, 1.0)
    logger.log_episode()
    logger.record(1, 100)
    
    # Re-init to test reading
    logger2 = MetricLogger(log_dir, resume=True)
    assert logger2.get_last_episode() == 1
