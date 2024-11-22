from datetime import datetime
import os

def write_to_log(log_path, log_entry):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_entry_starter = f"**************************************************{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}**************************************************\n\n"
    log_entry_ender = "\n\n=======================================================================================================================\n\n"

    with open(log_path, 'a') as file:
        file.write(log_entry_starter + log_entry + log_entry_ender)