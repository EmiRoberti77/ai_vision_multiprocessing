from datetime import datetime
import os

def file_name(prefix:str, ext:str='.jpg')->str:
    now = datetime.now()
    return f"{prefix}-{now.year}-{now.month}-{now.day}T{now.hour}-{now.minute}-{now.second}{ext}"


def folder_name(prefix:str)->str:
    now = datetime.now()
    return f"{prefix}-{now.year}-{now.month}-{now.day}T{now.hour}-{now.minute}-{now.second}"


def create_run_folder_output(save_run_root:str, prefix:str)->str:
    save_run_path = os.path.join(save_run_root, folder_name(prefix=prefix))
    if not os.path.exists(save_run_path):
            os.makedirs(save_run_path)
    return save_run_path