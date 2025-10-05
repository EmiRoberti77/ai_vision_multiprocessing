from datetime import datetime

def file_name(prefix:str, ext:str='.jpg')->str:
    now = datetime.now()
    return f"{prefix}-{now.year}-{now.month}-{now.day}T{now.hour}-{now.minute}-{now.second}{ext}"

def folder_name(prefix:str)->str:
    now = datetime.now()
    return f"{prefix}-{now.year}-{now.month}-{now.day}T{now.hour}-{now.minute}-{now.second}"