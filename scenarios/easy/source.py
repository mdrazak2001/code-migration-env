# code_migration_env\scenarios\easy\source.py
import os

def get_config(name):
    if name == "db_host":
        return "localhost"
    elif name == "db_port":
        return 5432
    else:
        return None

def read_file(path):
    f = open(os.path.join("/data", path), "r")
    content = f.read()
    f.close()
    return content