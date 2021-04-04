from loguru import logger
from datetime import datetime
import subprocess
import time


while True:
    logger.info("Update data", datetime.now())
    day = datetime.today().weekday()
    if day == 0:
        import create_tables 
        subprocess.call(['python3', 'run_parser.py'])  
        time.sleep(86400)
    else:
        time.sleep(86400)
        
