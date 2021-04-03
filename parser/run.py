from loguru import logger
from datetime import datetime
# import create_tables
# import run_parser
import time

while True:
    logger.info("Update data", datetime.now())
    day = datetime.today().weekday()
    if day == 0:
        import create_tables
        import run_parser
        time.sleep(86400)
    else:
        time.sleep(86400)
