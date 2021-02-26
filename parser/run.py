from loguru import logger
from datetime import datetime
from parser import run_parser
from table import create_table


create_table()
while True:
    logger.info("Update data", datetime.now())
    day = datetime.today().weekday()
    if day == 0:
        run_parser()
        time.sleep(86400)
    else:
        time.sleep(86400)
