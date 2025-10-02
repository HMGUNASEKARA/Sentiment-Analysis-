import logging
import sys

logging.basicConfig(
    level = logging.DEBUG,
    format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # This haddle the log format 
    handlers=[
        logging.FileHandler(filename='log.log',mode='w'), # This save the massage in the log.log file
        logging.StreamHandler(sys.stdout) # This print the massage in the console
    ]

)