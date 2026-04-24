import logging
import colorlog

logging.addLevelName(logging.INFO, "Info")
logging.addLevelName(logging.ERROR, "Error")
logging.addLevelName(logging.WARNING, "Warning")

SUCCESS_LEVEL = 25
logging.addLevelName(SUCCESS_LEVEL, "Success")

def success(self, message, *args, **kwargs):
    """Success logging function for logger."""
    
    if self.isEnabledFor(SUCCESS_LEVEL):
        self._log(SUCCESS_LEVEL, message, args, **kwargs)
        
logging.Logger.success = success


def get_logger(name: str = __name__) -> logging.Logger:
    """Get logger."""
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        
        formatter = colorlog.ColoredFormatter(
            "%(log_color)s%(levelname)s:%(reset)s %(message)s",
            log_colors={
                "Info": "blue",
                "Error": "red",
                "Success": "green",
                "Warning": "yellow"
            }
        )
        
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger