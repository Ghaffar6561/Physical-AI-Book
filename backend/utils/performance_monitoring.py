import time
from functools import wraps
from typing import Callable, Any
from ..logging_config import logger

def monitor_performance(func: Callable) -> Callable:
    """
    Decorator to monitor the performance of functions
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        
        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Log performance metrics
            logger.info(f"{func.__name__} executed in {execution_time:.2f} seconds")
            
            # Alert if execution time is too long (more than 10 seconds)
            if execution_time > 10:
                logger.warning(f"{func.__name__} took too long: {execution_time:.2f} seconds")
            
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.2f} seconds: {str(e)}")
            raise
    
    return wrapper