import re
from typing import Optional

def sanitize_input(text: Optional[str]) -> Optional[str]:
    """
    Sanitize user input to prevent injection attacks
    """
    if not text:
        return text
    
    # Remove potentially dangerous characters/sequences
    # This is a basic implementation - in production, use a more robust library
    sanitized = text
    
    # Remove potential script tags (case insensitive)
    sanitized = re.sub(r'<script[^>]*>.*?</script>', '', sanitized, flags=re.IGNORECASE | re.DOTALL)
    
    # Remove potential iframe tags
    sanitized = re.sub(r'<iframe[^>]*>.*?</iframe>', '', sanitized, flags=re.IGNORECASE | re.DOTALL)
    
    # Remove javascript: urls
    sanitized = re.sub(r'javascript:', '', sanitized, flags=re.IGNORECASE)
    
    # Remove data: urls
    sanitized = re.sub(r'data:', '', sanitized, flags=re.IGNORECASE)
    
    # Remove potential SQL injection patterns
    sql_patterns = [
        r"(?i)(union\s+select)",
        r"(?i)(drop\s+table)",
        r"(?i)(delete\s+from)",
        r"(?i)(insert\s+into)",
        r"(?i)(update\s+\w+\s+set)",
        r"(?i)(exec\s*\()",
        r"(?i)(execute\s*\()",
        r"(?i)(sp_)",
        r"(?i)(xp_)",
        r"(?i)(0x[0-9a-f]+)"
    ]
    
    for pattern in sql_patterns:
        sanitized = re.sub(pattern, '', sanitized)
    
    return sanitized.strip()