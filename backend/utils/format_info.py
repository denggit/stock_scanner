import logging


def stock_code_dot(code: str) -> str:
    """格式化股票代码"""
    code = code.strip()
    if '.' in code:
        return code
    elif code.startswith('6'):
        return f"sh.{code}"
    elif code.startswith('3') or code.startswith('0'):
        return f"sz.{code}"
    elif code.startswith('8') or code.startswith('4'):
        return f"bj.{code}"
    else:
        logging.warning(f"This stock code is invalid: {code}")
        return code


def stock_code_plain(code: str) -> str:
    """格式化股票代码"""
    return code.strip()[-6:]


def stock_code(code: str) -> str:
    """格式化股票代码"""
    code = code.strip()[-6:]
    if code.startswith('6'):
        return f"sh{code}"
    elif code.startswith('3') or code.startswith('0'):
        return f"sz{code}"
    elif code.startswith('8') or code.startswith('4'):
        return f"bj{code}"
    else:
        logging.warning(f"This stock code is invalid: {code}")
        return code


def time(seconds: int) -> str:
    """格式化时间"""
    hours = int(seconds // 3600)
    minutes = int(seconds % 3600 // 60)
    seconds = int(seconds % 60)

    if hours > 0:
        return f"{hours}小时{minutes}分钟{seconds}秒"
    elif minutes > 0:
        return f"{minutes}分钟{seconds}秒"
    else:
        return f"{seconds}秒"
