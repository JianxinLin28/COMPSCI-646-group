class MyLogger:
    _instance = None  # class-level variable

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _log(self, level: str, message: str, indent: int):
        if message != "" and message != None:
            print(f'{"    " * indent}[{level}] {message}')
        else:
            print()

    def info(self, message: str="", indent: int=0):
        # self._log("INFO", message, indent)
        pass
    
    def warn(self, message: str="", indent: int=0):
        self._log("WARN", message, indent)
    
    def error(self, message: str="", indent: int=0):
        self._log("ERROR", message, indent)
    
    def debug(self, message: str="", indent: int=0):
        self._log("DEBUG", message, indent)
