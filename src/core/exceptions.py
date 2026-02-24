# Exceptions Module
"""
自定义异常类
"""


class JurisAgentError(Exception):
    """JurisAgent 基础异常"""
    pass


class DocumentParseError(JurisAgentError):
    """文档解析异常"""
    def __init__(self, message: str, file_path: str = None):
        self.file_path = file_path
        super().__init__(f"文档解析错误: {message}" + (f" (文件: {file_path})" if file_path else ""))


class APIError(JurisAgentError):
    """API调用异常"""
    def __init__(self, message: str, status_code: int = None):
        self.status_code = status_code
        super().__init__(f"API错误: {message}" + (f" (状态码: {status_code})" if status_code else ""))


class AgentExecutionError(JurisAgentError):
    """Agent执行异常"""
    def __init__(self, agent_name: str, message: str):
        self.agent_name = agent_name
        super().__init__(f"Agent [{agent_name}] 执行错误: {message}")


class ValidationError(JurisAgentError):
    """数据验证异常"""
    pass
