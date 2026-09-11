"""MCP 실행과 사용자 추가 입력에 사용하는 공통 예외."""


class McpParameterInputRequired(ValueError):
    """MCP 호출 전 사용자에게 추가 입력을 받아야 함을 나타낸다."""

    def __init__(
        self,
        *,
        input_code: str,
        parameter_name: str,
        label: str,
        message: str,
        input_type: str = "text",
        expected_value: str | None = None,
        pattern: str | None = None,
        min_length: int | None = None,
        max_length: int | None = None,
        allowed_values: list[str] | None = None,
        validation_message: str | None = None,
        sensitive: bool = False,
        initial_error: str | None = None,
    ) -> None:
        self.input_code = input_code
        self.parameter_name = parameter_name
        self.label = label
        self.message = message
        self.input_type = input_type
        self.expected_value = expected_value
        self.pattern = pattern
        self.min_length = min_length
        self.max_length = max_length
        self.allowed_values = allowed_values
        self.validation_message = validation_message
        self.sensitive = sensitive
        self.initial_error = initial_error
        super().__init__(message)

