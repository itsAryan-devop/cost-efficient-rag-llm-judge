from pydantic_settings import BaseSettings
from pydantic import Field, ConfigDict


class PipelineSettings(BaseSettings):
    model_config = ConfigDict(env_file=".env", extra="ignore")

    # Judge config
    judge_provider: str = Field(default="gemini", description="gemini, groq, openai, or mock")
    judge_model: str = Field(default="gemini-2.5-flash")

    # Generator config (what produced the outputs being judged)
    generator_provider: str = Field(default="groq")
    generator_model: str = Field(default="llama-3.3-70b-versatile")

    # API keys
    gemini_api_key: str = ""
    gemini_api_keys: str = ""
    groq_api_key: str = ""
    openai_api_key: str = ""
    openrouter_api_key: str = ""

    # Pipeline settings
    max_judge_retries: int = 2
    judge_temperature: float = 0.0
    audit_log_path: str = "reports/p2_audit_log.jsonl"
    report_path: str = "reports/p2_evaluation_report.json"
    csv_report_path: str = "reports/p2_results.csv"


pipeline_settings = PipelineSettings()
