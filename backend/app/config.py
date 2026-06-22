"""Application settings loaded from environment / .env."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- LLM provider (OpenAI-compatible API surface) ---
    # Aliases preserved for backwards compatibility with the older LLM_* names.
    qwen_api_key: str = ""
    qwen_api_base: str = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    qwen_model: str = "qwen3.7-plus"

    llm_api_key: str = ""
    llm_base_url: str = ""
    llm_model: str = ""

    # --- Persistence ---
    database_url: str = "sqlite:///./data/klar.db"
    upload_dir: str = "./uploads"

    # --- ChromaDB ---
    chroma_path: str = "./data/chroma"
    chroma_collection: str = "klar_knowledge"

    # --- Auth ---
    jwt_secret: str = "dev-only-change-me-please-32-bytes-min"
    session_ttl_hours: int = 24 * 7  # 7 days
    reset_token_ttl_minutes: int = 15
    cookie_name: str = "klar_session"
    cookie_secure: bool = False  # set True in production
    cookie_samesite: str = "lax"  # lax | strict | none
    dev_auth_expose_reset_token: bool = False  # opt-in only; .env enables it for dev

    # --- App ---
    app_env: str = "dev"
    allowed_origins: str = "http://localhost:3000,http://localhost:5173"
    cors_origins: str = ""  # legacy alias

    # --- Resolved properties ---

    @property
    def effective_llm_api_key(self) -> str:
        return self.qwen_api_key or self.llm_api_key

    @property
    def effective_llm_base_url(self) -> str:
        return (
            self.qwen_api_base
            or self.llm_base_url
            or "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        )

    @property
    def effective_llm_model(self) -> str:
        return self.qwen_model or self.llm_model or "qwen3.7-plus"

    @property
    def allowed_origins_list(self) -> list[str]:
        raw = self.allowed_origins or self.cors_origins
        return [o.strip() for o in raw.split(",") if o.strip()]


settings = Settings()
