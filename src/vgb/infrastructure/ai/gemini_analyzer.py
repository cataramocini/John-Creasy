"""Analisador usando Google Gemini 2.5 Flash (multimodal)."""

import json
import re
import time

import structlog
from google import genai
from google.genai import types
from pydantic import BaseModel, Field, ValidationError

from vgb.application.ports.ai_analyzer import AIAnalysisResult, AIOccurrence, PDFAnalyzer
from vgb.domain.entities import SearchTarget
from vgb.domain.enums import AnalysisModel, OccurrenceType
from vgb.domain.exceptions import AnaliseIndisponivelError
from vgb.infrastructure.config import Settings

logger = structlog.get_logger()


class _OccurrenceSchema(BaseModel):
    type: str = Field(description="Um de: NOME, CARGO, BOTH")
    context: str = Field(
        description="Resumo do ato em linguagem natural. Seja completo. "
        "Exemplo: 'Fulano de Tal foi nomeado para o cargo de Apoio de Saneamento.'"
    )
    page: int | None = Field(description="Numero da pagina, se identificavel", default=None)
    confidence: float = Field(description="Confianca de 0.0 a 1.0")


class _ResultSchema(BaseModel):
    found: bool = Field(description="True se encontrou o nome ou cargo em contexto relevante")
    occurrences: list[_OccurrenceSchema] = Field(default_factory=list)


class GeminiAnalyzer(PDFAnalyzer):
    """Analisa PDFs usando Google Gemini multimodal com structured output."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        if not settings.gemini_api_key:
            raise AnaliseIndisponivelError("Gemini API key nao configurada")

        self._client = genai.Client(api_key=settings.gemini_api_key.get_secret_value())

    async def analyze(self, pdf_bytes: bytes, target: SearchTarget) -> AIAnalysisResult:
        start = time.monotonic()
        logger.info("gemini.analyze.start", nome=target.nome.valor, cargo=target.cargo.valor)

        try:
            prompt = self._build_prompt(target)
            response = await self._client.aio.models.generate_content(
                model="gemini-2.5-flash",
                contents=types.Content(
                    parts=[
                        types.Part.from_text(text=prompt),
                        types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
                    ]
                ),
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=_ResultSchema,
                    temperature=0.1,
                    max_output_tokens=2048,
                ),
            )

            raw_text = response.text or "{}"
            result = self._parse_response(raw_text)

            occurrences = [
                AIOccurrence(
                    type=(
                        OccurrenceType(o.type.lower())
                        if o.type.lower() in {e.value for e in OccurrenceType}
                        else OccurrenceType.NOME
                    ),
                    context=o.context,
                    page=o.page,
                    confidence=max(0.0, min(1.0, o.confidence)),
                )
                for o in result.occurrences
            ]

            elapsed_ms = int((time.monotonic() - start) * 1000)
            logger.info(
                "gemini.analyze.done",
                found=result.found,
                occurrences=len(occurrences),
                elapsed_ms=elapsed_ms,
            )

            return AIAnalysisResult(
                found=result.found,
                occurrences=occurrences,
                raw_response=raw_text,
                model_used=AnalysisModel.GEMINI_25_FLASH.value,
                confidence_score=max((o.confidence for o in occurrences), default=0.0),
                processing_time_ms=elapsed_ms,
            )
        except Exception as exc:
            logger.error("gemini.analyze.error", error=str(exc))
            raise AnaliseIndisponivelError(f"Gemini falhou: {exc}") from exc

    @staticmethod
    def _parse_response(raw_text: str) -> _ResultSchema:
        """Tenta parsear o JSON do Gemini, mesmo que truncado ou malformado."""
        # 1) Tenta parse direto com Pydantic
        try:
            return _ResultSchema.model_validate_json(raw_text)
        except ValidationError:
            pass

        # 2) Tenta parse com json.loads + model_validate
        try:
            data = json.loads(raw_text)
            return _ResultSchema.model_validate(data)
        except (json.JSONDecodeError, ValidationError):
            pass

        # 3) Tenta extrair o campo 'found' com regex e criar resultado mínimo
        # Isso lida com JSONs truncados que ainda contêm 'found'
        found_match = re.search(r'"found"\s*:\s*(true|false)', raw_text, re.IGNORECASE)
        if found_match:
            found = found_match.group(1).lower() == "true"
            logger.warning("gemini.json_truncated", raw_preview=raw_text[:200])
            return _ResultSchema(found=found, occurrences=[])

        # Se nada funcionou, falha para que o fallback seja acionado
        raise ValidationError.from_exception_data(
            title="JSON inválido do Gemini",
            line_errors=[{"loc": ("root",), "msg": f"Não foi possível parsear: {raw_text[:200]}", "type": "value_error"}],
        )

    def _build_prompt(self, target: SearchTarget) -> str:
        return (
            "Voce e um assistente juridico especializado em Diarios Oficiais municipais. "
            "Analise este Diario Oficial e verifique se ha qualquer mencao ao nome ou cargo.\n\n"
            f"Nome a buscar: {target.nome.valor}\n"
            f"Cargo de interesse: {target.cargo.valor}\n\n"
            "Instrucoes:\n"
            "- 'found' deve ser true SEMPRE que o nome ou cargo aparecerem no documento, "
            "  independentemente do contexto (nomeacao, exoneracao, lista de presenca, folha de pagamento, etc.).\n"
            "- context: em vez de trecho exato, gere um resumo completo em portugues explicando "
            "  o que aconteceu com a pessoa/cargo. Seja detalhado. Exemplo: 'Fulano de Tal foi nomeado para "
            "  o cargo de Apoio de Saneamento mediante portaria nº 123/2026.'\n"
            "- confidence: certeza de que a mencao foi encontrada (0.0 a 1.0).\n"
            "- Retorne JSON valido seguindo o schema fornecido."
        )
