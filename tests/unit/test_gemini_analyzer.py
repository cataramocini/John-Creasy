"""Testes unitarios do GeminiAnalyzer."""

import pytest
from pydantic import ValidationError

from vgb.infrastructure.ai.gemini_analyzer import GeminiAnalyzer


class TestParseResponse:
    def test_json_valido(self) -> None:
        raw = '{"found": true, "occurrences": [{"type": "CARGO", "context": "nomeado", "page": 1, "confidence": 0.9}]}'
        result = GeminiAnalyzer._parse_response(raw)
        assert result.found is True
        assert len(result.occurrences) == 1
        assert result.occurrences[0].type == "CARGO"

    def test_json_truncado_mas_com_found_true(self) -> None:
        """Cenario real: Gemini retorna JSON cortado no meio."""
        raw = '{\n  "found": true,\n  "context": "GABRIEL PEREIRA GAZOTTO, com matr'
        result = GeminiAnalyzer._parse_response(raw)
        assert result.found is True
        assert len(result.occurrences) == 0

    def test_json_truncado_com_found_false(self) -> None:
        raw = '{\n  "found": false,\n  "context": "Nada encontrado'
        result = GeminiAnalyzer._parse_response(raw)
        assert result.found is False

    def test_json_invalido_sem_found(self) -> None:
        raw = "não é json"
        with pytest.raises(ValidationError):
            GeminiAnalyzer._parse_response(raw)
