"""Testes unitarios do OcrAnalyzer."""

import io

import fitz
import pytest

from vgb.domain.entities import SearchTarget
from vgb.domain.enums import OccurrenceType
from vgb.domain.value_objects import Cargo, Nome
from vgb.infrastructure.ai.ocr_analyzer import OcrAnalyzer


def _make_pdf_with_text(text: str) -> bytes:
    """Cria um PDF em memoria com o texto fornecido."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 72), text, fontsize=12)
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


@pytest.fixture
def analyzer() -> OcrAnalyzer:
    return OcrAnalyzer(max_pages=10)


@pytest.fixture
def target() -> SearchTarget:
    return SearchTarget(nome=Nome("Fulano de Tal"), cargo=Cargo("Engenheiro"))


class TestOcrFindsBrokenLines:
    async def test_cargo_quebrado_em_duas_linhas(self, analyzer: OcrAnalyzer, target: SearchTarget) -> None:
        """Cenario real do dia 29/05: cargo quebrado em duas linhas no PDF."""
        text = "Declarar vago o cargo do servidor Joao Silva,\nEngenheiro\nCivil, lotado no Setor de RH"
        pdf_bytes = _make_pdf_with_text(text)

        result = await analyzer.analyze(pdf_bytes, target)

        assert result.found is True
        assert len(result.occurrences) == 1
        assert result.occurrences[0].type == OccurrenceType.CARGO

    async def test_cargo_com_espacos_duplos(self, analyzer: OcrAnalyzer, target: SearchTarget) -> None:
        """PDF gera dois espacos onde havia quebra de linha."""
        text = "Engenheiro  Civil, lotado no Setor"
        pdf_bytes = _make_pdf_with_text(text)

        result = await analyzer.analyze(pdf_bytes, target)

        assert result.found is True
        assert len(result.occurrences) == 1

    async def test_nome_e_cargo_encontrados(self, analyzer: OcrAnalyzer, target: SearchTarget) -> None:
        text = "Nomeia-se Fulano de Tal para o cargo de Engenheiro"
        pdf_bytes = _make_pdf_with_text(text)

        result = await analyzer.analyze(pdf_bytes, target)

        assert result.found is True
        assert result.occurrences[0].type == OccurrenceType.BOTH

    async def test_nao_encontra_quando_nao_existe(self, analyzer: OcrAnalyzer, target: SearchTarget) -> None:
        text = "Nomeia-se Joao da Silva para o cargo de Advogado"
        pdf_bytes = _make_pdf_with_text(text)

        result = await analyzer.analyze(pdf_bytes, target)

        assert result.found is False
        assert len(result.occurrences) == 0
