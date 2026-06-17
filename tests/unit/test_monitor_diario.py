"""Testes unitarios do MonitorDiarioUseCase."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from vgb.application.ports.ai_analyzer import AIAnalysisResult, AIOccurrence
from vgb.application.ports.source import SourceLink
from vgb.application.use_cases.monitor_diario import MonitorDiarioUseCase
from vgb.domain.enums import AnalysisModel, OccurrenceType
from vgb.infrastructure.config.settings import Settings


@pytest.fixture
def settings() -> Settings:
    return Settings(
        nome_busca="Maria",
        cargo_busca="Engenheiro",
        telegram_token="fake",
        telegram_chat_id="123",
        diario_url="https://example.gov.br/jornal/",
        diario_base_url="https://example.gov.br",
    )


@pytest.fixture
def use_case(settings: Settings) -> MonitorDiarioUseCase:
    source = AsyncMock()
    analyzer = AsyncMock()
    notifier = AsyncMock()
    edition_repo = AsyncMock()
    analysis_repo = AsyncMock()
    http_client = AsyncMock()

    return MonitorDiarioUseCase(
        source=source,
        analyzer=analyzer,
        notifier=notifier,
        edition_repo=edition_repo,
        analysis_repo=analysis_repo,
        http_client=http_client,
        settings=settings,
    )


def _make_response(content: bytes) -> MagicMock:
    resp = MagicMock()
    resp.content = content
    resp.raise_for_status = MagicMock()
    return resp


class TestExecute:
    async def test_fluxo_com_ocorrencia(self, use_case: MonitorDiarioUseCase) -> None:
        use_case._source.fetch_links.return_value = [SourceLink(title="DO 01", url="https://exemplo.gov.br/do.pdf")]
        use_case._edition_repo.get_processed_urls.return_value = set()
        use_case._edition_repo.get_by_url.return_value = None
        use_case._http.head.return_value = MagicMock(headers={"content-length": "1024"})
        use_case._http.get.return_value = _make_response(b"pdfdata")
        use_case._edition_repo.get_by_hash.return_value = None
        use_case._analyzer.analyze.return_value = AIAnalysisResult(
            found=True,
            occurrences=[
                AIOccurrence(
                    type=OccurrenceType.BOTH,
                    context="NOMEIA MARIA",
                    confidence=0.95,
                )
            ],
            model_used=AnalysisModel.GEMINI_25_FLASH.value,
            processing_time_ms=1234,
        )
        use_case._analysis_repo.save.side_effect = lambda a: a

        stats = await use_case.execute()

        assert stats["checked"] == 1
        assert stats["processed"] == 1
        assert stats["new"] == 1
        assert stats["found"] == 1
        assert stats["notified"] == 1
        use_case._notifier.send.assert_not_called()
        use_case._notifier.send_summary.assert_awaited_once()

        payload = use_case._notifier.send_summary.await_args.args[0]
        assert payload.total_found == 1
        assert len(payload.occurrences) == 1
        assert payload.occurrences[0].edition_title == "DO 01"
        assert payload.occurrences[0].context_snippet == "NOMEIA MARIA"

    async def test_fluxo_sem_ocorrencia(self, use_case: MonitorDiarioUseCase) -> None:
        use_case._source.fetch_links.return_value = [SourceLink(title="DO 01", url="https://exemplo.gov.br/do.pdf")]
        use_case._edition_repo.get_processed_urls.return_value = set()
        use_case._edition_repo.get_by_url.return_value = None
        use_case._http.head.return_value = MagicMock(headers={"content-length": "1024"})
        use_case._http.get.return_value = _make_response(b"pdfdata")
        use_case._edition_repo.get_by_hash.return_value = None
        use_case._analyzer.analyze.return_value = AIAnalysisResult(
            found=False,
            occurrences=[],
            model_used=AnalysisModel.GEMINI_25_FLASH.value,
            processing_time_ms=500,
        )
        use_case._analysis_repo.save.side_effect = lambda a: a

        stats = await use_case.execute()

        assert stats["checked"] == 1
        assert stats["processed"] == 1
        assert stats["new"] == 1
        assert stats["found"] == 0
        assert stats["notified"] == 0
        use_case._notifier.send.assert_not_called()
        use_case._notifier.send_summary.assert_awaited_once()

    async def test_falha_de_link_nao_quebra_outros(self, use_case: MonitorDiarioUseCase) -> None:
        use_case._source.fetch_links.return_value = [
            SourceLink(title="DO 01", url="https://exemplo.gov.br/do1.pdf"),
            SourceLink(title="DO 02", url="https://exemplo.gov.br/do2.pdf"),
        ]
        use_case._edition_repo.get_processed_urls.return_value = set()
        # Primeiro link: falha no download
        use_case._edition_repo.get_by_url.side_effect = [None, None]
        use_case._http.head.side_effect = [
            MagicMock(headers={"content-length": "1024"}),
            Exception("Timeout"),
        ]
        use_case._http.get.return_value = _make_response(b"pdfdata")
        use_case._edition_repo.get_by_hash.return_value = None
        use_case._analyzer.analyze.return_value = AIAnalysisResult(
            found=False,
            occurrences=[],
            model_used=AnalysisModel.OCR_LOCAL.value,
            processing_time_ms=100,
        )

        stats = await use_case.execute()

        assert stats["checked"] == 2
        assert stats["processed"] == 1
        assert stats["errors"] == 1
        use_case._notifier.send_summary.assert_awaited_once()

    async def test_source_falha_envia_resumo(self, use_case: MonitorDiarioUseCase) -> None:
        use_case._source.fetch_links.side_effect = Exception("Site fora do ar")

        stats = await use_case.execute()

        assert stats["errors"] == 1
        use_case._notifier.send_summary.assert_awaited_once()
        use_case._notifier.send.assert_not_called()

    async def test_so_analisa_pdf_novo_quando_8_ja_existem(self, use_case: MonitorDiarioUseCase) -> None:
        """Cenario real: 8 PDFs no banco, so 1 e novo. Deve pular os 7 antigos."""
        antigos = [SourceLink(title=f"DO {i:02d}", url=f"https://exemplo.gov.br/do{i}.pdf") for i in range(1, 8)]
        novo = SourceLink(title="DO 08", url="https://exemplo.gov.br/do8.pdf")
        use_case._source.fetch_links.return_value = antigos + [novo]

        # Simula 7 edicoes ja processadas no banco
        use_case._edition_repo.get_processed_urls.return_value = {link.url for link in antigos}
        use_case._edition_repo.get_by_url.return_value = None
        use_case._http.head.return_value = MagicMock(headers={"content-length": "1024"})
        use_case._http.get.return_value = _make_response(b"pdfdata_novo")
        use_case._edition_repo.get_by_hash.return_value = None
        use_case._analyzer.analyze.return_value = AIAnalysisResult(
            found=False,
            occurrences=[],
            model_used=AnalysisModel.OCR_LOCAL.value,
            processing_time_ms=100,
        )
        use_case._analysis_repo.save.side_effect = lambda a: a

        stats = await use_case.execute()

        assert stats["checked"] == 8
        assert stats["processed"] == 1
        assert stats["new"] == 1
        assert stats["found"] == 0
        # O analyzer deve ser chamado APENAS 1 vez (so para o novo)
        use_case._analyzer.analyze.assert_awaited_once()
        use_case._notifier.send_summary.assert_awaited_once()

    async def test_nao_envia_resumo_quando_nenhum_pdf_novo(self, use_case: MonitorDiarioUseCase) -> None:
        """Se todos os PDFs ja foram processados, nao deve enviar resumo no Telegram."""
        antigos = [SourceLink(title=f"DO {i:02d}", url=f"https://exemplo.gov.br/do{i}.pdf") for i in range(1, 9)]
        use_case._source.fetch_links.return_value = antigos
        use_case._edition_repo.get_processed_urls.return_value = {link.url for link in antigos}

        stats = await use_case.execute()

        assert stats["checked"] == 8
        assert stats["processed"] == 0
        assert stats["new"] == 0
        assert stats["found"] == 0
        use_case._analyzer.analyze.assert_not_called()
        use_case._notifier.send_summary.assert_not_called()

    async def test_busca_todos_links_e_analisa_so_novos(self, use_case: MonitorDiarioUseCase) -> None:
        """Pagina com 12 links, 10 antigos no banco, 2 novos → analisa apenas os 2 novos."""
        antigos = [SourceLink(title=f"DO {i:02d}", url=f"https://exemplo.gov.br/do{i}.pdf") for i in range(1, 11)]
        novos = [
            SourceLink(title="DO 11", url="https://exemplo.gov.br/do11.pdf"),
            SourceLink(title="DO 12", url="https://exemplo.gov.br/do12.pdf"),
        ]
        use_case._source.fetch_links.return_value = antigos + novos
        use_case._edition_repo.get_processed_urls.return_value = {link.url for link in antigos}
        use_case._edition_repo.get_by_url.return_value = None
        use_case._http.head.return_value = MagicMock(headers={"content-length": "1024"})
        use_case._http.get.return_value = _make_response(b"pdfdata_novo")
        use_case._edition_repo.get_by_hash.return_value = None
        use_case._analyzer.analyze.return_value = AIAnalysisResult(
            found=False,
            occurrences=[],
            model_used=AnalysisModel.OCR_LOCAL.value,
            processing_time_ms=100,
        )
        use_case._analysis_repo.save.side_effect = lambda a: a

        stats = await use_case.execute()

        assert stats["checked"] == 12
        assert stats["processed"] == 2
        assert stats["new"] == 2
        assert stats["found"] == 0
        assert use_case._analyzer.analyze.await_count == 2
        use_case._notifier.send_summary.assert_awaited_once()

    async def test_respeita_limite_max_pdfs_per_run(self, use_case: MonitorDiarioUseCase) -> None:
        """Pagina com 15 novos, max_pdfs_per_run=8 → analisa apenas 8."""
        novos = [SourceLink(title=f"DO {i:02d}", url=f"https://exemplo.gov.br/do{i}.pdf") for i in range(1, 16)]
        use_case._source.fetch_links.return_value = novos
        use_case._edition_repo.get_processed_urls.return_value = set()
        use_case._edition_repo.get_by_url.return_value = None
        use_case._http.head.return_value = MagicMock(headers={"content-length": "1024"})
        use_case._http.get.return_value = _make_response(b"pdfdata")
        use_case._edition_repo.get_by_hash.return_value = None
        use_case._analyzer.analyze.return_value = AIAnalysisResult(
            found=False,
            occurrences=[],
            model_used=AnalysisModel.OCR_LOCAL.value,
            processing_time_ms=100,
        )
        use_case._analysis_repo.save.side_effect = lambda a: a

        stats = await use_case.execute()

        assert stats["checked"] == 15
        assert stats["processed"] == 8
        assert stats["new"] == 8
        assert stats["found"] == 0
        assert use_case._analyzer.analyze.await_count == 8
        use_case._notifier.send_summary.assert_awaited_once()
