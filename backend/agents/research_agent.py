"""Research Assistant Agent (RAG) and Explainability Agent. Thin wrappers: the logic lives in backend/rag and
backend/explainability so it stays unit-testable."""
from __future__ import annotations

from backend.agents.orchestrator import Agent, AgentError
from backend.agents.services import Services
from backend.explainability.evidence_explainer import explain_answer


class ResearchAssistantAgent(Agent):
    name = "research"

    def __init__(self, sv: Services, pipeline_factory=None):
        self.sv, self._pf, self._pipe = sv, pipeline_factory, None

    def _pipeline(self):
        if self._pipe is None:
            if self._pf:
                self._pipe = self._pf()
            else:
                from backend.rag.llm import default_chain
                from backend.rag.pipeline import RAGConfig, RAGPipeline
                self._pipe = RAGPipeline(self.sv.get_production(), default_chain(), None, RAGConfig.load())
        return self._pipe

    def actions(self):
        return {"ask": self.ask, "compare": self.compare, "research_gaps": self.research_gaps, "refresh": self.refresh}

    def ask(self, question: str, history: list | None = None, document_ids: list | None = None) -> dict:
        return {"response": self._pipeline().ask(question, history or [], document_ids or [])}

    def compare(self, document_ids: list[str], aspect: str = "methodology") -> dict:
        if len(document_ids) < 2:
            raise AgentError("compare needs at least two papers")
        return {"response": self._pipeline().ask(f"Compare the {aspect} of these papers.", [], document_ids)}

    def research_gaps(self, topic: str, document_ids: list | None = None) -> dict:
        return {"response": self._pipeline().ask(f"What research gaps and open problems can be identified across these papers on {topic}?",
                                                 [], document_ids or [])}

    def refresh(self) -> dict:
        """Make the serving path see newly indexed papers."""
        return self.sv.get_production().refresh_corpus()


class ExplainabilityAgent(Agent):
    name = "explain"

    def __init__(self, sv: Services):
        self.sv, self._exp = sv, None

    def actions(self):
        return {"explain_answer": self.explain_answer, "explain_retrieval": self.explain_retrieval}

    def explain_answer(self, response: dict) -> dict:
        return {"explanation": explain_answer(response)}

    def explain_retrieval(self, query: str, k: int = 5, with_lime: bool = False) -> dict:
        if self._exp is None:
            from backend.explainability.retrieval_explainer import RetrievalExplainer
            self._exp = RetrievalExplainer(self.sv.get_production())
        return {"explanation": self._exp.explain(query, k, with_lime)}
