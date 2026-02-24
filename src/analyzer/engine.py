class DatasetAnalyzer:
    def __init__(self, ollama_model: str = 'llama3.1:8b'):
        self.model = ollama_model

    def analyze_dialog(self, dialog: list[dict]) -> dict:
        """
        Analyze a single dialog.
        To be implemented.
        """
        return {
            'intent': 'other',
            'satisfaction': 'neutral',
            'quality_score': 3,
            'agent_mistakes': []
        }
