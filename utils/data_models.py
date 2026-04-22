from pydantic import BaseModel

class userProteinInput(BaseModel):
    proteinId: str = None
    # proteinDataNames: list[str] = None
    # proteinDataValues: list[float]
    # recommendationTarget: str = None

class alternativeProtein(BaseModel):
    proteinId: str = None
    # proteinName: str = None
    # proteinContent: int = None
    # numberAminoAcids: int = None
    # proteinQuality: float = None

class ListAlernativeProtein(BaseModel):
    alternativeProteins: list[str] = list()
