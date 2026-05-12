"""
Enumerations centralisées pour le module core.

Avoir des enums au lieu de strings magiques évite :
    - les fautes de frappe silencieuses ('synthetic-data' vs 'synthetic_data')
    - la duplication de la string littérale dans tout le code
    - la difficulté à renommer (un seul endroit à changer ici)

Convention :
    - les valeurs (member.value) correspondent à la string utilisée dans le
      fichier TOML / la config. C'est la frontière avec le monde externe.
    - le code Python ne devrait JAMAIS comparer à des strings littérales :
      utiliser DataMode.SYNTHETIC, PipelineState.RUNNING, etc.
"""

from enum import Enum


class DataMode(Enum):
    """
    Mode des données d'entrée d'une reconstruction.

    REAL       : on reconstruit à partir d'une mesure expérimentale, on n'a
                 pas de vérité terrain.
    SYNTHETIC  : on connaît la 'vraie' image f_true et on simule les mesures
                 g via H @ f_true. Permet le calcul de métriques de qualité.
    """
    REAL = "real-data"
    SYNTHETIC = "synthetic-data"

    @classmethod
    def from_config(cls, raw_value: str) -> "DataMode":
        """
        Parse la string brute du fichier TOML en DataMode typé.

        Lève ValueError si la valeur n'est pas reconnue, ce qui est plus
        sûr que de laisser passer silencieusement.
        """
        for mode in cls:
            if mode.value == raw_value:
                return mode
        valid = ", ".join(repr(m.value) for m in cls)
        raise ValueError(
            f"Mode de données inconnu : {raw_value!r}. Valeurs valides : {valid}."
        )


class PipelineState(Enum):
    """
    États possibles d'un MaTirfPipeline durant son cycle de vie.

    Le cycle nominal est :
        IDLE -> LOADING -> COMPUTING -> COMPLETED

    Les états terminaux possibles sont :
        - COMPLETED   : succès, self.result est disponible
        - INTERRUPTED : l'utilisateur a cliqué Stop
        - FAILED      : une exception a été levée pendant setup() ou run()

    Convention :
        - verbes au gérondif/présent pour les états transitoires
          (LOADING, COMPUTING)
        - participe passé pour les états terminaux
          (COMPLETED, INTERRUPTED, FAILED)
    """
    IDLE = "idle"
    LOADING = "loading"
    COMPUTING = "computing"
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"
    FAILED = "failed"
