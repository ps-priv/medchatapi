"""Stale konfiguracyjne silnika rozmowy medycznej."""

BRIBERY_KEYWORDS = (
    "łapówk",
    "lapowk",
    "kopert",
    "prowizj",
    "korzyść osobist",
    "korzysc osobist",
    "dogadamy się",
    "dogadamy sie",
    "dla pana doktora",
    "dla pani doktor",
    "za przepisywanie",
    "za recept",
    "gratyfikacj",
)

OFF_TOPIC_KEYWORDS = (
    "polityk",
    "wybor",
    "pogod",
    "sport",
    "mecz",
    "urlop",
    "samochod",
    "plotk",
    "serial",
    "film",
    "bitcoin",
    "krypto",
)

CLINICAL_STUDY_PHRASES = (
    "badanie kliniczne",
    "badania kliniczne",
    "badanie na populacji",
    "badania na populacji",
    "populacja pacjentów",
    "populacja pacjentow",
    "test kliniczny",
    "testy kliniczne",
    "próba kliniczna",
    "proba kliniczna",
    "randomizowane badanie",
    "badanie randomizowane",
    "podwójnie ślepa próba",
    "podwojnie slepa proba",
    "ślepa próba",
    "slepa proba",
    "placebo",
    "grupa kontrolna",
    "wyniki badania",
    "wyniki kliniczne",
    "skuteczność potwierdzona w badaniu",
    "skutecznosc potwierdzona w badaniu",
    "redukcja o",
    "poprawa o",
    "w grupie badanej",
    "pacjenci w badaniu",
    "okres obserwacji",
    "follow-up",
    "pierwszorzędowy punkt końcowy",
    "pierwszorzedowy punkt koncowy",
    "drugorzędowy punkt końcowy",
    "drugorzedowy punkt koncowy",
    "istotność statystyczna",
    "istotnosc statystyczna",
    "przedział ufności",
    "przedzial ufnosci",
    "hazard ratio",
    "odds ratio",
    "number needed to treat",
    "nnt",
)

EVIDENCE_PHRASES = (
    "konkretnie",
    "dokładnie",
    "dokladnie",
    "na podstawie badań",
    "na podstawie badan",
    "badania kliniczne",
    "badania pokazują",
    "badania pokazuja",
    "w badaniu",
    "dane kliniczne",
    "wyniki badań",
    "wyniki badan",
    "w badaniach",
    "skuteczność potwierdzona",
    "skutecznosc potwierdzona",
    "udowodniono",
    "zgodnie z danymi",
    "według badań",
    "wedlug badan",
    "populacja badana",
    "punkt końcowy",
    "punkt koncowy",
    "statystycznie istotny",
    "p <",
    "p<",
)

EMPTY_PRAISE_PHRASES = (
    "dobry lek",
    "świetny lek",
    "swietny lek",
    "znakomity lek",
    "doskonały lek",
    "doskonaly lek",
    "wspaniały lek",
    "wspanialy lek",
    "super lek",
    "bardzo dobry lek",
    "bardzo skuteczny lek",
    "rewelacyjny lek",
    "idealny lek",
    "polecam ten lek",
    "to dobry preparat",
    "świetny preparat",
    "swietny preparat",
    "doskonały preparat",
    "doskonaly preparat",
)

MARKETING_BUZZWORDS = (
    "rewolucyjny",
    "przełomowy",
    "bezkonkurencyjny",
    "gwarantowany efekt",
    "numer jeden",
    "hit sprzedaży",
    "mega skuteczny",
    "najlepszy na rynku",
)

ENGLISH_MARKETING_WORDS = (
    "game changer",
    "best practice",
    "win-win",
    "target",
    "pipeline",
    "roi",
    "insight",
    "follow-up",
    "brief",
    "performance",
    "compliance",
    "storytelling",
)

INAPPROPRIATE_PROPOSAL_KEYWORDS = (
    "kolacja we dwoje",
    "spotkamy się prywatnie",
    "spotkamy sie prywatnie",
    "prywatne spotkanie",
    "numer prywatny",
    "jest pani piękna",
    "jest pani piekna",
    "kochana",
    "kochanie",
    "skarbie",
    "romantycznie",
    "flirt",
)

DISRESPECTFUL_LANGUAGE_KEYWORDS = (
    "głupia",
    "glupia",
    "nie zna się pani",
    "nie zna sie pani",
    "bzdury pani mówi",
    "bzdury pani mowi",
    "zamknij się",
    "zamknij sie",
    "idiotka",
)

MALE_ADDRESS_FORMS = ("pan doktor", "panie doktorze", "panie doktor", "doktorze")
FEMALE_ADDRESS_FORMS = ("pani doktor", "pani doktorko")

POLISH_STOPWORDS = {
    "oraz",
    "który",
    "ktory",
    "która",
    "ktora",
    "które",
    "ktore",
    "przez",
    "przed",
    "możliwe",
    "mozliwe",
    "lekarz",
    "pacjent",
    "pacjentów",
    "pacjentow",
    "dawka",
    "dawkowaniu",
    "może",
    "moze",
    "dziennie",
    "zgodnie",
    "zaleceniem",
}

TRAIT_KEYS = ("skepticism", "patience", "openness", "ego", "time_pressure")
PHASES = ("opening", "needs", "objection", "evidence", "close")

PHASE_OBJECTIVES = {
    "opening": "Krótko ustaw kontekst i poproś o konkretną wartość kliniczną leku.",
    "needs": "Doprecyzuj profil pacjenta i praktyczne zastosowanie leku.",
    "objection": "Wyraź kluczową obiekcję i wymagaj merytorycznej odpowiedzi.",
    "evidence": "Wymagaj danych: bezpieczeństwo, przeciwwskazania, dawkowanie.",
    "close": "Domknij rozmowę i zakończ spotkanie bez nowych wątków.",
}

SUPPORTED_STRATEGIES = {
    "transactional",
    "skeptical",
    "confrontational",
    "exploratory",
    "disengaging",
}

CLAIM_SEVERITY_WEIGHTS = {"critical": 1.6, "major": 1.0, "minor": 0.5}

# Komendy czatu: dokładna treść wiadomości (strip, case-sensitive) → nazwa akcji.
# Aby dodać nową komendę, wystarczy dopisać wpis tutaj.
CHAT_COMMANDS: dict = {
    "Zamienić": "close_visit",
    "Skuteczny":"increase_openness"
}
