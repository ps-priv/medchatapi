# Symulator Rozmowy Medycznej — Przewodnik dla Handlowca

## Czym jest ten produkt?

To API symulujące rozmowę między **przedstawicielem farmaceutycznym** (użytkownik) a **lekarzem** (AI). Produkt służy do **treningu handlowców** przed prawdziwymi wizytami u lekarzy — w bezpiecznym środowisku, bez ryzyka popełnienia błędu wobec realnego klienta.

Lekarz to agent AI z prawdziwymi cechami psychologicznymi, który reaguje na jakość argumentacji, błędy merytoryczne, próby manipulacji i styl komunikacji. Rozmowa jest oceniana po każdej turze i na końcu wizyty.

---

## Kluczowe wartości dla klienta

- **Trening bez ryzyka** — handlowiec może popełniać błędy i uczyć się na nich bez konsekwencji biznesowych
- **Realistyczny lekarz** — AI zmienia nastrój, poziom sceptycyzmu i cierpliwości w zależności od tego, co mówi rozmówca
- **Natychmiastowy feedback** — po każdej wypowiedzi system ocenia: trafność tematu, precyzję kliniczną, etykę, jakość języka
- **10 różnych typów lekarzy** — od przyjaznego ogólnego po zmęczonego cynika, każdy wymaga innej taktyki
- **Głos lekarza** — odpowiedzi mogą być syntezowane do audio (TTS), co zwiększa realizm

---

## Jak działa rozmowa — krok po kroku

### 1. Start wizyty (`/start`)
Klient (np. aplikacja szkoleniowa) wybiera:
- **ID lekarza** — np. `skeptical_expert`, `friendly_generalist`
- **ID leku** — np. `metformax_sr`

System tworzy sesję z profilem lekarza i katalogiem klinicznych twierdzeń dla wybranego leku.

### 2. Wymiana wiadomości (`/message`)
Handlowiec wysyła wiadomość tekstową. System:
1. Analizuje wypowiedź pod kątem twierdzeń o leku — weryfikuje je z katalogiem (poprawne / niepotwierdzone / fałszywe)
2. Wykrywa sygnały jakościowe: ogólniki, język marketingowy, anglicyzmy, dane kliniczne, badania
3. Aktualizuje cechy psychologiczne lekarza i poziom frustracji
4. Generuje odpowiedź lekarza przez GPT (model: `gpt-5.4`)
5. Zwraca metryki tury i aktualny stan rozmowy

### 3. Zakończenie wizyty (`/finish`)
System generuje **raport końcowy**: ocena całej rozmowy, decyzja lekarza (przepiszę / odrzucam / do przemyślenia), podsumowanie błędów i mocnych stron.

---

## Psychologia lekarza — jak działa AI

Każdy lekarz ma **5 cech psychologicznych** (wartości 0.0–1.0):

| Cecha | Co oznacza |
|---|---|
| `skepticism` | Jak bardzo lekarz kwestionuje twierdzenia |
| `patience` | Jak długo toleruje słabą argumentację |
| `openness` | Gotowość do zmiany zdania |
| `ego` | Oczekiwanie szacunku i profesjonalizmu |
| `time_pressure` | Jak bardzo się spieszy |

Cechy **zmieniają się w trakcie rozmowy** w zależności od zachowania handlowca:

| Zachowanie handlowca | Efekt |
|---|---|
| Powołuje się na badania kliniczne, populację pacjentów | ↓ presja czasu, ↑ otwartość, ↓ sceptycyzm |
| Mówi "to dobry lek", "świetny preparat" bez danych | ↑ sceptycyzm |
| Używa danych, mówi konkretnie, precyzyjnie | ↓ presja czasu |
| Używa języka marketingowego, anglicyzmów | ↑ sceptycyzm, ↓ otwartość |
| Podaje fałszywe dane o leku | ↑↑ sceptycyzm, ↓ cierpliwość |
| Propozycja korupcyjna (łapówka) | Natychmiastowe zakończenie wizyty |

---

## 10 typów lekarzy

| ID | Nazwa | Charakter | Trudność |
|---|---|---|---|
| `friendly_generalist` | Przyjazny ogólny | Otwarty, ciepły lekarz rodzinny | Łatwy |
| `busy_pragmatist` | Zabiegany pragmatyk | Mało czasu, chce konkretów | Łatwy |
| `young_enthusiast` | Młody entuzjasta | Ciekawy nowości, otwarty | Średni |
| `curious_specialist` | Ciekawy specjalista | Dociekliwy, zadaje wiele pytań | Średni |
| `empathetic_clinician` | Empatyczny klinicysta | Spokojny, ceni relacje | Średni |
| `skeptical_expert` | Sceptyczny ekspert | Wymaga dowodów, krytyczny | Średni |
| `academic_researcher` | Naukowiec akademicki | Skupiony na danych i metodologii | Trudny |
| `dominant_authority` | Dominujący autorytet | Silne ego, oczekuje szacunku | Trudny |
| `defensive_traditionalist` | Defensywny tradycjonalista | Przywiązany do starych metod | Trudny |
| `tired_cynic` | Zmęczony cynik | Wypalony, sceptyczny, mało czasu | Trudny |

---

## Fazy rozmowy

Każda wizyta przechodzi przez 5 faz:

1. **Opening** — lekarz nie wie po co przyszedł rozmówca, oczekuje przedstawienia celu
2. **Needs** — ustalanie kontekstu klinicznego i wskazań
3. **Objection** — lekarz wyraża obiekcje, testuje argumentację
4. **Evidence** — wymaga danych: bezpieczeństwo, dawkowanie, przeciwwskazania
5. **Close** — domknięcie rozmowy i decyzja lekarza

---

## Ocena po każdej turze

Każda odpowiedź handlowca jest punktowana (0–1):

| Metryka | Co mierzy |
|---|---|
| `topic_adherence` | Czy rozmowa dotyczy leku |
| `clinical_precision` | Czy dane są poprawne klinicznie |
| `ethics` | Brak prób korupcji i nieprofesjonalnych zachowań |
| `language_quality` | Brak anglicyzmów i sloganów marketingowych |
| `critical_claim_coverage` | Ile kluczowych tematów (skuteczność, bezpieczeństwo, dawkowanie) zostało omówionych |

---

## Decyzja lekarza na koniec wizyty

| Wartość | Znaczenie |
|---|---|
| `will_prescribe` | Lekarz zdecydował się przepisać lek |
| `trial_use` | Lekarz chce spróbować z jednym pacjentem |
| `recommend` | Lekarz będzie polecał innym |
| `undecided` | Lekarz nie podjął decyzji |
| `reject` | Lekarz odrzuca lek |

---

## Możliwości integracji technicznej

| Endpoint | Funkcja |
|---|---|
| `POST /start` | Rozpoczęcie wizyty (wybór lekarza i leku) |
| `POST /message` | Wysłanie wiadomości, odbiór odpowiedzi lekarza |
| `POST /finish` | Zakończenie i raport końcowy |
| `POST /transcribe` | Zamiana nagrania audio (base64) na tekst |
| `POST /transcribe-file` | Zamiana pliku audio (multipart) na tekst |
| `POST /tts` | Zamiana tekstu odpowiedzi lekarza na audio MP3 |

**Stack techniczny:** Python, FastAPI, OpenAI GPT (generowanie odpowiedzi), OpenAI TTS (głos lekarza), OpenAI Whisper (transkrypcja mowy). Deployment przez Docker.

**Wymagania:** klucz API OpenAI, serwer z Dockerem lub środowisko Python 3.10+.

**Ograniczenia obecnej wersji:**
- Sesje są przechowywane w pamięci serwera — restart kasuje aktywne rozmowy
- Jeden serwer obsługuje wszystkich użytkowników (brak skalowania poziomego)
- Brak persystencji historii rozmów (baza danych nie jest podpięta)

---

## Jak zaprezentować klientowi — scenariusz demo

1. **Wybierz najtrudniejszego lekarza** (`tired_cynic` lub `dominant_authority`) i pokaż jak reaguje na słabą argumentację
2. **Pokaż kontrast** — ta sama rozmowa z `friendly_generalist`, żeby pokazać jak różne typy wymagają różnych taktyk
3. **Celowo popełnij błąd** — podaj fałszywą informację o leku i pokaż jak lekarz to wykrywa i reaguje
4. **Pokaż raport końcowy** — metryki i decyzja lekarza jako konkretny feedback szkoleniowy
5. **Włącz głos** (TTS) — odpowiedzi lekarza jako audio zwiększają realizm i zaangażowanie

---

## Dodawanie nowych leków i lekarzy

- **Nowy lek** — dodaj wpis do `drugs.json` z listą klinicznych twierdzeń (`claims`), ich wagą (`critical`/`major`/`minor`) i wzorcami poprawnych/fałszywych sformułowań
- **Nowy lekarz** — dodaj profil do `doctor_archetypes.json` z cechami psychologicznymi, stylem komunikacji, głosem TTS i preferowanymi strategiami
