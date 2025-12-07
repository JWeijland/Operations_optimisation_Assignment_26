# Hoe werken de twee oplossingsmethoden?

## Het Probleem

We hebben vliegtuigen die moeten landen op landingsbanen. Elk vliegtuig:
- Heeft een **gewenste landingstijd** (target time)
- Mag **vroeg of laat** landen, maar dat kost geld
- Moet **voldoende afstand** houden van andere vliegtuigen (wake turbulence)

**Doel:** Vind een schema met de **laagste totale kosten**.

---

## Methode 1: Greedy Heuristiek (Snel, maar niet perfect)

### Simpel gezegd:
Plan de vliegtuigen één voor één in volgorde, en kies steeds de best mogelijke optie voor het huidige vliegtuig.

### Hoe het werkt (stap voor stap):

**Stap 1: Sorteer vliegtuigen**
```
Volgorde: Sorteer op gewenste landingstijd (vroegste eerst)
Voorbeeld: Vliegtuig A wil om 18:10, B om 18:15, C om 18:12
→ Volgorde wordt: A, C, B
```

**Stap 2: Plan elk vliegtuig één voor één**

Voor elk vliegtuig (in volgorde):

1. **Probeer elke landingsbaan**
   - Baan 1: Wat is de vroegst mogelijke tijd?
   - Baan 2: Wat is de vroegst mogelijke tijd?
   - etc.

2. **Bereken kosten per baan**
   - Als vliegtuig te vroeg landt → kosten = €100/minuut te vroeg
   - Als vliegtuig te laat landt → kosten = €200/minuut te laat
   - Als vliegtuig op tijd landt → kosten = €0

3. **Kies de baan met laagste kosten**

**Voorbeeld:**
```
Vliegtuig A (wil landen om 18:10):
- Baan 1: Kan om 18:10 → kosten €0 ✓ BESTE
- Baan 2: Kan om 18:10 → kosten €0
→ Kies baan 1

Vliegtuig B (wil landen om 18:12):
- Baan 1: Moet wachten tot 18:13 (A + separatie) → €200 te laat
- Baan 2: Kan om 18:12 → kosten €0 ✓ BESTE
→ Kies baan 2

Vliegtuig C (wil landen om 18:15):
- Baan 1: Kan om 18:15 → kosten €0 ✓ BESTE
- Baan 2: Moet wachten tot 18:16 → €200 te laat
→ Kies baan 1
```

**Resultaat:**
```
Baan 1: A (18:10), C (18:15)
Baan 2: B (18:12)
Totale kosten: €0
```

### Waarom heet het "Greedy"?

Omdat het **gulzig** is: het kijkt alleen naar wat NU het beste is, niet naar wat later gebeurt.

**Probleem:** Dit kan soms een niet-optimale keuze maken.

**Voorbeeld van fout:**
```
Vliegtuig A kiest baan 1 (€0 kosten nu)
Maar later blijkt dat dit vliegtuig D dwingt om veel te laat te landen (€500 kosten)
Als A baan 2 had gekozen (€50 kosten), was D op tijd geweest
→ Greedy koos niet de beste totale oplossing
```

### Voordelen:
- ✅ **Zeer snel** (0.01 seconden)
- ✅ Altijd een **haalbare** oplossing
- ✅ Werkt ook met 100+ vliegtuigen

### Nadelen:
- ❌ Niet gegarandeerd de **beste** oplossing (meestal 5-15% duurder dan optimaal)

---

## Methode 2: MILP Solver (Traag, maar optimaal)

### Simpel gezegd:
Probeer **alle mogelijke combinaties** systematisch uit en vind de allerbeste oplossing.

### Hoe het werkt (stap voor stap):

**Stap 1: Formuleer als wiskundig probleem**
```
Variabelen:
- x₁ = landingstijd vliegtuig 1
- x₂ = landingstijd vliegtuig 2
- δ₁₂ = 1 als vliegtuig 1 voor vliegtuig 2 landt, 0 anders
- etc.

Doel:
- Minimaliseer: totale kosten

Regels (constraints):
- Elk vliegtuig binnen tijdvenster
- Voldoende separatie tussen vliegtuigen
- Elk vliegtuig op precies 1 baan
```

**Stap 2: Los op met "Branch and Bound"**

Dit is een slimme manier om alle opties te proberen zonder ECHT alles te proberen.

**Voorbeeld:**
```
Start: Hoeveel mogelijke schema's zijn er?
- 40 vliegtuigen
- 4 banen
- Elk vliegtuig kan op verschillende tijden landen
→ Miljarden mogelijkheden!

Branch and Bound werkt zo:

1. Begin met een "relaxatie":
   "Stel dat vliegtuigen ook tussen banen kunnen landen"
   → Dit geeft een ONDERGRENS: €1250 (kan niet goedkoper)

2. Splits het probleem:
   "Vliegtuig 1 op baan 1" VS "Vliegtuig 1 op baan 2"

   Branch 1: V1 op baan 1
   → Beste oplossing: €1280

   Branch 2: V1 op baan 2
   → Beste oplossing: €1350

3. Splits verder (alleen branch 1, want die is goedkoper):
   "Vliegtuig 2 voor V1" VS "Vliegtuig 2 na V1"

   En zo verder...

4. Stop branches die NIET beter kunnen zijn:
   Als we al een oplossing hebben van €1300,
   en een branch heeft ondergrens €1350
   → Skip deze branch (kan toch niet beter)

5. Blijf splitsen tot je de beste oplossing hebt
```

**Visualisatie:**
```
                    Start
                   /    \
           V1-baan1      V1-baan2
           (€1280)       (€1350) ← Skip! Te duur
              |
           /     \
    V2-voor-V1  V2-na-V1
      (€1285)    (€1290)
         |
       ...
```

**Resultaat:**
```
Na 10.000 combinaties geprobeerd te hebben:
Beste schema gevonden met kosten €1285
```

### Extra trucjes die de solver gebruikt:

**1. Cutting Planes**
"Voeg slimme regels toe die domme oplossingen uitsluiten"
```
Bijvoorbeeld: "Als vliegtuig A te zwaar is voor baan 1, probeer het niet eens"
→ Minder combinaties om te proberen
```

**2. Heuristieken in de solver**
"Raad soms een goede oplossing"
```
Als we δ₁₂ = 0.7 hebben (fractie),
rond af naar 1 en kijk of dat een goede oplossing geeft
→ Sneller een bovengrens vinden
```

**3. Presolve**
"Vereenvoudig het probleem voordat je begint"
```
Als vliegtuig A ALTIJD na 18:00 moet landen,
verwijder alle opties vóór 18:00
→ Kleiner probleem
```

### Voordelen:
- ✅ **Gegarandeerd beste oplossing** (binnen time limit)
- ✅ Je weet hoe goed het is (optimality gap)

### Nadelen:
- ❌ **Langzaam** (10-50 seconden voor 40 vliegtuigen)
- ❌ Schaalt slecht (50+ vliegtuigen wordt te traag)
- ❌ Kan time-out hebben zonder oplossing

---

## Vergelijking

| Aspect | Greedy Heuristiek | MILP Solver |
|--------|-------------------|-------------|
| **Snelheid** | 0.01 seconden | 10-50 seconden |
| **Kwaliteit** | 5-15% slechter dan optimaal | Optimaal |
| **Garantie** | Altijd haalbare oplossing | Beste oplossing (als tijd genoeg) |
| **Max. vliegtuigen** | 100+ | ~50 |
| **Wanneer gebruiken?** | Real-time, quick preview | Offline planning, finale schema |

---

## Hoe worden ze gebruikt in je code?

### Scenario oplossen met beide methoden:

```python
# 1. Maak scenario (30 vliegtuigen, 2 banen)
instance = create_schiphol_evening_rush(
    num_aircraft=30,
    num_runways=2
)

# 2. Maak solver
solver = OptimalSolver(instance)

# 3. Los op met BEIDE methoden en vergelijk
result = solver.solve_and_compare(
    num_runways=2,
    time_limit=50  # 50 seconden voor MILP
)

# Resultaat:
# - Heuristic cost: €4250 (oplossing in 0.02s)
# - Optimal cost: €3850 (oplossing in 45s)
# - Gap: 10.4% (heuristiek is 10% duurder)
# - Speedup: 2250x (heuristiek is 2250x sneller)
```

### In de sensitivity analysis:

Voor elk scenario (bijv. 30 vliegtuigen, 2 banen):
1. **Greedy** runt eerste → geeft snel een goede oplossing
2. **MILP** runt daarna → geeft de beste oplossing
3. Vergelijk ze → bereken gap en speedup
4. Sla beide op in CSV

Zo kun je zien:
- Hoe goed is de heuristiek? (gap)
- Is het de tijd waard om MILP te draaien? (speedup vs gap)

---

## Simpele Analogie

**Greedy Heuristiek** = Route plannen met "neem steeds de eerstvolgende afslag richting bestemming"
- Snel beslissen
- Werkt meestal goed
- Maar soms mis je een betere route

**MILP Solver** = Route plannen met Google Maps (alle routes vergelijken)
- Vindt de echt beste route
- Duurt langer om te berekenen
- Maar garantie op beste resultaat

Voor dagelijkse ritten → Greedy is prima
Voor belangrijke planning → MILP is beter
