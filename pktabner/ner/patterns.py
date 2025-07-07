import re

# -----------------------------------------------------------------------------
# Precompiled basic patterns
# -----------------------------------------------------------------------------

COMMON_STOPWORD_PATTERNS = (
    r"\bof\b"
    r"|\bat\b"
    r"|\bby\b"
    r"|\bon\b"
    r"|\bthe\b"
    r"|\bfrom\b"
    r"|\band\b"
    r"|\bis\b"
    r"|\bin\b"
    r"|\ban\b"
    r"|\bwith\b"
    r"|\bby\b"
    r"|\bfor\b"
    r"|\bit\b"
    r"|\bas\b"
    r"|\bbetween\b"
    r"|\bbased\b"
)

DASH_VARIANTS = [
    "\u2010",  # Hyphen (‐)
    "\u2011",  # Non-breaking hyphen (‑)
    "\u2012",  # Figure dash (‒)
    "\u2013",  # En dash (–)
    "\u2014",  # Em dash (—)
    "\u2015",  # Horizontal bar (―)
    "\u2212",  # Minus sign (−)
    "\uFE58",  # Small em dash (﹘)
    "\uFE63",  # Small hyphen-minus (﹣)
    "\uFF0D"  # Full-width hyphen-minus (－)
]
DASH_PATTERN = "[" + "".join(DASH_VARIANTS) + "]"

# Text preprocessing tokenizers
STOP_WORDS_RE = re.compile(COMMON_STOPWORD_PATTERNS, re.IGNORECASE)
HTML_TAG_RE = re.compile(r"<[^>]+>")
PLURAL_RE = re.compile(r"(?<!s)(?<!mea)(?<!michaeli)s\b")
BIO_PLURAL_RE = re.compile(r"bioavailabilities|bioavailabilitie")
HL_PLURAL_RE = re.compile(r"half[\s\-_]*(lives|live|times)")
DASH_RE = re.compile(DASH_PATTERN)
MULTIPLE_DASH_RE = re.compile(r"--+")
FRACTION_SLASH_RE = re.compile(r"⁄", re.IGNORECASE)

# -----------------------------------------------------------------------------
# Precompiled units patterns
# -----------------------------------------------------------------------------

UNIT_SYNONYMS = {
"·": [" x " , "*" , "•" , "."] ,
"µg ": [" micrograms " , " micro g " , " microg " , " microgram " , "µg " , " mug "] ,
" h ": [" hr " , " hrs " , " hour " , " hours "] ,
"%": [" percent " , " percentage "] ,
"µl ": [" microliters " , " microlitre ", " microliter " , " micro l " , " microl " , "µl "] ,
" l ": [" liters " , " litre " , " liter " , " litres "] ,
" dl ": [" deciliter " , " dliter "] ,
" min ": [" minutes " , " minute " , " mins "] ,
" d ": [" days " , " day "] ,
" month ": [" months "] ,
" kg ": [" kilogram " , " kilograms "] ,
" s ": [" sec "] ,
" ms ": [" milisec " , " miliseconds " , " msec "] ,
" nM ": [" nmol " , " nanomol "] ,
" mM ": [" mmol " , " milimol "] ,
"µM ": [" mumol " , " micromol " , " micromols " , " micro mol ", " micro-mol ", " mumol " , "µmol " , "µmol " , "µM "] ,
" pM ": [" pmol " , " pmols " , " picomol "],
"iu": ["iu", "IU", "international units"],
}

UNIT_SPAN_PATTERN = (
    r"\b(?:"
    r"(?:\d*\s*(?:per\s*)?)?"  # optional leading number/per
    r"(?:pg|ng|µg|ug|mg|g|kg|"  # mass
    r"pmol|nmol|µmol|mmol|mol|"  # molar
    r"nl|µl|ul|ml|l|dl|m2|m²|"   # volume
    r"s|ms|min|h|hr|hrs|d|day|"
    r"iu)"
    r"(?:[·*/\-\s]*"
    r"(?:pg|ng|µg|ug|mg|g|kg|pmol|nmol|µmol|mmol|mol|nl|µl|ul|ml|l|dl|m2|m²|s|ms|min|h|hr|hrs|d|day|iu)"
    r"(?:[-−]?[1-9])?"
    r"){0,3}"
    r")\b"
)

def normalise_unit_synonyms(text: str, unit_map: dict) -> str:
    for canonical, synonyms in unit_map.items():
        for syn in synonyms:
            pattern = re.escape(syn.strip())
            text = re.sub(pattern, canonical.strip(), text, flags=re.IGNORECASE)
    return text


def contains_unit_span(text: str, unit_map: dict, unit_pattern: str) -> re.Match | None:
    normalised_text = normalise_unit_synonyms(text, unit_map)
    return re.search(unit_pattern, normalised_text, re.IGNORECASE)

UNIT_MAGNITUDES = {
" TIME ": [" ms " , " s " , " min " , " h " , " d " , " month "] ,
" MASS ": [" ng " , "µg " , " mg " , " g " , " kg " , " pg "] ,
" VOLUME ": [" nl " , "µl " , " ml " , " l " , " dl "] ,
" CONCENTRATION ": [" pM " , " nM " , "µM " , " mM " , " M "] ,
" PERCENTAGE ": ["%"] ,
}

# -----------------------------------------------------------------------------
# Precompiled context patterns
# -----------------------------------------------------------------------------

# todo: recognise [2,688‐70,300] as numeric cell (range in sq. braclets),.

# nb can be rows or columns ...
HEADER_PATTERNS = {
    # n sub subject header
    "n_sub_header": (
            r"(?:"
            r"(n°|number|no\.|n)(\s*of)?\s*(?:subjects?|participants?|patients?|animals?)(?:\s*enrolled)?"
            r"|(?:subjects?|participants?|patients?|animals?)\s*(n°|number|no\.|n)"
            r"|(?:subjects?|participants?|patients?)\s*enrolled"
            r"|study\s*size"
            r"|^\(?\s*n\s*\)?$"
            r"|\bno\." #N.B must not match "No. 6" etc. in a cell 
            r")"
        ), # add: subjects, n

    # age subject header
    "age_header": (
    r"\b(?:ages|age)\b(?!\s*\d+)"
    ),

    "weight_header" : (
        r"\b(body[-\s]*(weight|wt)|weights?|bwt)\b" # WT is wildtype (mice) - case sensitive or remove overlap?
    ),

    "dose_header": (
    r"\b(doses?|dosage)\b" # need to match the full text of a header ... #Dose (mg/kg)	-> maybe be celver and look for mention of dose without any numeric values?
    ), # todo: Dose Level (mg/kg/week)

    # maybe a "unit/unit" col header?
}

N_SUB_PATTERNS = {
    "n_sub_inline_patterns": (
        r"\bn\s*=\s*\d+\b"
    )
}

SAMPLE_TYPE_PATTERNS = {
    "whole_blood": r"\bwhole\s*blood\b|\bblood\b(?!\s*(pressure|sugar|test|flow|volume))",
    "plasma": r"\bplasma\b",
    "serum": r"\bserum\b",
    "csf": r"\bcerebrospinal\s*fluid\b|\bcsf\b",
    "urine": r"\burinary\b|\burine\b",
    "feces": r"\bfeces|faeces|feacal|fecel\b",  # uk and usa spelling
    "saliva": r"\bsalivary\b|\bsaliva\b",
    "sweat": r"\bsweat\b",
    "bile": r"\bbile\b",
    "breast_milk": r"\bbreast\s*milk\b",
    "sputum": r"\bsputum\b",
    "cornea": r"\bcornea\b",
    "brain": r"\bbrain\b",
    "spleen": r"\bspleen\b",
    "liver": r"\bliver\b",
    "lung": r"\blung\b|\blung\s*tissue\b",
    "umbilical_vein": r"\bumbilical\s*vein\b",
    "synovial_fluid": r"\bsynovial\s*fluid\b|\bsynovial\s*lavage\s*fluid\b|\bsynovium\b",
    "eyelid": r"\beyelid\b",
    "interstitial_fluid": r"\binterstitial\s*fluid\b|\bisf\b",
    "conjunctiva": r"\bconjunctiva\b",
    "foetal_plasma": r"\bfoetal\s*plasma\b",
    "kidney": r"\bkidneys?\b",
    "heart": r"\bheart\b",
    "perilymph": r"\bperilymph\b",
    "lymphatic_fluid": r"\blymphatic\s*fluid\b",
    "ileum": r"\bileum\b",
    "spleen": r"\bspleen\b",
    "thymus": r"\bthymus\b",
    "thyroid": r"\bthyroid\b",
    "ovaries": r"\bovaries\b",
    "bone": r"\bbone\b",
    "adipose": r"\badipose\b",
}

AGE_PATTERNS = {
    # Neonates: Birth to 27 days
    "neonates": (
        r"\bnewborns?\b"
        r"|\bneonates?\b"
        r"|\b(?:[1-4])\s*[-–]?\s*week[-\s]*old\b"
        r"|\bage\s*?(≤|<)\s*1\s*(month|mo)\b"
        r"|(<|≤)\s*1\s*(month|mo)(\s*old)?\b"
    ),

    # Infants: 28 days to 23 months
    "infants": (
        r"\binfants?\b"
        r"|\b(?:[1-9]|1[0-9]|2[0-3])\s*[-]?\s*months?[-\s]*old\b"
        r"|\bage\s*(<|≤|up to|less than)\s*2\s*(years?|yrs?)\b"
        r"|\bage\s*0\s*(to|–|-)\s*2\s*(years?|yrs?)\b"
        r"|\b0\s*(to|–|-)\s*2\s*(years?|yrs?)\b"  # <-- NEW
        r"|\b\d+\s*(mo|months?)\s*(to|–|-)\s*2\s*(years?|yrs?)\b"
        r"|\bsubjects?\s*<\s*2\s*(years?|yrs?)\b"
        r"|\bage\s*≤?\s*24\s*(months?|mo)\b"
        r"|\b\d+\s*(months?|mo)\s*old\b(?=.*\b<\s*2\s*(years?|yrs?)\b)?"
    ),

    # Children: 2 to 11 years
    "children": (
    r"\bchildren\b"
    r"|\bolder\s*children\b"
    r"|\bage\s*(>|≥)?\s*2\s*(years?|yrs?)\s*(to|–|—|-)\s*(<|≤)?\s*1[0-2]\s*(years?|yrs?)\b"
    r"|\bsubjects?\s*<\s*12\s*(years?|yrs?)\b"
    r"|\b≥\s*2\s*(years?|yrs?)\s*to\s*<\s*12\s*(years?|yrs?)\b"
    r"|\b(?:[2-9]|1[0-1])\s*(yrs?|years?)\s*old\b"
),

    # Adolescents: 12 to 17 years
    "adolescents": (
    r"adolescents?"
    r"|(?:>|≥)?\s*1[2-7]\s*(years?|yrs?)\s*(to|–|—|-)\s*(<|≤)?\s*1[3-8]\s*(years?|yrs?)"
    r"|\b≥\s*1[2-7]\s*(years?|yrs?)\s*to\s*<\s*18\s*(years?|yrs?)"
    r"|\b1[2-7]\s*(to|–|—|-)\s*1[3-8]\s*(years?|yrs?)"
    r"|\bage\s*1[2-7]\s*(years?|yrs?)"
)
,

    # Pediatric (0–17): includes all above
    "children": (
    r"\bchildren\b"
    r"|\bolder\s*children\b"
    r"|age\s*(>|≥)?\s*2\s*(years?|yrs?)\s*(to|–|—|-)\s*(<|≤)?\s*1[0-2]\s*(years?|yrs?)\b"
    r"|(?:>|≥)\s*2\s*(years?|yrs?)\s*(to|–|—|-)\s*(<|≤)?\s*1[0-2]\s*(years?|yrs?)\b"
    r"|\bsubjects?\s*<\s*12\s*(years?|yrs?)\b"
    r"|≥\s*2\s*(years?|yrs?)\s*to\s*<\s*12\s*(years?|yrs?)\b"
    r"|\b(?:[2-9]|1[0-1])\s*(yrs?|years?)\s*old\b"
)
,

    # Adults: 18 to 64 years
    "adults": (
        r"\badults?\b"
        r"|\bage\s*(>|≥)\s*18\s*(years?|yrs?)\s*(to|-)\s*(<|≤)?\s*65\s*(years?|yrs?)\b"
        r"|\baged\s*18\s*to\s*64\b"
    ),

    # Elderly: 65+
    "elderly_patients": (
    r"\belderly\b"
    r"|\bseniors?\b"
    r"|\bolder\s*adults?\b"
    r"|\bgeriatric\s*(populations?|patients?)\b"
    r"|\bage\s*(>|≥)\s*65\s*(years?|yrs?)\b"
    r"|\baged\s*65\s*(years?|yrs?)?\s*(and\s*older|and\s*above|plus)\b"
)
,

}

SPECIES_PATTERNS = {

    "animals": (
        r"\b(animals?|"
        r"rats?|mouse|mice|rodents?|rabbits?|hamsters?|"
        r"dogs?|canines?|cats?|felines?|"
        r"goats?|sheep|cows?|llamas?|alpacas?|"
        r"monkeys?|macaques?|chimpanzees?|primates?)\b"
    ),

    "humans": (
        r"\b(subjects?|patients?|participants?|humans?|volunteers?)\b"
    ),
}


ROUTE_PATTERNS = {

    "intravenous": (
        r"\bintra[\s\-_]*venous\b"
        r"|\biv[\s\-_]*infusion\b"
        r"|i\.v\.[\s\-_]*infusion\b"
        r"|\binfusion\b"
        r"|\bbolus\b"
        r"|i\.v\."
        r"|\biv\b"

    ), # Todo: must not recongise : Study IV

    "intramuscular": (
        r"\bintra[\s\-_]*muscular\b"
        r"|\bim\b"
        r"|i\.m\."
    ),

    "subcutaneous": (
        r"\bsub[\s\-_]*cutaneous\b"
        r"|\bsub[\s\-_]*cut\b"
        r"|\bsubq\b"
        r"|\bsc\b"
        r"|\bsq\b"
        r"|s\.c\."
    ),

    "oral": (
        r"\borally\b"
        r"|\boral\b"
        r"|\bpo\b"
        r"|p\.o\."
        r"|\bper\s*os\b"
    ),

    "sublingual": (
        r"\bsub[\s\-_]*lingually\b"
        r"|\bsub[\s\-_]*lingual\b"
        r"|\bsl\b"
        r"|s\.l\."
    ),

    "buccal": (
        r"\bbuccal\b"
        r"|\bbuc\b"
        r"|b\.u\.c\."
    ),

    "inhaled": (
        r"\binhalation\b"
        r"|\binhaled\b"
        r"|\bpulmonary\b"
        r"|\bvia[\s\-_]*inhalation\b"
    ),

    "intranasal": (
        r"\bintra[\s\-_]*nasal\b"
        r"|\bintra[\s\-_]*nasally\b"
        r"|\bnasal[\s\-_]*spray\b"
        r"|\bnasally\b"
        r"|i\.n\."
    ),

    "topical": (
        r"\btopical\b"
        r"|\bcutaneous\b"
        r"|\bapplied[\s\-_]*(to|on)[\s\-_]*skin\b"
    ),

    "transdermal": (
        r"\btransdermal\b"
        r"|\bpatch\b"
        r"|\btd\b"
        r"|t\.d\."
    ),

    "intradermal": (
        r"\bintrar[\s\-_]*dermal\b"
        r"|\bintradermal\b"
        r"|\bintrar[\s\-_]*dermally\b"
        r"|\bid\b"
        r"|i\.d\."
    ),

    "rectal": (
        r"\brectal\b"
        r"|\bsuppositor(y|ies)\b"
        r"|\bpr\b"
        r"|p\.r\."
        r"|\bper[\s\-_]*rectum\b"
    ),

    "vaginal": (
        r"\bvaginal\b"
        r"|\bvaginally\b"
        r"|\bpv\b"
        r"|p\.v\."
        r"|\bper[\s\-_]*vaginam\b"
    ),

    "intrauterine": (
        r"\bintra[\s\-_]*uterine\b"
        r"|\biu\b"
        r"|i\.u\."
    ),

    "intrathecal": (
        r"\bintra[\s\-_]*thecal\b"
        r"|\bit\b"
        r"|i\.t\."
    ),

    "epidural": (
        r"\bepidural\b"
        r"|\bepi\b"
        r"|e\.p\.i\."
    )
}

DOSE_PATTERN = (
    r"\b\d+(\.\d+)?\s*"                             # Number (int or decimal)
    r"(mg|g|µg|mcg|ug|ng)\b"                         # Mass unit
    r"(\s*(/|per|\s+per\s+)\s*"                      # Optional normalizer intro
    r"(kg|kg[-\s]*1|kg⁻¹|m2|m²|m[-\s]*2|day|d|hr|h|hours?|min|minute|seconds?|s)"
    r"(\s*/\s*(kg|m2|m²|day|d|hr|h|hours?|min|minute|s|seconds?))?"  # Optional second normalizer
    r")?"
)

WEIGHT_PATTERNS = {
    "weight_value": r"\b\d+(\.\d+)?[-\s]*(kilograms?|kg)\b",
    "weight_range": r"\b\d+(\.\d+)?[-\s]*[-–][-\s]*\d+(\.\d+)?[-\s]*(kilograms?|kg)\b", # TO DO ADD mean +/- s.d. kg etc.
}

MEASURE_PATTERNS = {
    # Ratios
    "ratios": (
        r"\b(geometric[-\s]*mean[-\s]*ratio|gmr|"
        r"ratio\s*of[-\s]*geometric[-\s]*means?|"
        r"ratios?)\b"
    ), # also parameters with : or /

    # Central Tendency
    "geometric_mean": r"\b(geometric[-\s]*means?|geom[-\s]*means?|(geom|geo)\.\s*mean|g\s*means?|gmean|gm)\b",

    "mean": r"\b(arithmetic[-\s]*means?|mean[-\s]*values?|means?|averages?)\b", # todo: maybe build in to loom for full sopan mean =- sd etc.

    "typical_value": r"\btypical[-\s]*value\b|\btv\b|t\.v\.|θ|\bestimate\b",

    "median": r"\bmedian\b",

    # Variability
    "range": r"\brange\b|min\,\s*max|min–max",
    "interquartile_range": (
        r"\b(interquartile[-\s]*range|"
        r"25(th)?[-–][-\s]*75(th)?[-\s]*percentiles?"
        r"|iqr)\b"
    ),
    "confidence_interval": (
        r"(\d{1,3}(\.\d+)?\s*%[\s\-_]*ci|"
        r"ci[\s\-_]*interval[-\s]*range|"
        r"confidence[\s\-_]*intervals?|"
        r"\bci\b)"
    ),
    "standard_deviation": r"\b(standard[-\s]*deviation|sd|s\.d\.)\b",
    "standard_error": r"\b(standard[-\s]*error|s\.e\.m|se|s\.e\.)\b",
    "coefficient_of_variation": r"\b(cv%|cv\s*%|cv|geocv)\b",
    "variance": r"\bvar(iance)?\b",
}


PK_VARS = {
    # Inter-individual Variability (IIV)
    "iiv": (
        r"\b(inter[-\s]*(subject|individual)[-\s]*(variance|variability|cv)(\s*ω2)?|"
        r"between[-\s]*(subject|individual)[-\s]*(variance|variability|cv\s*%|cv)|"
        r"inter[-\s]*(subject|individual)"
        r"iiv|bsv|ω2)\b" # ωcl2
    ),

    # Intra-individual Variability (WSV)
    "wsv": (
        r"\b(intra[-\s]*(subject|individual)[-\s]*(variance|variability|cv)|"
        r"within[-\s]*(subject|individual)[-\s]*(variance|variability|cv)|"
        r"residual[-\s]*(variance|variability|cv)|"
        r"intra[-\s]*(subject|individual)"
        r"intra[-\s]*iv|"
        r"wsv|ε|ω)\b"
    ),
}

DRUG_PATTERNS = {
    "placebo":  r"\bplacebo\b"
}

SAMPLE_TIME_PATTERNS = {
    "time": r"\bday\s*\d+\b"
}



###############

PATTERN_GROUPS = {
    "MEASURE": MEASURE_PATTERNS,
    "SAMPLE_TIME": SAMPLE_TIME_PATTERNS,
    "DOSE": {"doses": DOSE_PATTERN},
    "WEIGHT": WEIGHT_PATTERNS,
    "AGE": AGE_PATTERNS,
    "N_SUB": N_SUB_PATTERNS,
    "UNITS": {"units": UNIT_SPAN_PATTERN},
    "ROUTE": ROUTE_PATTERNS,
    "SPECIES": SPECIES_PATTERNS,
    "SAMPLE": SAMPLE_TYPE_PATTERNS,
}

def compile_patterns(pattern_groups):
    compiled = []
    for top_label, sub_patterns in pattern_groups.items():
        for sub_label, raw_pattern in sub_patterns.items():
            try:
                #print(f"Compiling pattern: group={top_label}, sub={sub_label}, pattern={raw_pattern}, type={type(raw_pattern)}")
                compiled.append({
                    "label": top_label,
                    "sub": sub_label,
                    "pattern": re.compile(raw_pattern, flags=re.IGNORECASE),
                })
            except re.error as e:
                print(f"❌ Regex compile error in group '{top_label}', pattern '{sub_label}': {raw_pattern}")
                print(f"   → Error: {e}")
                raise
    return compiled

COMPILED_PATTERNS = compile_patterns(PATTERN_GROUPS)
