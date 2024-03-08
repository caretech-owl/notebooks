import re
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from cassis.cas import Cas

from cto_notebooks.utils.data import despacyfy


class SectionEnum(Enum):
    ANREDE = "Anrede"
    AKTUELL_DIAGNOSEN = "AktuellDiagnosen"
    DIAGNOSEN = "Diagnosen"
    AUFNAHMEMEDIKATION = "AufnahmeMedikation"
    AUR = "AllergienUnverträglichkeitenRisiken"
    ANAMNESE = "Anamnese"
    KU_BEFUNDE = "KUBefunde"
    ECHO_BEFUNDE = "EchoBefunde"
    BEFUNDE = "Befunde"
    LABOR = "Labor"
    ZUSAMMENFASSUNG = "Zusammenfassung"
    MIX = "Mix"
    ENTLASSMEDIKATION = "EntlassMedikation"
    ABSCHLUSS = "Abschluss"


@dataclass
class CardioDeSection:
    section: SectionEnum
    preprocessed: str
    raw: str
    text: str = ""
    summary: Optional[str] = None

    def __repr__(self) -> str:
        return (
            f"CardioDeSection<{self.section}>\n--------\n{self.preprocessed}---------"
        )


@dataclass
class CardioDeData:
    title: str
    is_patientin: bool = True
    fields: Dict[str, str] = field(default_factory=dict)
    sections: List[CardioDeSection] = field(default_factory=list)

    @property
    def section_list(self) -> Counter:
        return Counter(sec.section for sec in self.sections)

    def __repr__(self) -> str:
        s_r = "\n\n".join(repr(sec) for sec in self.sections)
        return (
            f"CardioDeData<{self.title}>\n==========\n"
            f"Fields: {self.fields}\n{s_r}\n==========="
        )


str_replacements = [
    ("Fr.", "B-SALUTE"),
    ("entlassen Frau", "entlassen B-SALUTE"),
    ("B-SALUTE I-ORG", "B-SALUTE B-PER"),
    ("B-SALUTE Die", "B-SALUTE. Die"),
    ("B-SALUTE Des", "B-SALUTE. Des"),
    ("B-SALUTE der I-ORG", "B-SALUTE der B-ORG"),
    ("B-SALUTE geboren", "geboren"),
    ("SALUTE PER", "B-SALUTE B-PER"),
    ("I-TITLE PER", "I-TITLE B-PER"),
    ("i-ORG", "I-ORG"),
    ("(ORG)", "(B-ORG)"),
    ("ISH (I-ORG)", "B-ORG I-ORG"),
    (" ORG", " B-ORG"),
    ("B-ORG B-ORG", "B-ORG I-ORG"),
    ("B-LOC I-ORG", "B-ORG I-ORG"),
    ("(ORG ", "(B-ORG "),
    ("in I-ORG I-ORG", "in B-ORG I-ORG"),
    (
        "B-ORG für Prävention I-ORG I-ORG der I-ORG I-ORG",
        "B-ORG I-ORG I-ORG I-ORG I-ORG der I-ORG I-ORG",
    ),
    ("Europäischen I-ORG I-ORG I-ORG", "B-ORG I-ORG I-ORG I-ORG"),
    ("B-ORG I-PER I-ORG", "B-ORG I-ORG I-ORG"),
    ("Krankenhaus I-ORG", "B-ORG I-ORG"),
    ("Krankenhaus, I-ORG", "B-ORG I-ORG"),
    ("KH I-ORG", "B-ORG I-ORG"),
    ("Radiol. Praxis I-ORG", "B-ORG I-ORG I-ORG"),
    ("Aufnahme von B-SALUTE B-SALUTE", "Aufnahme von B-SALUTE B-PER"),
    ("einer niedergelassenen B-SALUTE", "einer niedergelassenen Kollegin"),
    (" PHONE I-PHONE", " B-PHONE I-PHONE"),
]

reg_replacements = [
    # Some dates are noted in a mixed format (B-DAY B-MONTH <Pseudo year>)
    (r"B-DAY B-MONTH <\[Pseudo\]\s\d{4}>", "B-DAY B-MONTH B-YEAR"),
    # After the above some patient names were leaking
    (r"B-SALUTE\s([A-Z]|\* )[a-z]+", "B-SALUTE B-PER"),
    # Remove unspecific references to colleagues
    (
        r"([Zz]eitnah|[Dd]urch|ihrem|alle|unserer?|niedergell?assenen?|logischen|unseren|der|den|einem|[Dd]ie)\sB-SALUTE",
        r"\1 Kollegen",
    ),
]

pseudo_date_matcher = re.compile(r"(?P<date><\[Pseudo\][^>]+>)")
patient_matcher = re.compile(r"B-SALUTE B-PER(\sI-PER)?")
person_matcher = re.compile(
    r"(?P<person>(B-SALUTE\s{,2}){0,3}(B-PER )?(B-TITLE(\sI-TITLE)*\s)?(B-PER(\sI-PER)*))"  # noqa: E501
)
org_matcher = re.compile(
    r"(?P<org>B-ORG((\s(in|der|-|am))*\sI-ORG)*(\s\(B-ORG(\sI-ORG)*\))?)"
)
date_matcher = re.compile(
    r"(?P<date>(B-DATE|B-DAY(\sB-MONTH(\sB-YEAR)?)?|B-MONTH(\sB-YEAR)?|B-YEAR))"
)
location_matcher = re.compile(r"(?P<org>B-LOC(\sI-LOC)*)")
address_matcher = re.compile(r"(?P<address>B-PLZ(\sB-LOC){,2}(\sB-ADDR)?(\sI-ADDR)*)")
phone_matcher = re.compile(r"(?P<phone>B-PHONE(\sI-PHONE)*)")


class Placeholder:
    def __init__(self, prefix: str, field_list: List) -> None:
        self.prefix = prefix
        self.cnt = -1
        self.field_list = field_list

    def __call__(self, _: re.Match) -> str:
        self.cnt += 1
        field_name = f"{self.prefix}_{self.cnt}"
        self.field_list[field_name] = ""
        return f"{{{field_name}}}"


def generate_data_from_cas(cas: Cas) -> CardioDeData:
    data = CardioDeData(title=cas.get_document_annotation().documentTitle)
    secs = cas.select("webanno.custom.Sectionsentence")
    n = len(secs)
    s = cas.get_sofa().sofaString
    recorded_dates = {}

    for i in range(n):
        raw = s[secs[i].begin : secs[i + 1].begin] if i < n - 1 else s[secs[i].begin :]
        entry = despacyfy(raw)
        if secs[i].Sectiontypes == SectionEnum.ANREDE.value:
            data.is_patientin = "Ihre Patientin" in entry
        for f, t in str_replacements:
            entry = entry.replace(f, t)
        for r, t in reg_replacements:
            entry = re.sub(r, t, entry)

        entry = re.sub(date_matcher, Placeholder("date", data.fields), entry)
        date_cnt = len(data.fields)
        # Collect, unify and replace date annotations
        for match in pseudo_date_matcher.finditer(entry):
            date_orig = match.group("date")
            date_str = date_orig.replace(":", "")
            date_str = re.sub(r"/(\d\d)>", r"/20\1>", date_str)
            if date_str not in recorded_dates:
                recorded_dates[date_str] = f"date_{date_cnt}"
                date_cnt += 1
            if date_orig != date_str and date_orig not in recorded_dates:
                recorded_dates[date_orig] = recorded_dates[date_str]
        for date_str, variable in recorded_dates.items():
            entry = entry.replace(date_str, f"{{{variable}}}")

        # replace the patient pattern with a placeholder
        entry = re.sub(patient_matcher, f"{{patient_salute}} {{patient_name}}", entry)
        entry = re.sub(person_matcher, Placeholder("person", data.fields), entry)
        entry = re.sub(org_matcher, Placeholder("org", data.fields), entry)
        entry = re.sub(address_matcher, Placeholder("address", data.fields), entry)
        entry = re.sub(location_matcher, Placeholder("loc", data.fields), entry)
        entry = re.sub(phone_matcher, Placeholder("phone", data.fields), entry)

        data.fields["patient_name"] = ""
        data.fields["patient_salute"] = ""
        data.sections.append(
            CardioDeSection(
                section=SectionEnum(secs[i].Sectiontypes), preprocessed=entry, raw=raw
            )
        )
    for v in recorded_dates.values():
        data.fields[v] = ""
    return data
