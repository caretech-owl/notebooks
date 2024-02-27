import glob
from typing import List, Optional
from zipfile import ZipFile

from cassis import Cas, load_cas_from_xmi, load_typesystem


def load_cas_zips(path: str, limit: Optional[int] = None) -> List[Cas]:
    data = []
    for archive in glob.glob(f"{path}/*.zip"):
        data.append(_load_cas_zip(archive))
        if limit is not None and len(data) >= limit:
            break
    for cas in data:
        secs = cas.select("webanno.custom.Sectionsentence")
        for i in range(len(secs)):
            for j in range(i + 1, len(secs)):
                if secs[i].begin >= secs[j].begin:
                    msg = f"Section {secs[i]} should begin before {secs[j]}"
                    raise AssertionError(msg)
    return data


def _load_cas_zip(path: str) -> Cas:
    with ZipFile(path, "r") as archive:
        flist = [f.filename for f in archive.filelist]
        if "TypeSystem.xml" not in flist:
            msg = f"TypeSystem.xml missing from {path}.\nAvailable files:\n{flist}"
            raise IOError(msg)

        data_file = "INITIAL_CAS.xmi"
        if "INITIAL_CAS.xmi" not in flist:
            if "rsari.xmi" not in flist:
                msg = (
                    "INITIAL_CAS.xmi/rsari.xmi missing from"
                    f"{path}.\nAvailable files:\n{flist}"
                )
                raise IOError(msg)
            data_file = "rsari.xmi"

        with archive.open("TypeSystem.xml", "r") as f:
            typesystem = load_typesystem(f)

        with archive.open(data_file, "r") as f:
            cas = load_cas_from_xmi(f, typesystem=typesystem)

    return cas
