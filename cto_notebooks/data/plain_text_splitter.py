# %%
from typing import List

from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_plain_text(txt: str) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=25)
    res = []
    txts = splitter.split_text(txt)
    for substring in txts:
        pos = txt.index(substring)
        col = 0
        line = 0
        for c in txt[:pos]:
            if c == "\n":
                line += 1
                col = 0
            else:
                col += 1
        res.append(
            Document(page_content=substring, metadata={"position": f"{line}:{col}"})
        )
    return res
